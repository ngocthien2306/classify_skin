import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.base_model import BaseModel

class CapsuleLayer(nn.Module):
    """Capsule layer with routing"""
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None, routing_iterations=3):
        super(CapsuleLayer, self).__init__()
        self.num_route_nodes = num_route_nodes
        self.num_capsules = num_capsules
        self.routing_iterations = routing_iterations
        
        if kernel_size is not None:
            self.conv = nn.Conv2d(in_channels, num_capsules * out_channels, kernel_size, stride, padding=1)
        else:
            self.W = nn.Parameter(torch.randn(1, num_route_nodes, num_capsules, out_channels, in_channels))

    def forward(self, x):
        if hasattr(self, 'conv'):
            # For primary capsules
            u = self.conv(x)
            u = u.view(x.size(0), self.num_capsules, -1)
        else:
            # For routing capsules
            batch_size = x.size(0)
            x = x.unsqueeze(2).unsqueeze(4)
            W = self.W
            
            u_hat = torch.matmul(W, x)
            b = torch.zeros(batch_size, self.num_route_nodes, self.num_capsules, 1).to(x.device)
            
            for _ in range(self.routing_iterations):
                c = F.softmax(b, dim=2)
                s = (c * u_hat).sum(dim=1, keepdim=True)
                v = self.squash(s)
                if _ < self.routing_iterations - 1:
                    b = b + (u_hat * v).sum(dim=-1, keepdim=True)
            
            return v.squeeze(1)
        
        return self.squash(u)
    
    def squash(self, s):
        """Squashing function to scale vectors"""
        squared_norm = (s ** 2).sum(-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm) / torch.sqrt(squared_norm + 1e-8)
        return scale * s

class ChannelAttention(nn.Module):
    """Channel attention module"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return torch.sigmoid(out).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    """Spatial attention module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(x))

class MultiStageAttention(nn.Module):
    """Multi-stage attention module combining channel and spatial attention"""
    def __init__(self, channels):
        super(MultiStageAttention, self).__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
        
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class EMSACNet(BaseModel):
    """Enhanced Multi-Stage Attention-Capsule Network"""
    def __init__(self, num_classes=7):
        super(EMSACNet, self).__init__()
        
        # Feature extraction module with large kernel (31x31)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=31, stride=2, padding=15)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # First stage attention
        self.attention1 = MultiStageAttention(64)
        
        # Primary capsules
        self.primary_caps = CapsuleLayer(
            num_capsules=32,
            num_route_nodes=-1,  # Not used for primary capsules
            in_channels=64,
            out_channels=8,
            kernel_size=9,
            stride=2
        )
        
        # Secondary stage attention
        self.attention2 = MultiStageAttention(32*8)
        
        # Routing capsules
        self.digit_caps = CapsuleLayer(
            num_capsules=num_classes,
            num_route_nodes=32*6*6,  # Calculated based on input size
            in_channels=8,
            out_channels=16,
            routing_iterations=3
        )
        
    def forward(self, x):
        # Feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # First attention
        x = self.attention1(x)
        
        # Primary capsules
        x = self.primary_caps(x)
        
        # Reshape for attention
        b, c, h, w = x.shape
        x = x.view(b, -1, h, w)
        
        # Second attention
        x = self.attention2(x)
        
        # Reshape back for routing capsules
        x = x.view(b, -1, 8)
        
        # Digit capsules
        x = self.digit_caps(x)
        
        # Calculate class probabilities
        classes = torch.sqrt((x ** 2).sum(2))
        
        return classes
    
    def loss(self, data, target, m_plus=0.9, m_minus=0.1, lambda_=0.5):
        """Custom loss function combining margin loss and reconstruction loss"""
        target = F.one_hot(target, num_classes=7)
        
        # Margin loss
        v_c = torch.sqrt((data ** 2).sum(dim=2, keepdim=True))
        
        max_l = F.relu(m_plus - v_c).view(data.size(0), -1)
        max_r = F.relu(v_c - m_minus).view(data.size(0), -1)
        
        loss_l = target * max_l ** 2
        loss_r = lambda_ * (1.0 - target) * max_r ** 2
        margin_loss = loss_l + loss_r
        margin_loss = margin_loss.sum(dim=1).mean()
        
        return margin_loss