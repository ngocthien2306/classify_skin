import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .base_model import BaseModel

class CapsuleLayer(nn.Module):
    """Capsule layer with routing"""
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, routing_iterations=3):
        super(CapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.num_route_nodes = num_route_nodes  
        self.routing_iterations = routing_iterations
        
        # Weight matrix to transform from in_channels to out_channels
        self.W = nn.Parameter(torch.randn(1, num_route_nodes, num_capsules, out_channels, in_channels))
        
    def squash(self, tensor):
        """Squashing function to scale vectors"""
        squared_norm = (tensor ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm + 1e-8)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(2).unsqueeze(4)
        
        # Calculate u_hat by matrix multiplication of W and input x
        u_hat = torch.matmul(self.W, x).squeeze(-1)  # [batch, routes, caps, out_channels]
        
        # Initialize coupling coefficients
        b = torch.zeros(batch_size, self.num_route_nodes, self.num_capsules, 1).to(x.device)
        
        # Dynamic Routing
        for _ in range(self.routing_iterations):
            c = F.softmax(b, dim=2)  # [batch, routes, caps, 1]
            s = (c * u_hat).sum(dim=1, keepdim=True)  # [batch, 1, caps, out_channels]
            v = self.squash(s)  # [batch, 1, caps, out_channels]
            
            if _ < self.routing_iterations - 1:
                b = b + (u_hat * v).sum(dim=-1, keepdim=True)  # [batch, routes, caps, 1]
        
        return v.squeeze(1)  # [batch, caps, out_channels]

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
    def __init__(self, num_classes=7):
        super(EMSACNet, self).__init__()
        
        # Feature extraction with large kernel
        self.conv1 = nn.Conv2d(3, 64, kernel_size=31, stride=2, padding=15)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # First stage attention
        self.attention1 = MultiStageAttention(64)
        
        # Primary capsules path
        self.primary_caps = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=9, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Second stage attention
        self.attention2 = MultiStageAttention(256)
        
        # Convert to capsules
        self.conv_caps = nn.Conv2d(256, 32*8, kernel_size=3, stride=2, padding=1)
        
        # Calculate routes for digit capsules (based on feature map size)
        self.num_routes = 27 * 27 * 32  # Based on conv_caps output size
        
        # Digit capsules
        self.digit_caps = CapsuleLayer(
            num_capsules=num_classes,  # 7 classes
            num_route_nodes=self.num_routes, 
            in_channels=8,  # Input capsule dimension
            out_channels=16,  # Output capsule dimension
            routing_iterations=3
        )
    
    def squash(self, x, dim=-1):
        """Squashing function to scale vectors"""
        squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * x / torch.sqrt(squared_norm + 1e-8)
        
    def forward(self, x):
        # print("Input shape:", x.shape)
        
        # Feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print("After conv1:", x.shape)
        
        # First attention
        x = self.attention1(x)
        # print("After attention1:", x.shape)
        
        # Primary capsules
        x = self.primary_caps(x)
        # print("After primary caps:", x.shape)
        
        # Second attention
        x = self.attention2(x)
        # print("After attention2:", x.shape)
        
        # Convert to capsules 
        x = self.conv_caps(x)
        # print("After conv caps:", x.shape)
        
        # Reshape for digit capsules
        x = x.view(x.size(0), -1, 8)  # [batch_size, routes, capsule_dim]
        # print("Before digit caps:", x.shape)
        
        # Digit capsules
        x = self.digit_caps(x)  # [batch_size, num_classes, 16] 
        # print("After digit caps:", x.shape)
        
        # Calculate class probabilities
        classes = torch.sqrt((x ** 2).sum(2))  # [batch_size, num_classes]
        # print("Final output:", classes.shape)
        
        return classes
    
    def loss(self, x, target, m_plus=0.9, m_minus=0.1, lambda_=0.5):
        """
        Custom margin loss for capsule network.
        
        Args:
            x: Output from the network (class probabilities)
            target: True labels
            m_plus: The margin for positive cases (default: 0.9)
            m_minus: The margin for negative cases (default: 0.1)
            lambda_: Down-weighting of the loss for absent digits (default: 0.5)
            
        Returns:
            Total loss value
        """
        # Convert target to one-hot encoding
        target = F.one_hot(target, num_classes=self.num_classes).float()
        
        # Compute the basic loss for each dimension
        losses = target * F.relu(m_plus - x) ** 2 + \
                lambda_ * (1.0 - target) * F.relu(x - m_minus) ** 2
                
        # Sum over the digit axis
        losses = losses.sum(dim=1)
        
        # Average over the batch
        return losses.mean()