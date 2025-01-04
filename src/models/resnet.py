import torch.nn as nn
from torchvision import models
from .base_model import BaseModel

class SkinLesionModel(BaseModel):
    """Modified ResNet model for skin lesion classification"""
    
    def __init__(self, config):
        """
        Args:
            config (dict): Configuration dictionary
        """
        super(SkinLesionModel, self).__init__()
        
        # Load pre-trained ResNet
        if config['model']['name'] == 'resnet18':
            self.backbone = models.resnet18(pretrained=config['model']['pretrained'])
        elif config['model']['name'] == 'resnet34':
            self.backbone = models.resnet34(pretrained=config['model']['pretrained'])
        elif config['model']['name'] == 'resnet50':
            self.backbone = models.resnet50(pretrained=config['model']['pretrained'])
        else:
            raise ValueError(f"Unsupported model: {config['model']['name']}")

        # Get number of features in last layer
        num_features = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, config['model']['num_classes'])
        )
        
        # Initialize the new layers
        for m in self.backbone.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        return self.backbone(x)