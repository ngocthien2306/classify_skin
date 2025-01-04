from .resnet import SkinLesionModel 
from .emsac import EMSACNet

def get_model(config, device):
    """
    Factory function to create models
    
    Args:
        config (dict): Configuration dictionary
        device (torch.device): Device to put the model on
        
    Returns:
        model: Initialized model
    """
    model_name = config['model']['name']
    
    if model_name == 'emsac':
        model = EMSACNet(num_classes=config['model']['num_classes'])
    elif 'resnet' in model_name:
        model = SkinLesionModel(config)
    else:
        raise ValueError(f"Model {model_name} not supported")
        
    return model.to(device)