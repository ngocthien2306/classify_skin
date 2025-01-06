from .resnet import SkinLesionModel 
from .emsac import EMSACNet
from .emsac_fixcaps import EMSAC_FixCapsNet
from torchsummary import summary
from src.utils.logger import get_logger

logger = get_logger(__name__)

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
    elif model_name == 'emsac_cap':
        model = EMSAC_FixCapsNet(
            conv_inputs=config['model']['n_channels'],
            conv_outputs=config['model']['conv_outputs'],
            primary_units=config['model']['num_primary_units'],
            primary_unit_size=config['model']['primary_unit_size'],
            num_classes=config['model']['num_classes'],
            output_unit_size=config['model']['output_unit_size'],
            init_weights=config['model']['init_weights'],
            mode=config['model']['mode']
        )
        model = model.to(device)
        summary(model,(config['model']['n_channels'], config['model']['img_size'], config['model']['img_size']))
        logger.info(summary(model,(config['model']['n_channels'], config['model']['img_size'], config['model']['img_size'])))
        
    elif 'resnet' in model_name:
        model = SkinLesionModel(config)
    else:
        raise ValueError(f"Model {model_name} not supported")
        
    return model.to(device)