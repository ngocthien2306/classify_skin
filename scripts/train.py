import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch 
import pandas as pd
from torch.utils.data import DataLoader

from src.data.dataset import HAM10000Dataset, SkinLesionDataset
from src.data.transforms import get_train_transforms, get_test_transforms, get_val_transforms
from src.models import get_model
from src.training import get_optimizer, get_scheduler, get_criterion, Trainer
from src.utils import get_logger
from src.visualization import Visualizer

logger = get_logger(__name__)

def main():
    # Load config
    with open('config/config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device
    logger.info(f'Using device: {device}')
    
    # Create output directories
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['visualization']['output_dir'], exist_ok=True)
    
    if config['paths']['mode'] == 'csv':
        # Load data
        train_df = pd.read_csv(config['paths']['train_metadata'])
        test_df = pd.read_csv(config['paths']['test_metadata'])
        
        # Create datasets
        train_dataset = HAM10000Dataset(
            train_df,
            config['data']['augmented_dir'],
            transform=get_train_transforms(config)
        )
        test_dataset = HAM10000Dataset(
            test_df,
            config['data']['augmented_dir'],
            transform=get_val_transforms(config)
        )
    
    else:
        train_dataset = SkinLesionDataset(
            root_dir=config['paths']['root_path'],
            split='train',
            transform=get_train_transforms(config)
        )
        
        test_dataset = SkinLesionDataset(
            root_dir=config['paths']['root_path'],
            split='valid', 
            transform=get_val_transforms(config)
        )
        
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
        
        
    # Calculate class weights if needed
    if config['training']['use_class_weights']:
        class_weights = train_dataset.get_class_weights()
    else:
        class_weights = None
    
    # Initialize model, optimizer, scheduler, criterion
    model = get_model(config, device)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    criterion = get_criterion(config, class_weights)
    
    # Initialize trainer and visualizer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device
    )
    visualizer = Visualizer(config)
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(test_loader, test_loader)
    
    # Plot training history
    visualizer.plot_training_history(
        history,
        save_path=os.path.join(config['visualization']['output_dir'], 'training_history.png')
    )
    
    logger.info("Training completed!")

if __name__ == '__main__':
    main()