import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import pandas as pd
from torch.utils.data import DataLoader

from src.data.dataset import HAM10000Dataset
from src.data.transforms import get_test_transforms
from src.models import get_model
from src.training import calculate_metrics
from src.utils import get_logger
from src.visualization import Visualizer

logger = get_logger(__name__)

def load_checkpoint(checkpoint_path, model, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['best_val_acc']

def evaluate(model, test_loader, device, visualizer, config):
    """Evaluate model on test set"""
    model.eval()
    all_images = []
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predictions = outputs.max(1)
            
            # Save first batch for visualization
            if batch_idx == 0:
                visualizer.plot_model_predictions(
                    model, images, labels, device,
                    save_path=os.path.join(config['visualization']['output_dir'], 'predictions.png')
                )
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
    
    # Calculate and print metrics
    metrics = calculate_metrics(model, test_loader, device)
    logger.info("\nClassification Report:")
    logger.info(f"\n{metrics['classification_report']}")
    
    # Plot confusion matrix
    visualizer.plot_confusion_matrix(
        metrics['confusion_matrix'],
        save_path=os.path.join(config['visualization']['output_dir'], 'confusion_matrix.png')
    )
    
    return metrics

def main():
    # Load config
    with open('config/config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Load test data
    test_df = pd.read_csv(config['paths']['test_metadata'])
    
    # Create test dataset and dataloader
    test_dataset = HAM10000Dataset(
        test_df,
        config['data']['augmented_dir'],
        transform=get_test_transforms(config)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    # Initialize model
    model = get_model(config, device)
    
    # Load best checkpoint
    checkpoint_path = os.path.join(config['training']['checkpoint_dir'], 'best_model.pth')
    model, best_val_acc = load_checkpoint(checkpoint_path, model, device)
    logger.info(f"Loaded checkpoint with validation accuracy: {best_val_acc:.2f}%")
    
    # Initialize visualizer
    visualizer = Visualizer(config)
    
    # Evaluate model
    logger.info("Starting evaluation...")
    metrics = evaluate(model, test_loader, device, visualizer, config)
    
    logger.info("Evaluation completed!")

if __name__ == '__main__':
    main()