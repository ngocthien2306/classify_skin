import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import seaborn as sns
import time

from src.utils.utils import ImageShow,draw_size_acc,one_hot
from src.utils.utils import confusion_matrix,metrics_scores,pff

from src.models.emsac_fixcaps import EMSAC_FixCapsNet
# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class Config:
    """Configuration class for evaluation parameters"""
    def __init__(self):
        # Data settings
        self.img_size = 299
        self.batch_size = 32
        self.num_workers = 4
        self.test_transforms = transforms.Compose([
            transforms.Resize((312, 312)),
            transforms.CenterCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Model settings  
        self.n_classes = 7
        self.class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        
        # Paths
        self.data_dir = 'dataset/test'
        self.model_path = '/root/nguyen/research/ML/classify_skin/notebooks/tmp/HAM10000/0107_001107/best_HAM10000_0107_001107.pth'
        self.results_dir = f'./results/evaluation_{time.strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(self.results_dir, exist_ok=True)

def load_model(model_path, model_class, config):
    """Load trained model"""
    try:
        state_dict = torch.load(model_path)
        
        # Remove profiling keys if present
        filtered_state_dict = {
            k: v for k, v in state_dict.items() 
            if 'total_ops' not in k and 'total_params' not in k
        }
        
        # Initialize model
        model = model_class(
            conv_inputs=3,
            conv_outputs=128,
            primary_units=8,
            primary_unit_size=16 * 6 * 6,
            num_classes=config.n_classes,
            output_unit_size=16,
            init_weights=True,
            mode='128'
        ).to(device)
        
        # Load weights
        model.load_state_dict(filtered_state_dict)
        print("Model loaded successfully")
        return model
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def get_test_loader(config):
    """Create test data loader"""
    test_dataset = datasets.ImageFolder(
        root=config.data_dir,
        transform=config.test_transforms
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    return test_loader

def evaluate_model(model, test_loader, config):
    """Evaluate model and compute metrics"""
    model.eval()
    confusion_matrix = torch.zeros(config.n_classes, config.n_classes)
    
    # Storage for predictions
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    print("\nEvaluating model...")
    with torch.no_grad():
        for data, targets in tqdm(test_loader):
            # Move data to device
            data = data.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(data)
            v_mag = torch.sqrt(torch.sum(outputs**2, dim=2, keepdim=True))
            probabilities = F.softmax(v_mag.squeeze(), dim=1)
            predictions = v_mag.data.max(1, keepdim=True)[1].cpu()
            
            # Update confusion matrix
            for t, p in zip(targets.cpu().view(-1), predictions.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
                
            # Store predictions
            all_predictions.extend(predictions.numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return {
        'confusion_matrix': confusion_matrix,
        'predictions': np.array(all_predictions),
        'targets': np.array(all_targets),
        'probabilities': np.array(all_probabilities)
    }

def calculate_metrics(results, config): 
    """Calculate evaluation metrics including F1-score"""
    confusion_matrix = results['confusion_matrix']
    
    # Overall accuracy
    accuracy = 100. * confusion_matrix.diag().sum() / confusion_matrix.sum()
    
    # Per-class metrics
    per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
    per_class_precision = confusion_matrix.diag() / confusion_matrix.sum(0)
    per_class_recall = confusion_matrix.diag() / confusion_matrix.sum(1)
    
    # Calculate F1 score
    # F1 = 2 * (precision * recall) / (precision + recall)
    per_class_f1 = 2 * (per_class_precision * per_class_recall) / (per_class_precision + per_class_recall)
    
    # Handle division by zero cases
    per_class_f1 = torch.nan_to_num(per_class_f1, 0)  # Replace NaN with 0
    
    return {
        'accuracy': accuracy,
        'per_class_accuracy': per_class_accuracy,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1
    }

def plot_results(results, metrics, config):
    """Plot and save evaluation results with enhanced visualization"""
    
    # Set figure size
    plt.figure(figsize=(12, 8))
    
    # Convert confusion matrix to integers
    conf_matrix = results['confusion_matrix'].numpy().astype(int)
    
    # Create heatmap with improved styling
    sns.heatmap(
        conf_matrix,
        annot=True,                    # Show numbers in cells
        fmt='d',                       # Use integer format for display
        cmap='Blues',                  # Use blue color palette
        cbar_kws={'label': 'Count'},   # Add colorbar label
        square=True,                   # Make cells square
        xticklabels=config.class_names,
        yticklabels=config.class_names,
        annot_kws={'size': 10, 'weight': 'bold'}  # Make numbers bold
    )

    # Customize title and labels
    plt.title('Confusion Matrix', size=14, pad=20, weight='bold')
    plt.xlabel('Predicted Label', size=12, labelpad=10)
    plt.ylabel('True Label', size=12, labelpad=10)

    # Rotate x-axis labels for better readability 
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save plot with high resolution
    plt.savefig(
        f'{config.results_dir}/confusion_matrix.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()
    
    # Print metrics
    print(f"\nOverall Accuracy: {metrics['accuracy']:.2f}%")
    print("\nPer-class Performance:")
    print(f"{'Class':10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 60)  # Increased length for new column
    
    # Calculate average F1 score
    avg_f1 = metrics['per_class_f1'].mean() * 100
    
    for i, class_name in enumerate(config.class_names):
        print(f"{class_name:10} "
              f"{metrics['per_class_accuracy'][i]*100:10.2f}% "
              f"{metrics['per_class_precision'][i]*100:10.2f}% "
              f"{metrics['per_class_recall'][i]*100:10.2f}% "
              f"{metrics['per_class_f1'][i]*100:10.2f}%")
    
    print("-" * 60)
    print(f"Average F1-Score: {avg_f1:.2f}%")
              
def save_results(results, metrics, config):
    """Save evaluation results"""
    torch.save({
        'confusion_matrix': results['confusion_matrix'],
        'predictions': results['predictions'],
        'targets': results['targets'],
        'probabilities': results['probabilities'],
        'metrics': metrics
    }, f'{config.results_dir}/evaluation_results.pth')

def main():
    # Initialize configuration
    config = Config()
    
    # Load model
    model = load_model(config.model_path, EMSAC_FixCapsNet, config)
    
    # Get test data loader
    test_loader = get_test_loader(config)
    
    # Evaluate model
    results = evaluate_model(model, test_loader, config)
    
    # Calculate metrics
    metrics = calculate_metrics(results, config)
    
    # Plot results
    plot_results(results, metrics, config)
    
    # Save results
    save_results(results, metrics, config)
    
if __name__ == '__main__':
    main()
