import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import cv2
from pathlib import Path
from ..utils.logger import get_logger

logger = get_logger(__name__)

class Visualizer:
    """Class for visualization utilities"""
    
    def __init__(self, config):
        self.config = config
        self.classes = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        plt.style.use('seaborn')
        
    def plot_training_history(self, history, save_path=None):
        """
        Plot training history
        
        Args:
            history (dict): Training history containing losses and accuracies
            save_path (str, optional): Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.set_title('Training and Validation Loss')
        
        # Plot accuracy
        ax2.plot(history['train_acc'], label='Train Accuracy')
        ax2.plot(history['val_acc'], label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.set_title('Training and Validation Accuracy')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Training history plot saved to {save_path}")
        plt.show()
        
    def plot_confusion_matrix(self, confusion_matrix, save_path=None):
        """
        Plot confusion matrix
        
        Args:
            confusion_matrix (np.array): Confusion matrix
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix, 
                   annot=True, 
                   fmt='d',
                   cmap='Blues',
                   xticklabels=self.classes,
                   yticklabels=self.classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Confusion matrix plot saved to {save_path}")
        plt.show()
        
    def plot_class_distribution(self, df, save_path=None):
        """
        Plot class distribution
        
        Args:
            df (pd.DataFrame): DataFrame containing class labels
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        class_counts = [df[cls].sum() for cls in self.classes]
        
        bars = plt.bar(self.classes, class_counts)
        plt.title('Class Distribution')
        plt.xlabel('Classes')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Class distribution plot saved to {save_path}")
        plt.show()
        
    def plot_augmentation_examples(self, original_img_path, augmented_paths, save_path=None):
        """
        Plot original image with its augmented versions
        
        Args:
            original_img_path (str): Path to original image
            augmented_paths (list): List of paths to augmented images
            save_path (str, optional): Path to save the plot
        """
        n_images = len(augmented_paths) + 1
        fig, axes = plt.subplots(1, n_images, figsize=(4*n_images, 4))
        
        # Plot original image
        original = cv2.imread(original_img_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        axes[0].imshow(original)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Plot augmented images
        for i, path in enumerate(augmented_paths, 1):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i].imshow(img)
            axes[i].set_title(f'Augmented {i}')
            axes[i].axis('off')
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Augmentation examples plot saved to {save_path}")
        plt.show()
        
    def plot_model_predictions(self, model, images, true_labels, device, save_path=None):
        """
        Plot model predictions vs true labels
        
        Args:
            model (nn.Module): Trained model
            images (torch.Tensor): Batch of images
            true_labels (torch.Tensor): True labels
            device (torch.device): Device to run model on
            save_path (str, optional): Path to save the plot
        """
        model.eval()
        with torch.no_grad():
            outputs = model(images.to(device))
            _, predicted = outputs.max(1)
            
        # Convert tensors to numpy
        images = images.cpu().numpy()
        predicted = predicted.cpu().numpy()
        true_labels = true_labels.cpu().numpy()
        
        # Plot images with their predictions
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        axes = axes.ravel()
        
        for idx, ax in enumerate(axes):
            if idx < len(images):
                # Denormalize image
                img = np.transpose(images[idx], (1, 2, 0))
                img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
                img = np.clip(img, 0, 1)
                
                ax.imshow(img)
                ax.set_title(f'True: {self.classes[true_labels[idx]]}\n' 
                           f'Pred: {self.classes[predicted[idx]]}',
                           color='green' if predicted[idx] == true_labels[idx] else 'red')
                ax.axis('off')
                
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Model predictions plot saved to {save_path}")
        plt.show()

    def plot_learning_rate_finder(self, lrs, losses, save_path=None):
        """
        Plot results from learning rate finder
        
        Args:
            lrs (list): List of learning rates
            losses (list): Corresponding losses
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        plt.semilogx(lrs, losses)
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate vs Loss')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Learning rate finder plot saved to {save_path}")
        plt.show()