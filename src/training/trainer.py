import torch
import time
from tqdm import tqdm
from .metrics import calculate_metrics
from ..utils.logger import get_logger

logger = get_logger(__name__)

class Trainer:
    """
    Trainer class for model training
    """
    def __init__(self, model, criterion, optimizer, scheduler, config, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.epochs = config['training']['epochs']
        self.best_val_acc = 0.0
        
    def train_epoch(self, train_loader):
        """Single epoch training"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss/(batch_idx+1),
                'acc': 100.*correct/total
            })
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """Validation step"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': running_loss/(batch_idx+1),
                    'acc': 100.*correct/total
                })
                
        val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total
        return val_loss, val_acc
    
    def train(self, train_loader, val_loader):
        """Full training loop"""
        logger.info("Starting training...")
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        for epoch in range(self.epochs):
            logger.info(f'\nEpoch {epoch+1}/{self.epochs}')
            
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_acc = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, 'best_model.pth')
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Log results
            logger.info(f'Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}%')
            logger.info(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%')
            
            # Calculate per-class metrics every 5 epochs
            if (epoch + 1) % 5 == 0:
                metrics = calculate_metrics(self.model, val_loader, self.device)
                logger.info(f"Per-class metrics:\n{metrics}")
                
        return history
    
    def save_checkpoint(self, epoch, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        
        checkpoint_path = f"{self.config['training']['checkpoint_dir']}/{filename}"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")