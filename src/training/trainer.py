import torch
import time
from tqdm import tqdm
from .metrics import calculate_metrics
from ..utils.logger import get_logger
from src.utils.utils import ImageShow,draw_size_acc,one_hot
import os
logger = get_logger(__name__)

class Trainer:
   def __init__(self, model, criterion, optimizer, scheduler, config, device):
       self.model = model
       self.criterion = criterion 
       self.optimizer = optimizer
       self.scheduler = scheduler
       self.config = config
       self.device = device
       self.epochs = config['training']['epochs']
       self.best_val_acc = 0.0
       self.n_classes = config['model']['num_classes']

   def train_emsac_epoch(self, train_loader):
       """Training epoch for EMS-ACNet"""
       self.model.train()
       running_loss = 0.
       r_pre = 0.
       train_tmp_result = torch.zeros(self.n_classes, self.n_classes)
       batch_size = train_loader.batch_size
       
       for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            current_batch_size = data.size(0)
            target_indices = target
            target_one_hot = one_hot(target, length=self.n_classes)
            data, target = data.to(self.device), target_one_hot.to(self.device)

            # Forward pass
            output = self.model(data)
            loss = self.model.loss(output, target, size_average=True)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            v_mag = torch.sqrt(torch.sum(output**2, dim=2, keepdim=True))
            pred = v_mag.data.max(1, keepdim=True)[1].cpu().squeeze()
            r_pre += pred.eq(target_indices.view_as(pred)).squeeze().sum()
            
            for i in range(current_batch_size):
                pred_x = pred.numpy()
                train_tmp_result[target_indices[i]][pred_x[i]] += 1
               
       epoch_acc = r_pre / len(train_loader.dataset)
       epoch_loss = running_loss / len(train_loader)
       
       return epoch_loss, epoch_acc * 100, train_tmp_result

   def validate_emsac(self, val_loader):
        """Validation for EMS-ACNet"""
        self.model.eval()
        running_loss = 0.
        r_pre = 0.
        val_tmp_result = torch.zeros(self.n_classes, self.n_classes)
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(val_loader)):
                current_batch_size = data.size(0)
                target_indices = target
                target_one_hot = one_hot(target, length=self.n_classes)
                data, target = data.to(self.device), target_one_hot.to(self.device)

                # Forward pass
                output = self.model(data)
                loss = self.model.loss(output, target, size_average=True)
                running_loss += loss.item()
                
                # Calculate accuracy
                v_mag = torch.sqrt(torch.sum(output**2, dim=2, keepdim=True))
                pred = v_mag.data.max(1, keepdim=True)[1].cpu()
                r_pre += pred.eq(target_indices.view_as(pred)).squeeze().sum()

                # Update confusion matrix
                for i in range(current_batch_size):
                    pred_x = pred.numpy()
                    val_tmp_result[target_indices[i]][pred_x[i]] += 1

        val_acc = r_pre / len(val_loader.dataset)
        val_loss = running_loss / len(val_loader)
        
        return val_loss, val_acc * 100, val_tmp_result

   def train_standard_epoch(self, train_loader):
       """Standard training epoch"""
       self.model.train()
       running_loss = 0.0
       correct = 0
       total = 0
       
       for images, labels in tqdm(train_loader):
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
           
       epoch_loss = running_loss / len(train_loader)
       epoch_acc = 100. * correct / total
       return epoch_loss, epoch_acc, None

   def validate_standard(self, val_loader):
       """Standard validation"""
       self.model.eval()
       running_loss = 0.0
       correct = 0
       total = 0
       
       with torch.no_grad():
           for images, labels in tqdm(val_loader):
               images, labels = images.to(self.device), labels.to(self.device)
               
               outputs = self.model(images)
               loss = self.criterion(outputs, labels)
               
               running_loss += loss.item()
               _, predicted = outputs.max(1)
               total += labels.size(0)
               correct += predicted.eq(labels).sum().item()
               
       val_loss = running_loss / len(val_loader)
       val_acc = 100. * correct / total
       return val_loss, val_acc, None

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
           if self.config['model']['name'] == 'emsac_cap':
               train_loss, train_acc, train_conf_matrix = self.train_emsac_epoch(train_loader)
               val_loss, val_acc, val_conf_matrix = self.validate_emsac(val_loader)
               
               # Save best model based on validation accuracy
               if val_acc > self.best_val_acc:
                   self.best_val_acc = val_acc 
                   self.save_emsac_checkpoint(
                       epoch, val_conf_matrix,
                       f"{self.config['training']['checkpoint_dir']}/best_model.pth"
                   )
                   
               # Save confusion matrices
               if epoch % self.config['training']['save_interval'] == 0:
                   save_dir = f"{self.config['training']['checkpoint_dir']}/{epoch}"
                   os.makedirs(save_dir, exist_ok=True)
                   self.save_confusion_matrix(train_conf_matrix, f"{save_dir}/train_conf.pth")
                   self.save_confusion_matrix(val_conf_matrix, f"{save_dir}/val_conf.pth")
           else:
               train_loss, train_acc, _ = self.train_standard_epoch(train_loader)
               val_loss, val_acc, _ = self.validate_standard(val_loader)
               
               if val_acc > self.best_val_acc:
                   self.best_val_acc = val_acc
                   self.save_checkpoint(epoch, 'best_model.pth')
           
           # Update learning rate
           self.scheduler.step(val_loss)
           
           # Update history
           history['train_loss'].append(train_loss)
           history['train_acc'].append(train_acc)
           history['val_loss'].append(val_loss)
           history['val_acc'].append(val_acc)
           
           # Log results
           logger.info(f'Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}%')
           logger.info(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%')
           
           # Calculate metrics every N epochs
           if (epoch + 1) % self.config['training']['metrics_interval'] == 0:
               metrics = calculate_metrics(self.model, val_loader, self.device)
               logger.info(f"Per-class metrics:\n{metrics}")
               
       return history

   def save_checkpoint(self, epoch, filename):
       """Save standard model checkpoint"""
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

   def save_emsac_checkpoint(self, epoch, conf_matrix, filename):
       """Save EMS-ACNet checkpoint with confusion matrix"""
       checkpoint = {
           'epoch': epoch,
           'model_state_dict': self.model.state_dict(),
           'optimizer_state_dict': self.optimizer.state_dict(), 
           'scheduler_state_dict': self.scheduler.state_dict(),
           'best_val_acc': self.best_val_acc,
           'confusion_matrix': conf_matrix,
           'config': self.config
       }
       torch.save(checkpoint, filename)
       logger.info(f"EMS-ACNet checkpoint saved: {filename}")

   def save_confusion_matrix(self, conf_matrix, filename):
       """Save confusion matrix"""
       torch.save(conf_matrix, filename)
       logger.info(f"Confusion matrix saved: {filename}")