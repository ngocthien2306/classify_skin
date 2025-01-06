import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from ..utils.logger import get_logger
from ..utils.utils import one_hot

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
       self.n_classes = config['model']['num_classes']
       
       # Setup dirs
       self.checkpoint_dir = Path(config['training']['checkpoint_dir'])
       self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
       
       # Stats
       self.best_train = 0.0
       self.best_val = 0.0
       self.train_evl_result = None
       self.val_evl_result = None
       
       # Metrics history
       self.metrics = {
           'train_loss': [], 'train_acc': [],
           'val_loss': [], 'val_acc': []
       }
       
   def train_emsac_epoch(self, train_loader, epoch):
       self.model.train()
       running_loss, r_pre = 0., 0.
       batch_idx = 0
       train_tmp_result = torch.zeros(self.n_classes, self.n_classes)
       
       for data, target in tqdm(train_loader, desc='Training'):
           batch_idx += 1
           target_indices = target
           target_one_hot = one_hot(target, length=self.n_classes)
           data = data.to(self.device)
           target_one_hot = target_one_hot.to(self.device)
           
           # Forward
           output = self.model(data)
           loss = self.model.loss(output, target_one_hot, size_average=True)
           
           # Backward
           self.optimizer.zero_grad()
           loss.backward()
           self.optimizer.step()
           
           running_loss += loss.item()
           
           # Calculate acc
           v_mag = torch.sqrt(torch.sum(output**2, dim=2, keepdim=True))
           pred = v_mag.data.max(1, keepdim=True)[1].cpu().squeeze()
           r_pre += pred.eq(target_indices.view_as(pred)).squeeze().sum()
           tmp_pre = r_pre/(batch_idx*train_loader.batch_size)
           
           # Update confusion matrix
           tmp_size = train_loader.batch_size
           if batch_idx % len(train_loader) == 0 and len(train_loader.dataset) % tmp_size != 0:
               tmp_size = len(train_loader.dataset) % tmp_size
               
           for i in range(tmp_size):
               pred_x = pred.numpy()
               train_tmp_result[target_indices[i]][pred_x[i]] += 1
               
           # Save best model during training
           if self.best_train < tmp_pre and tmp_pre >= 0.80:
               torch.save(self.model.state_dict(), 
                        self.checkpoint_dir / 'iter_best.pth')
               
           # Log progress
           if batch_idx % self.config['training']['log_interval'] == 0:
               logger.info(f'Epoch: {epoch} [{batch_idx}/{len(train_loader)}] '
                         f'Loss: {loss.item():.5f} Acc: {tmp_pre:.5f}')
               
       epoch_acc = r_pre / len(train_loader.dataset)
       epoch_loss = running_loss / len(train_loader)
       
       # Save best train model
       if self.best_train < epoch_acc:
           self.best_train = epoch_acc
           self.train_evl_result = train_tmp_result.clone()
           torch.save(self.model.state_dict(),
                     self.checkpoint_dir / 'train_best.pth')
           torch.save(train_tmp_result,
                     self.checkpoint_dir / 'train_conf.pth')
                     
       return epoch_loss, epoch_acc, train_tmp_result
       
   def validate_emsac(self, val_loader, split='val'):
       self.model.eval()
       evl_tmp_result = torch.zeros(self.n_classes, self.n_classes)
       
       with torch.no_grad():
           for batch_idx, (data, target) in enumerate(tqdm(val_loader, desc=split.title())):
               batch_idx += 1
               target_indices = target
               target_one_hot = one_hot(target, length=self.n_classes)
               data = data.to(self.device) 
               target_one_hot = target_one_hot.to(self.device)
               
               output = self.model(data)
               v_mag = torch.sqrt(torch.sum(output**2, dim=2, keepdim=True))
               pred = v_mag.data.max(1, keepdim=True)[1].cpu()
               
               # Update confusion matrix
               tmp_size = val_loader.batch_size
               if batch_idx % len(val_loader) == 0 and len(val_loader.dataset) % tmp_size != 0:
                   tmp_size = len(val_loader.dataset) % tmp_size
                   
               for i in range(tmp_size):
                   pred_y = pred.numpy()
                   evl_tmp_result[target_indices[i]][pred_y[i]] += 1
                   
           # Calculate accuracy
           diag_sum = torch.sum(evl_tmp_result.diagonal())
           all_sum = torch.sum(evl_tmp_result)
           acc = 100. * float(torch.div(diag_sum, all_sum))
           
           # Save best validation model
           if split == 'val' and acc > self.best_val:
               self.best_val = acc
               self.val_evl_result = evl_tmp_result.clone()
               torch.save(self.model.state_dict(), 
                        self.checkpoint_dir / 'val_best.pth')
               torch.save(evl_tmp_result,
                        self.checkpoint_dir / 'val_conf.pth')
                        
           logger.info(f'{split.title()} Acc: {acc:.3f}% '
                      f'Best {split.title()}: {self.best_val:.3f}%')
                  
       return None, acc, evl_tmp_result
       
   def train(self, train_loader, val_loader):
        logger.info('Starting training...')
        
        for epoch in range(self.epochs):
            # Train
            train_loss, train_acc, train_conf = self.train_emsac_epoch(
                train_loader, epoch)
                
            # Validate  
            val_loss, val_acc, val_conf = self.validate_emsac(
                val_loader, split='val')
               
            # Update learning rate
            if val_loss:
                self.scheduler.step(val_loss) 
            else:
                self.scheduler.step(train_loss)
           
            # Update metrics
            self.metrics['train_loss'].append(train_loss)
            self.metrics['train_acc'].append(train_acc)
            self.metrics['val_loss'].append(val_loss if val_loss else 0)
            self.metrics['val_acc'].append(val_acc)
            
            # Log epoch results
            logger.info(f'Epoch {epoch}: train_loss={train_loss:.5f}, '
                        f'train_acc={train_acc:.5f}, val_acc={val_acc:.5f}')
                        
            # Plot if needed
            if (epoch + 1) % self.config['training']['plot_interval'] == 0:
                self.plot_progress()
                
        return self.metrics
        
   def plot_progress(self):
       plt.figure(figsize=(12,4))
       
       plt.subplot(121)
       plt.plot(self.metrics['train_loss'], label='Train')
       plt.title('Loss vs Epoch')
       plt.xlabel('Epoch')
       plt.ylabel('Loss') 
       plt.legend()
       
       plt.subplot(122)
       plt.plot(self.metrics['train_acc'], label='Train')
       plt.plot(self.metrics['val_acc'], label='Val')
       plt.title('Accuracy vs Epoch')
       plt.xlabel('Epoch')
       plt.ylabel('Accuracy (%)')
       plt.legend()
       
       plt.tight_layout()
       plt.savefig(self.checkpoint_dir / 'training_progress.png')
       plt.close()