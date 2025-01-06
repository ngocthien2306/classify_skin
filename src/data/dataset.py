import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from pathlib import Path

class HAM10000Dataset(Dataset):
    """
    HAM10000 dataset class for PyTorch
    """
    def __init__(self, df, image_dir, transform=None):
        """
        Args:
            df (pandas.DataFrame): DataFrame containing image metadata
            image_dir (str): Directory containing images
            transform (callable, optional): Optional transform to be applied on images
        """
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.class_columns = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get image path
        img_name = self.df.iloc[idx]['image']
        img_path = os.path.join(self.image_dir, img_name + '.jpg')
        
        # Get label as class index
        label_onehot = self.df.iloc[idx][self.class_columns].values.astype(np.float32)
        label = np.argmax(label_onehot)
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

    def get_class_weights(self):
        """
        Calculate class weights for imbalanced dataset
        """
        class_counts = [self.df[col].sum() for col in self.class_columns]
        total = sum(class_counts)
        class_weights = [total/count for count in class_counts]
        return torch.FloatTensor(class_weights)
    
class SkinLesionDataset(Dataset):
    """Dataset class for skin lesion classification with folder-based organization"""
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Root directory containing train/test/valid folders
            split (str): Which dataset split to use ('train', 'test', or 'valid')
            transform (callable, optional): Optional transform to be applied on images
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Define class names based on folders
        self.classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Get all image paths and labels
        self.samples = self._get_samples()
        
    def _get_samples(self):
        """Get all image paths and their corresponding labels"""
        samples = []
        split_dir = self.root_dir / self.split
        
        for class_name in self.classes:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
                
            for img_path in class_dir.glob('*.jpg'):  # Can add more extensions if needed
                samples.append((str(img_path), self.class_to_idx[class_name]))
                
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load and convert image to RGB
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)
    
    def get_class_weights(self):
        """
        Calculate class weights for imbalanced dataset using effective samples
        Reference: https://arxiv.org/abs/1901.05555
        """
        class_counts = [0] * len(self.classes)
        total_samples = len(self.samples)
        beta = 0.9999  # Smoothing factor
        
        for _, label in self.samples:
            class_counts[label] += 1
            
        # Calculate effective number of samples
        effective_nums = [1.0 - np.power(beta, count) for count in class_counts]
        weights = [(1.0 - beta) / num if num > 0 else 0 for num in effective_nums]
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights) * len(self.classes)
        
        return torch.FloatTensor(weights)