import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

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