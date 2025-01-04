import os
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
from .transforms import AugmentationPipeline

class DataAugmentor:
    """Class for handling data augmentation process"""
    
    def __init__(self, config):
        self.config = config
        self.augmentation = AugmentationPipeline(config)
        
    def augment_image(self, image_path, num_augmentations):
        """
        Augment a single image multiple times
        
        Args:
            image_path (str): Path to image
            num_augmentations (int): Number of augmented versions to create
            
        Returns:
            list: List of augmented images
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        augmented_images = []
        for _ in range(num_augmentations):
            augmented = self.augmentation.train_transform(image=image)['image']
            augmented_images.append(augmented)
            
        return augmented_images
    
    def augment_class(self, df, class_name, target_samples):
        """
        Augment images of a specific class to reach target number of samples
        
        Args:
            df (pd.DataFrame): DataFrame with image metadata
            class_name (str): Name of the class to augment
            target_samples (int): Target number of samples
            
        Returns:
            pd.DataFrame: DataFrame with augmented data
        """
        class_df = df[df[class_name] == 1].copy()
        current_samples = len(class_df)
        
        if current_samples >= target_samples:
            return pd.DataFrame()  # No augmentation needed
            
        augmented_data = []
        samples_needed = target_samples - current_samples
        multiplier = samples_needed // current_samples + 1
        
        for idx, row in tqdm(class_df.iterrows(), 
                           total=len(class_df),
                           desc=f"Augmenting {class_name}"):
            image_path = os.path.join(self.config['data']['raw_dir'], 
                                    f"{row['image']}.jpg")
            
            for i in range(multiplier):
                if len(augmented_data) >= samples_needed:
                    break
                    
                new_row = row.copy()
                new_row['image'] = f"{row['image']}_aug_{i+1}"
                augmented_data.append(new_row)
                
        return pd.DataFrame(augmented_data)
    
    def augment_dataset(self, df):
        """
        Augment entire dataset based on config settings
        
        Args:
            df (pd.DataFrame): Original DataFrame with image metadata
            
        Returns:
            pd.DataFrame: Augmented DataFrame
        """
        augmented_df = df.copy()
        target_samples = self.config['augmentation']['target_samples']
        
        for class_name in ['MEL', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']:
            aug_class_df = self.augment_class(df, class_name, target_samples)
            if not aug_class_df.empty:
                augmented_df = pd.concat([augmented_df, aug_class_df], 
                                       ignore_index=True)
                
        return augmented_df
    
    def save_augmented_data(self, augmented_df):
        """
        Save augmented images and metadata
        
        Args:
            augmented_df (pd.DataFrame): DataFrame with augmented data
        """
        output_dir = self.config['data']['augmented_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        # Save original images
        for idx, row in tqdm(augmented_df.iterrows(), 
                           desc="Saving images",
                           total=len(augmented_df)):
            if '_aug_' not in row['image']:
                src_path = os.path.join(self.config['data']['raw_dir'], 
                                      f"{row['image']}.jpg")
                dst_path = os.path.join(output_dir, f"{row['image']}.jpg")
                if not os.path.exists(dst_path):
                    cv2.imwrite(dst_path, cv2.imread(src_path))
                    
            else:
                # Generate and save augmented image
                original_name = row['image'].split('_aug_')[0]
                original_path = os.path.join(self.config['data']['raw_dir'], 
                                          f"{original_name}.jpg")
                img = cv2.imread(original_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                augmented = self.augmentation.train_transform(image=img)['image']
                augmented = augmented.numpy().transpose(1, 2, 0)
                augmented = (augmented * 255).astype(np.uint8)
                cv2.imwrite(
                    os.path.join(output_dir, f"{row['image']}.jpg"),
                    cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                )
        
        # Save metadata
        augmented_df.to_csv(
            os.path.join(self.config['data']['processed_dir'], 'augmented_metadata.csv'),
            index=False
        )