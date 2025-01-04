import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from src.utils import get_logger

logger = get_logger(__name__)

def group_original_and_augmented(df):
    """
    Group original images with their augmented versions
    Returns a dictionary where key is original image name and value is list of all related images
    """
    image_groups = {}
    
    for idx, row in df.iterrows():
        img_name = row['image']
        # If this is augmented image (contains '_aug_')
        if '_aug_' in img_name:
            original_name = img_name.split('_aug_')[0]
            if original_name not in image_groups:
                image_groups[original_name] = []
            image_groups[original_name].append(img_name)
        else:
            # This is original image
            if img_name not in image_groups:
                image_groups[img_name] = []
            image_groups[img_name].append(img_name)
    
    return image_groups

def split_maintaining_groups(df, image_groups, test_size=0.2, random_state=42):
    """
    Split data while keeping original and augmented images together
    """
    # Get list of original images
    original_images = list(image_groups.keys())
    
    # Split original images
    train_originals, test_originals = train_test_split(
        original_images, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df[df['image'].isin(original_images)][['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']].values
    )
    
    # Get all related images for each split
    train_images = []
    test_images = []
    
    for orig in train_originals:
        train_images.extend(image_groups[orig])
    for orig in test_originals:
        test_images.extend(image_groups[orig])
    
    # Create train and test DataFrames
    train_df = df[df['image'].isin(train_images)].copy()
    test_df = df[df['image'].isin(test_images)].copy()
    
    return train_df, test_df

def analyze_split(train_df, test_df, output_dir):
    """
    Analyze and visualize the train-test split
    """
    classes = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    
    # Calculate class distributions
    train_dist = []
    test_dist = []
    
    logger.info("\nTrain set distribution:")
    for cls in classes:
        train_total = train_df[cls].sum()
        train_orig = train_df[~train_df['image'].str.contains('_aug_', na=False)][cls].sum()
        train_aug = train_df[train_df['image'].str.contains('_aug_', na=False)][cls].sum()
        train_dist.append(train_total)
        logger.info(f"{cls}: Total={train_total} (Original={train_orig}, Augmented={train_aug})")
    
    logger.info("\nTest set distribution:")
    for cls in classes:
        test_total = test_df[cls].sum()
        test_orig = test_df[~test_df['image'].str.contains('_aug_', na=False)][cls].sum()
        test_aug = test_df[test_df['image'].str.contains('_aug_', na=False)][cls].sum()
        test_dist.append(test_total)
        logger.info(f"{cls}: Total={test_total} (Original={test_orig}, Augmented={test_aug})")
    
    # Plot distribution comparison
    plt.figure(figsize=(12, 6))
    x = np.arange(len(classes))
    width = 0.35
    
    plt.bar(x - width/2, train_dist, width, label='Train')
    plt.bar(x + width/2, test_dist, width, label='Test')
    
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.title('Train-Test Split Distribution')
    plt.xticks(x, classes, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'train_test_distribution.png'))
    plt.close()

def main():
    # Load config
    with open('config/config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    os.makedirs(config['visualization']['output_dir'], exist_ok=True)
    
    # Load augmented data
    logger.info("Loading augmented dataset...")
    augmented_df = pd.read_csv(config['paths']['augmented_metadata'])
    
    # Group original and augmented images
    logger.info("Grouping original and augmented images...")
    image_groups = group_original_and_augmented(augmented_df)
    
    # Split data
    logger.info("Splitting dataset into train and test sets...")
    train_df, test_df = split_maintaining_groups(
        augmented_df, 
        image_groups, 
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )
    
    # Analyze split
    logger.info("Analyzing train-test split...")
    analyze_split(
        train_df, 
        test_df, 
        config['visualization']['output_dir']
    )
    
    # Save splits
    train_df.to_csv(config['paths']['train_metadata'], index=False)
    test_df.to_csv(config['paths']['test_metadata'], index=False)
    logger.info(f"Saved train split to {config['paths']['train_metadata']}")
    logger.info(f"Saved test split to {config['paths']['test_metadata']}")
    
    logger.info("Data splitting completed!")

if __name__ == '__main__':
    main()