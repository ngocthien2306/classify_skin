import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.augmentation import DataAugmentor
from src.visualization import Visualizer
from src.utils import get_logger
import numpy as np
import cv2
from pathlib import Path

logger = get_logger(__name__)

def analyze_dataset(df, output_dir, config):
    """Analyze the dataset and create visualizations"""
    logger.info("Analyzing dataset...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get class distribution
    class_counts = {}
    classes = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    for cls in classes:
        class_counts[cls] = df[cls].sum()
    
    # Plot class distribution
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_counts.keys(), class_counts.values())
    plt.title('Class Distribution before Augmentation')
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution_before.png'))
    plt.close()
    
    # Calculate and print class percentages
    total_samples = sum(class_counts.values())
    logger.info("\nClass Distribution:")
    for cls, count in class_counts.items():
        percentage = (count / total_samples) * 100
        logger.info(f"{cls}: {count} samples ({percentage:.2f}%)")
    
    # Analyze image properties
    image_dir = Path(config['data']['raw_dir'])
    heights = []
    widths = []
    aspect_ratios = []
    
    logger.info("\nAnalyzing image properties...")
    for idx, row in df.iterrows():
        img_path = image_dir / f"{row['image']}.jpg"
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            heights.append(h)
            widths.append(w)
            aspect_ratios.append(w/h)
    
    # Plot image size distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.hist(heights, bins=30)
    ax1.set_title('Height Distribution')
    ax1.set_xlabel('Height')
    ax1.set_ylabel('Count')
    
    ax2.hist(widths, bins=30)
    ax2.set_title('Width Distribution')
    ax2.set_xlabel('Width')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'image_size_distribution.png'))
    plt.close()
    
    # Plot aspect ratio distribution
    plt.figure(figsize=(10, 5))
    plt.hist(aspect_ratios, bins=30)
    plt.title('Aspect Ratio Distribution')
    plt.xlabel('Aspect Ratio')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'aspect_ratio_distribution.png'))
    plt.close()
    
    # Print image statistics
    logger.info("\nImage Statistics:")
    logger.info(f"Height - Mean: {np.mean(heights):.2f}, Std: {np.std(heights):.2f}")
    logger.info(f"Width - Mean: {np.mean(widths):.2f}, Std: {np.std(widths):.2f}")
    logger.info(f"Aspect Ratio - Mean: {np.mean(aspect_ratios):.2f}, Std: {np.std(aspect_ratios):.2f}")
    
    return class_counts

def prepare_data(config):
    """Prepare and augment dataset"""
    # Load metadata
    df = pd.read_csv(config['paths']['metadata'])
    logger.info(f"Loaded {len(df)} samples from metadata")
    
    # Analyze original dataset
    output_dir = os.path.join(config['visualization']['output_dir'], 'data_analysis')
    class_counts = analyze_dataset(df, output_dir, config)
    
    # Initialize augmentor
    augmentor = DataAugmentor(config)
    
    # Augment dataset
    logger.info("Starting data augmentation...")
    augmented_df = augmentor.augment_dataset(df)
    
    # Save augmented data
    augmentor.save_augmented_data(augmented_df)
    
    # Analyze augmented dataset
    logger.info("\nAnalyzing augmented dataset...")
    aug_class_counts = {}
    classes = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    for cls in classes:
        aug_class_counts[cls] = augmented_df[cls].sum()
    
    # Plot comparison of class distribution
    plt.figure(figsize=(12, 6))
    x = np.arange(len(classes))
    width = 0.35
    
    plt.bar(x - width/2, [class_counts[cls] for cls in classes], width, label='Original')
    plt.bar(x + width/2, [aug_class_counts[cls] for cls in classes], width, label='Augmented')
    
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution Comparison')
    plt.xticks(x, classes, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution_comparison.png'))
    plt.close()
    
    # Print augmentation results
    logger.info("\nAugmentation Results:")
    for cls in classes:
        original = class_counts[cls]
        augmented = aug_class_counts[cls]
        increase = augmented - original
        increase_percent = (increase / original) * 100
        logger.info(f"{cls}: {original} -> {augmented} (+{increase} samples, +{increase_percent:.2f}%)")

def main():
    # Load config
    with open('config/config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Create output directories
    os.makedirs(config['data']['processed_dir'], exist_ok=True)
    os.makedirs(config['data']['augmented_dir'], exist_ok=True)
    os.makedirs(config['visualization']['output_dir'], exist_ok=True)
    
    # Prepare and augment data
    prepare_data(config)
    
    logger.info("Data preparation completed!")

if __name__ == '__main__':
    main()