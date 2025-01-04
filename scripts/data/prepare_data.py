import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)
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