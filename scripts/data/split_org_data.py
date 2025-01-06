import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)
import sys
import yaml
import logging
import random
import shutil
from pathlib import Path
from tqdm import tqdm
from src.utils import get_logger


logger = get_logger(__name__)

def make_directory(dir_path: str):
    """Create directory, removing if it already exists"""
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def split_data(config):
    """Split data into train, validation and test sets based on config"""
    
    # Set random seed for reproducibility
    random.seed(config['data']['random_state'])
    
    # Get paths from config
    data_root = config['data']['raw_dir']
    train_root = os.path.join(config['data']['processed_dir'], 'train')
    val_root = os.path.join(config['data']['processed_dir'], 'val')
    test_root = os.path.join(config['data']['processed_dir'], 'test')
    
    # Get split rates from config
    test_size = config['data']['test_size']
    val_size = config['data']['val_size'] 
    
    # Verify data root exists
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data directory {data_root} not found")
        
    # Get class directories
    data_classes = [cla for cla in os.listdir(data_root) 
                   if os.path.isdir(os.path.join(data_root, cla))]
    
    # Create output directories
    make_directory(train_root)
    make_directory(val_root)
    make_directory(test_root)
    
    # Create class subdirectories
    for class_name in data_classes:
        make_directory(os.path.join(train_root, class_name))
        make_directory(os.path.join(val_root, class_name))
        make_directory(os.path.join(test_root, class_name))
    
    # Track statistics
    total_images = 0
    class_stats = {}
    
    # Split data for each class
    for class_name in data_classes:
        class_path = os.path.join(data_root, class_name)
        images = os.listdir(class_path)
        num_images = len(images)
        total_images += num_images
        
        # Calculate split sizes
        n_test = int(num_images * test_size)
        n_val = int(num_images * val_size)
        n_train = num_images - n_test - n_val
        
        # Randomly sample test and validation sets
        test_images = set(random.sample(images, k=n_test))
        remaining_images = list(set(images) - test_images)
        val_images = set(random.sample(remaining_images, k=n_val))
        
        # Copy images to respective directories
        for idx, image in enumerate(tqdm(images, desc=f"Processing {class_name}")):
            src_path = os.path.join(class_path, image)
            
            if image in test_images:
                dst_path = os.path.join(test_root, class_name)
                shutil.copy(src_path, dst_path)
            elif image in val_images:
                dst_path = os.path.join(val_root, class_name)
                shutil.copy(src_path, dst_path)
            else:
                dst_path = os.path.join(train_root, class_name)
                shutil.copy(src_path, dst_path)
        
        # Record statistics
        class_stats[class_name] = {
            'total': num_images,
            'train': n_train,
            'val': n_val,
            'test': n_test
        }
    
    return total_images, class_stats

def log_statistics(total_images, class_stats):
    """Log split statistics"""
    logger.info(f"\nTotal images processed: {total_images}")
    logger.info("\nClass-wise split statistics:")
    
    for class_name, stats in class_stats.items():
        logger.info(f"\n{class_name}:")
        logger.info(f"Total: {stats['total']}")
        logger.info(f"Training: {stats['train']} ({stats['train']/stats['total']*100:.1f}%)")
        logger.info(f"Validation: {stats['val']} ({stats['val']/stats['total']*100:.1f}%)")
        logger.info(f"Testing: {stats['test']} ({stats['test']/stats['total']*100:.1f}%)")

def main():
    # Load config
    with open('config/config.yaml') as f:
        config = yaml.safe_load(f)
    
    logger.info("Starting data splitting process...")
    
    try:
        # Create output directories
        os.makedirs(config['data']['processed_dir'], exist_ok=True)
        os.makedirs(config['visualization']['output_dir'], exist_ok=True)
        
        # Split data
        total_images, class_stats = split_data(config)
        
        # Log statistics
        log_statistics(total_images, class_stats)
        
        logger.info("Data splitting completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during data splitting: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()