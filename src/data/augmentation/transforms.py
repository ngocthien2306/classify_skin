import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

class AugmentationPipeline:
    """Class handling image augmentation pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.train_transform = self.get_train_transform()
        self.valid_transform = self.get_valid_transform()
        
    def get_train_transform(self):
        """Get augmentation pipeline for training"""
        return A.Compose([
            A.Resize(
                height=self.config['data']['image_size'],
                width=self.config['data']['image_size']
            ),
            A.OneOf([
                A.RandomRotate90(p=0.5),
                A.Rotate(
                    limit=self.config['augmentation']['rotate_limit'],
                    p=0.5
                ),
            ], p=0.5),
            A.Flip(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=self.config['augmentation']['brightness_contrast_limit'],
                    contrast_limit=self.config['augmentation']['brightness_contrast_limit'],
                    p=0.7
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.5
                ),
            ], p=0.5),
            A.OneOf([
                A.GaussNoise(p=0.3),
                A.GaussianBlur(p=0.3),
                A.MotionBlur(p=0.3),
            ], p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=45,
                p=0.5
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
        
    def get_valid_transform(self):
        """Get augmentation pipeline for validation/testing"""
        return A.Compose([
            A.Resize(
                height=self.config['data']['image_size'],
                width=self.config['data']['image_size']
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])