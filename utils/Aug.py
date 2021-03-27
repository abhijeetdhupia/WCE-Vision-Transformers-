# Albumentation as a class 

import cv2 
from albumentations import (CLAHE, Compose, ElasticTransform, GridDistortion,
                            HorizontalFlip, OneOf, OpticalDistortion,
                            RandomBrightnessContrast, RandomGamma,
                            RandomRotate90, ShiftScaleRotate, Transpose,
                            VerticalFlip)
                            
class Augmentation(object):

    def __init__(self):
        super(Augmentation, self).__init__()
        self._hflip = HorizontalFlip(p=0.5)
        self._vflip = VerticalFlip(p=0.5)
        self._clahe = CLAHE(p=.3)
        self._rotate = RandomRotate90(p=.3)
        self._brightness = RandomBrightnessContrast(p=0.3)
        self._gamma = RandomGamma(p=0.3)
        self._transpose = Transpose(p=0.3)
        self._elastic = ElasticTransform(
            p=.3, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
        self._distort = GridDistortion(p=0.3)
        self._affine = ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.3)

    def _aug(self):
        aug = [self._hflip, self._vflip, self._clahe, self._rotate, self._brightness,
                self._gamma, self._transpose, self._elastic, self._distort, self._affine]
        
        return Compose(aug)


# Official example from the Albumentation documentation

'''

import albumentations as  A
import cv2

# Declare an augmentation pipeline
transform = A.Compose([
    # A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.CLAHE(p=.3), 
    A.RandomRotate90(p=.3),
    A.RandomBrightnessContrast(p=0.3),
    A.RandomGamma(p=0.3), 
    A.Transpose(p=0.3),
    A.ElasticTransform(p=.3, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03), 
    A.GridDistortion(p=0.3), 
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.3), 
])

# Read an image with OpenCV and convert it to the RGB colorspace
image = cv2.imread("/data/healthcare/Datasets/kid-1/train/Angioectasias/angioectasia-P0-2.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Augment an image
transformed = transform(image=image)
transformed_image = transformed["image"]
print(f"OG Image Shape: {image.shape}")
print(f"Transformed Image Shape: {transformed_image.shape}")/

# Write the augmented image 
cv2.imwrite("/data/healthcare/Abhijeet/og_image.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
cv2.imwrite("/data/healthcare/Abhijeet/transformed.jpg", cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))

'''

