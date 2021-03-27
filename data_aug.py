import os 
import cv2
import glob 
import pprint 
from utils import Aug 
from pathlib import Path

# root = "/data/healthcare/Datasets/tree_data/"
root = "/data/healthcare/Datasets/abnormalvnormal"

for file_path in Path(f'{root}').glob('**/*.png'):
    image = cv2.imread(f"{file_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    objct = Aug.Augmentation()
    _aug = objct._aug()

    # For a single image 
    augmented_image = _aug(image=image)
    aug_img = augmented_image['image']

    # file_path1 = str(file_path).split('tree_data')[0] + "tree_data1" + str(file_path).split('tree_data')[1]
    file_path1 = str(file_path).split('abnormalvnormal')[0] + "abnormalvnormal1" + str(file_path).split('abnormalvnormal')[1]
    new_path = str(file_path1).split(".")[0] 
    cv2.imwrite(f"{new_path}_aug2.png", cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
    # print("Done!")

# For multiple images 

# augmented_image = _aug(image=image, mask=image, mask1 = image)
# aug_img = augmented_image['image']
# print(aug_img.shape)
# aug_mask = augmented_image['mask']     
# print(aug_mask.shape)
# aug_mask1 = augmented_image['mask1']     
# print(aug_mask1.shape)