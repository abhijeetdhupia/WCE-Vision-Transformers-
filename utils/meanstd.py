"""
The mean and std must only be calculated from the training data. 
This will load PIL images by default so we pass the ToTensor transform which 
converts all the PIL images to tensors and scales them from 0-255 to 0-1.
We then loop through each image and calculate the mean and std across the height 
and width dimensions with dim = (1,2), summing all the means and stds and then 
finding the average by dividing them by the number of examples, len(train_data).
Again, this only needs to be calculated once per dataset and the means and stds 
calculated here can be re-used without calculating them for other runs.
"""
import os
import yaml 
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

#Config File 
with open("../configs/config.yaml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
images_dir = cfg['dataset']['mean_stdpath']
train_dir = os.path.join(images_dir, 'train')
# print("Train Directory: ",train_dir)

classes = os.listdir(images_dir)
means = torch.zeros(3)
stds = torch.zeros(3)
train_data = datasets.ImageFolder(root = train_dir, transform = transforms.ToTensor())

print("It might take a few minutes to show the result.")
for img, label in train_data:
    means += torch.mean(img, dim = (1,2))
    stds += torch.std(img, dim = (1,2))

means /= len(train_data)
stds /= len(train_data)
    
print(f'Calculated means: {means}')
print(f'Calculated stds: {stds}')

# Not needed for mean and std calculations 
"""
# test_dir = os.path.join(images_dir, 'test')
# print("Test Directory: ",test_dir)
"""