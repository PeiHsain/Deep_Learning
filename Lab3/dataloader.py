import pandas as pd
import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
from PIL import Image
import os


def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        self.data_transform_train = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=30),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.3375, 0.2618, 0.1873],
                                    std=[0.2918, 0.2088, 0.1684])
            ])
        self.data_transform_test = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.3375, 0.2618, 0.1873],
                                    std=[0.2918, 0.2088, 0.1684])
            ])
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        path = os.path.join(self.root, self.img_name[index]+".jpeg")
        PIL_img = Image.open(path).convert('RGB') # Load the image
        label = self.label[index]
        if self.mode == 'train':
            img = self.data_transform_train(PIL_img)
            # Convert the pixel value to [0, 1]
            # Transpose the image shape from [H, W, C] to [C, H, W]
            
        elif self.mode == 'test':
            img = self.data_transform_test(PIL_img)
            # Normalize the data
        return img, label
