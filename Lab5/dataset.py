import torch
import os
import json
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

class ICLEVR_dataset(Dataset):
    def __init__(self, file_path, img_path, mode='train', transform=train_transform):
        assert mode == 'train' or mode == 'eval' or mode == 'test'
        self.root = file_path
        self.img_root = img_path
        self.mode = mode
        self.transform = transform

        # Get the number of objects and the idexes, 24 objects
        self.obj_label = json.load(open(os.path.join(self.root, 'objects.json')))
        self.obj_len = len(self.obj_label)
        # Get dataset information
        if self.mode == 'train':
            # keys are filenames and values are objects, 18009 data
            data = json.load(open(os.path.join(self.root, 'train.json')))
            self.img_name = data.keys() # image file name
            self.obj = data.values() # objects
        elif self.mode == 'eval':
            # each element includes multiple objects, 32 data
            self.obj = json.load(open(os.path.join(self.root, 'test.json')))
        else:
            # for testing
            self.obj = json.load(open(os.path.join(self.root, 'new_test.json')))
        
        self.data_len = len(self.obj)
        self.data_label = self.one_hot() # one hot vector of label

    def one_hot(self):
        one_hot_vector = []
        for i in range(self.data_len):
            tmp_vector = np.zeros(self.obj_len)
            for obj_name in self.obj[i]:
                idx = self.obj_label.get(obj_name)
                tmp_vector[idx] = 1
            one_hot_vector.append(tmp_vector)
        return one_hot_vector
    
    def __len__(self):
        # number of dataset
        return len(self.data_len)

    def __getitem__(self, index):
        label = self.data_label[index]
        if self.mode == 'train':
            # get training image and its label
            img = Image.open(os.path.join(self.img_root, self.img_name[index])).convert('RGB')
            img = self.transform(img)
            return img, label
        else:
            # get testing label
            return label