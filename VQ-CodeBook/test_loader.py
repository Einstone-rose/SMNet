import os
import cv2
import numpy as np
from PIL import Image
import torch.utils.data as u_data
from torchvision import datasets, transforms
import torch

def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    return img

class getTestData(u_data.Dataset):
    def __init__(self, path):
        self.path = path
        self.files = sorted(os.listdir(os.path.join(self.path, 'input_crop')))
        self.transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 

    def __len__(self):
        return len(self.files) 

    def __getitem__(self, index):

        index = index % len(self.files)
        file_name = self.files[index]
        cnt = int(file_name.split('_')[-1].split('.')[0])
        gt_img = Image.open(os.path.join(self.path, 'input_crop', file_name))
        gt_img = gt_img.convert('RGB')
        gt_img = self.transform(gt_img)
        idx = torch.Tensor(np.array([cnt]))
        #gt_img = img_as_float(gt_img)
        #gt_img = gt_img.transpose((2, 0, 1))
        return gt_img, idx
