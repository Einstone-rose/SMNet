import os
import cv2
import numpy as np
from skimage import img_as_float32 as img_as_float
import random
from torchvision import transforms, datasets
from scipy.io import loadmat
import torch.utils.data as u_data
import torch
def data_augmentation(image, mode):
    """
    Performs data augmentation of the input image
    Input:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
                0 - no transformation
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    """
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')

    return out


def random_augmentation(*args):
    out = []
    if random.randint(0, 1) == 1:
        flag_aug = random.randint(1, 7)
        for data in args:
            out.append(data_augmentation(data, flag_aug).copy())
    else:
        for data in args:
            out.append(data)
    return out


def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    return img

def is_png_file(filename):
    return any(filename.endswith(extension) for extension in ['.PNG', '.png'])

class SIDDData(u_data.Dataset):
    def __init__(self, path, patch_size):
        """
        Args:
            h5_path (str): path of the hdf5 file
            length (int): length of Datasets
        """
        super(SIDDData, self).__init__()
        # 需要进行额外的排序才能将noisy和gt图像实现对齐
        self.path = path
        clean_files = sorted(os.listdir(os.path.join(path, 'GT')))
        noisy_files = sorted(os.listdir(os.path.join(path, 'Noisy')))
        map_files = sorted(os.listdir(os.path.join(path, 'Smap')))
        # training data
        self.clean_filenames = [os.path.join(path, 'GT', x) for x in clean_files]
        self.noisy_filenames = [os.path.join(path, 'Noisy', x) for x in noisy_files]
        self.map_filenames = [os.path.join(path, 'Smap', x) for x in map_files]
        
        self.num_images = len(self.clean_filenames)
        self.pch_size = patch_size

    def __len__(self):
        return self.num_images

    def crop_patch(self, n_img, gt_img, map_img):
        H, W, C = n_img.shape
        # minus the bayer patter channel
        ind_H = random.randint(0, H - self.pch_size)
        ind_W = random.randint(0, W - self.pch_size)
        im_noisy = n_img[ind_H:ind_H + self.pch_size, ind_W:ind_W + self.pch_size, :]
        im_gt = gt_img[ind_H:ind_H + self.pch_size, ind_W:ind_W + self.pch_size, :]
        im_map = map_img[ind_H:ind_H + self.pch_size, ind_W:ind_W + self.pch_size, :]
        return im_noisy, im_gt, im_map

    def __getitem__(self, index):
        index = index % len(self.clean_filenames)
        # 导入
        noisy_img = load_img(self.noisy_filenames[index])
        gt_img = load_img(self.clean_filenames[index])
        map_img = np.load(self.map_filenames[index])

        #print(noisy_img.shape, gt_img.shape, map_img.shape)

        noisy_img, gt_img = np.array(noisy_img), np.array(gt_img)
        map_img = np.expand_dims(map_img, axis=2) # (512,512,1)
        # 进行crop
        if noisy_img.shape[0] > self.pch_size:
            noisy_img, gt_img, map_img  = self.crop_patch(noisy_img, gt_img, map_img)  # 随机裁剪 
        # 转化数据类型
        gt_img = img_as_float(gt_img)
        noisy_img = img_as_float(noisy_img)
        map_img = map_img.astype(np.float32)

        #print(gt_img.shape, noisy_img.shape, map_img.shape) 
        noisy_img, gt_img, map_img = random_augmentation(noisy_img, gt_img, map_img)
        gt_img = gt_img.transpose((2, 0, 1))
        noisy_img = noisy_img.transpose((2, 0, 1)) # [C,H,W]
        map_img = map_img.transpose((2, 0, 1))
        return noisy_img, gt_img, map_img


class SIDDValData(u_data.Dataset):
    def __init__(self, path):
        clean_files = sorted(os.listdir(os.path.join(path, 'GT')))
        noisy_files = sorted(os.listdir(os.path.join(path, 'Noisy')))
        map_files = sorted(os.listdir(os.path.join(path, 'Smap')))
        
        # training data
        self.clean_filenames = [os.path.join(path, 'GT', x) for x in clean_files]
        self.noisy_filenames = [os.path.join(path, 'Noisy', x) for x in noisy_files]
        self.map_filenames = [os.path.join(path, 'Smap', x) for x in map_files]
        self.num_images = len(self.clean_filenames)
        self.pch_size = 128

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        index = index % len(self.clean_filenames)
        noisy_img = load_img(self.noisy_filenames[index])
        gt_img = load_img(self.clean_filenames[index])
        map_img = np.load(self.map_filenames[index])

        noisy_img, gt_img = np.array(noisy_img), np.array(gt_img)
 
        # 转化数据类型
        gt_img = img_as_float(gt_img)
        noisy_img = img_as_float(noisy_img)
        map_img = map_img.astype(np.float32)

        gt_img = gt_img.transpose((2, 0, 1))
        noisy_img = noisy_img.transpose((2, 0, 1)) # [C,H,W]
        map_img = map_img.transpose((2, 0, 1))
        return noisy_img, gt_img, map_img

#class SIDDValData(u_data.Dataset):
#    def __init__(self, path):
#        val_data_dict = loadmat(os.path.join(path, 'validation', 'Noisy', 'ValidationNoisyBlocksSrgb.mat'))
#        val_data_noisy = val_data_dict['ValidationNoisyBlocksSrgb']
#        val_data_dict = loadmat(os.path.join(path, 'validation', 'GT', 'ValidationGtBlocksSrgb.mat'))
#        val_data_gt = val_data_dict['ValidationGtBlocksSrgb']
#        self.num_img, self.num_block, h_, w_, c_ = val_data_gt.shape
#        self.val_data_noisy = np.reshape(val_data_noisy, (-1, h_, w_, c_))
#        self.val_data_gt = np.reshape(val_data_gt, (-1, h_, w_, c_))
#
#    def __len__(self):
#        return self.num_img * self.num_block
#
#    def __getitem__(self, index):
#        noisy_img, gt_img = self.val_data_noisy[index], self.val_data_gt[index]
#        gt_img = img_as_float(gt_img)
#        noisy_img = img_as_float(noisy_img)
#        gt_img = gt_img.transpose((2, 0, 1))
#        noisy_img = noisy_img.transpose((2, 0, 1))
#        # dataPair = {'gt': gt_img, 'noisy': noisy_img}
#        return noisy_img, gt_img
