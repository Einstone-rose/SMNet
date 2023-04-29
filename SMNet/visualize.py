from __future__ import print_function
import os
import torch.distributed as dist
import pandas as pd
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from data import build_dataset
from base_train import train
from base_evaluation import test
from SMNet import SMNet
from losses import CharbonnierLoss
from warmup_scheduler import GradualWarmupScheduler
import torch
import random
from skimage import img_as_float32 as img_as_float

import os
import torch
import numpy as np
import cv2
import argparse
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
#from skimage.measure import compare_psnr, compare_ssim
from tqdm import tqdm
from skimage import img_as_ubyte
from data import build_dataset
import torch.nn.functional as F
import os
import cv2
import copy
import time
import math
import torch.nn as nn
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def init_parse():
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    # Training settings
    parser.add_argument('--upscale_factor', type=int, default=1, help="super resolution upscale factor")
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
    parser.add_argument('--nEpochs', type=int, default=3000, help='number of epochs to train for')
    parser.add_argument('--start_iter', type=int, default=1, help='starting epoch')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate. default=0.0001')
    parser.add_argument('--data_augmentation', type=bool, default=True, help='if adopt augmentation when training')
    parser.add_argument('--hr_train_dataset', type=str, default='DIV2K_train_HR', help='the training dataset')
    parser.add_argument('--Ispretrained', type=bool, default=True, help='If load checkpoint model')
    parser.add_argument("--noiseL", type=list, default=25, help='noise level')
    parser.add_argument('--save_folder', default='/home/share/Einstone/SMNet/checkpoint', help='Location to save checkpoint models')
    parser.add_argument('--vis_results', default='/home/share/Einstone/SMNet/visual_results', help='Location to save checkpoint models')
    parser.add_argument('--save_pretrain', type=str, default='', help='the sub path to save pretrained model')


    parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size, default=1')
    parser.add_argument('--seed', type=int, default=1111, help='random seed to use. Default=123')
    parser.add_argument('--test_dataset', type=str, default='Set12', help='the testing dataset')
    parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')

    # Global settings
    parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
    parser.add_argument('--gpus', default=1, type=int, help='number of gpus')
    parser.add_argument('--workers', default=1, type=int, help='number of workers')
    parser.add_argument('--data_dir', type=str, default='./Dataset', help='the dataset dir')
    parser.add_argument('--sidd_path', default=r'/home/share/Einstone/Denoise/data/SIDD_Small_sRGB_Only/patches', help='sidd path')
    parser.add_argument('--test_path', default=r'/home/share/Einstone/Codebook/VQ-VAE-master/data/test', help='sidd path')
    #parser.add_argument('--model_type', type=str, default='unet', help='the name of model')
    parser.add_argument('--patch_size', type=int, default=128, help='Size of cropped denoising image')
    parser.add_argument('--Isreal', default=True, help='If training/testing on RGB images')
    # Distributed Training
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--resume', action='store_true', default=False, help='load the log')
    args = parser.parse_args()
    return args

def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def load_gray_img(filepath):
    return np.expand_dims(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), axis=2)

def save_gray_img(filepath, img):
    cv2.imwrite(filepath, img)

def compute_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def compute_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def crop_patch(n_img, gt_img, map_img):
    pch_size = 128
    H, W, C = n_img.shape
    ind_H = 50
    ind_W = 50
    im_noisy = n_img[ind_H:ind_H + pch_size, ind_W:ind_W + pch_size, :]
    im_gt = gt_img[ind_H:ind_H + pch_size, ind_W:ind_W + pch_size, :]
    im_map = map_img[ind_H:ind_H + pch_size, ind_W:ind_W + pch_size, :]
    return im_noisy, im_gt, im_map

def main():
    config = init_parse()
    check_dir = os.path.join(config.save_folder, config.save_pretrain)
    if not os.path.exists(check_dir):
        os.makedirs(check_dir)
    check_path = os.path.join(check_dir, 'best.pth')
    
    if config.resume:
        logs = torch.load(check_path)
        print("loading checkpoint...")

#    set_seed(1111)
    cudnn.benchmark = True
    torch.backends.cudnn.enabled = False
    train_loader, val_loader = build_dataset(config) # 构建数据集
    print('===> Loading datasets')
    model = SMNet(3).cuda()

    if config.resume:
        model.load_state_dict(logs['model_state']) # 导入模型
        config.start_iter = logs['epoch']
        max_psnr = logs['max_psnr']
        max_ssim = logs['max_ssim']
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-8)
        optimizer.load_state_dict(logs['optim_state'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.nEpochs-config.start_iter+1, eta_min=1e-6)

    # Start to Test
    clean_files = sorted(os.listdir(os.path.join(config.test_path, 'GT')))
    noisy_files = sorted(os.listdir(os.path.join(config.test_path, 'Noisy')))
    map_files = sorted(os.listdir(os.path.join(config.test_path, 'Smap')))

    clean_filenames = [os.path.join(config.test_path, 'GT', x) for x in clean_files]
    noisy_filenames = [os.path.join(config.test_path, 'Noisy', x) for x in noisy_files]
    map_filenames =  [os.path.join(config.test_path, 'Smap', x) for x in map_files]

    f = 0
    model.eval()
    psnr, ssim = [], []
    with torch.no_grad():
        for clean_file, noisy_file, map_file in tqdm(zip(clean_filenames, noisy_filenames, map_filenames)):
            # 读取文件
            clean = cv2.cvtColor(cv2.imread(clean_file), cv2.COLOR_BGR2RGB)
            noisy = cv2.cvtColor(cv2.imread(noisy_file), cv2.COLOR_BGR2RGB)
            smap = np.load(map_file)
            #smap = np.expand_dims(map_original, axis=2)

            smap_img = smap
            noisy_img = noisy
            clean_img = clean
            #noisy_img, clean_img, smap_img  = crop_patch(noisy, clean, smap)  # 随机裁剪 
            
            # 转化类型
            noisy_input = img_as_float(noisy_img)
            noisy_input = noisy_input.transpose((2, 0, 1)) # [C,H,W]
            smap_input = smap_img.astype(np.float32)
            #print(smap_input.shape)
            smap_input = smap_input.transpose((2, 0, 1))
            smap_input = torch.Tensor(smap_input).unsqueeze(0).cuda()
            noisy_input = torch.Tensor(noisy_input).unsqueeze(0).cuda()

            restored = model(noisy_input, smap_input)
            restored = torch.clamp(restored, 0, 1)
            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
            restored = img_as_ubyte(restored[0])
            #f = os.path.splitext(os.path.split()[-1])[0]
            f = f + 1
            #print(f)
            #exit()
            save_img(os.path.join(config.vis_results, 'denoise', str(f) + '_denoise.png'), restored)
            save_img(os.path.join(config.vis_results, 'noisy', str(f) + '_noisy.png'), noisy_img)
            save_img(os.path.join(config.vis_results, 'clean', str(f) + '_clean.png'), clean_img)
            #np.save(os.path.join(config.vis_results, 'smap', f + '_smap.npy'), smap_img)

        #avg_psnr = sum(psnr) / len(psnr)
        #avg_ssim = sum(ssim) / len(ssim)
        #print('PSNR: {:f}----SSIM: {:f}\n'.format(avg_psnr, avg_ssim))
        #print(f"\nRestored images are saved at {config.vis_results}")

if __name__ == '__main__':
    main()

