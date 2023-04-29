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
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
    parser.add_argument('--save_folder', default='xxxxx/SMNet/checkpoint', help='Location to save checkpoint models')
    parser.add_argument('--save_pretrain', type=str, default='', help='the sub path to save pretrained model')

    # Testing settings
    parser.add_argument('--testBatchSize', type=int, default=32, help='testing batch size, default=1')
    parser.add_argument('--seed', type=int, default=1111, help='random seed to use. Default=123')
    parser.add_argument('--test_dataset', type=str, default='Set12', help='the testing dataset')
    parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')

    # Global settings
    parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
    parser.add_argument('--gpus', default=1, type=int, help='number of gpus')
    parser.add_argument('--workers', default=1, type=int, help='number of workers')
    parser.add_argument('--data_dir', type=str, default='./Dataset', help='the dataset dir')
    parser.add_argument('--sidd_path', default=r'xxxx/xxxx/train', help='sidd path')
    parser.add_argument('--test_path', default=r'xxxx/xxxx/test', help='sidd path')
    #parser.add_argument('--model_type', type=str, default='unet', help='the name of model')
    parser.add_argument('--patch_size', type=int, default=128, help='Size of cropped denoising image')
    parser.add_argument('--Isreal', default=True, help='If training/testing on RGB images')

    # Distributed Training
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--resume', action='store_true', default=False, help='load the log')
    args = parser.parse_args()
    return args


def main():
    config = init_parse()
    check_dir = os.path.join(config.save_folder, config.save_pretrain)
    if not os.path.exists(check_dir):
        os.makedirs(check_dir)
    check_path = os.path.join(check_dir, 'best.pth')
    
    if config.resume:
        logs = torch.load(check_path)
        print("loading checkpoint...")

    cudnn.benchmark = True
    torch.backends.cudnn.enabled = False
    train_loader, val_loader = build_dataset(config)
    print('===> Loading datasets')
    model = SMNet(3).cuda()
    criterion = CharbonnierLoss().cuda()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.nEpochs-3, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=3, after_scheduler=scheduler_cosine)
    scheduler.step()

    max_psnr = 0.0
    max_ssim = 0.0

    if config.resume:
        model.load_state_dict(logs['model_state'])
        config.start_iter = logs['epoch']
        max_psnr = logs['max_psnr']
        max_ssim = logs['max_ssim']
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-8)
        optimizer.load_state_dict(logs['optim_state'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.nEpochs-config.start_iter+1, eta_min=1e-6)

    psnr_list = []
    ssim_list = []
    for epoch in range(config.start_iter, config.nEpochs + 1):
        train(epoch, model, train_loader, optimizer, scheduler, criterion)
        psnr, ssim = test(model, val_loader)
        if psnr > max_psnr:
            save_results = {
            'max_psnr': psnr,
            'model_state': model.state_dict(),
            'epoch': epoch,
            'max_ssim': ssim,
            'optim_state': optimizer.state_dict()
            }
            torch.save(save_results, check_path)
            
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        data_frame = pd.DataFrame(data={'epoch': epoch, 'PSNR': psnr_list, 'SSIM': ssim_list})
        data_frame.to_csv(os.path.join(check_dir, 'training_logs_smnet.csv'))
        # # learning rate is decayed by a factor of 10 every half of total epochs
        if (epoch + 1) % (10) == 0:
            for param_group in optimizer.param_groups:
                print('Learning rate decay: lr={}'.format(param_group['lr']))


if __name__ == '__main__':
    main()
