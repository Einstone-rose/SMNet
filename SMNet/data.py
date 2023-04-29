import torch
from os.path import join
from torchvision.transforms import Compose, ToTensor
from dataset import DatasetFromFolderEval, DatasetFromFolder
from utils.sidd_dataset import SIDDData, SIDDValData
import torch.utils.data as u_data


def transform():
    return Compose([
        ToTensor(),
    ])

def get_training_set(data_dir, hr, upscale_factor, patch_size, data_augmentation):
    hr_dir = join(data_dir, hr)
    return DatasetFromFolder(hr_dir, patch_size, upscale_factor, data_augmentation, transform=transform())


def get_eval_set(lr_dir, upscale_factor):
    return DatasetFromFolderEval(lr_dir, upscale_factor, transform=transform())

# 构建datasets和dataloader
def build_dataset(args):
    # training data
    train_dataset = SIDDData(args.sidd_path, args.patch_size)
    train_loader = u_data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.threads)
    # validation data
    valid_dataset = SIDDValData(args.test_path)
    valid_loader = u_data.DataLoader(valid_dataset, batch_size=args.testBatchSize, num_workers=args.threads)
    return train_loader, valid_loader
