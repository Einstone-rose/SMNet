import os
import cv2
import sys
import time
import logging
import argparse
import numpy as np
import torch.utils.data
from torch import optim
import torch.backends.cudnn as cudnn
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from test_loader import getTestData
from vq_vae.util import setup_logging_from_args
from vq_vae.auto_encoder import *
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

models = {
    'custom': {'vqvae': VQ_CVAE},
    'imagenet': {'vqvae': VQ_CVAE},
    'cifar10': {'vqvae': VQ_CVAE},
    'mnist': {'vqvae': VQ_CVAE},
}
datasets_classes = {
    'custom': datasets.ImageFolder,
    'imagenet': datasets.ImageFolder,
    'cifar10': datasets.CIFAR10,
    'mnist': datasets.MNIST
}
dataset_train_args = {
    'custom': {},
    'imagenet': {},
    'cifar10': {'train': True, 'download': True},
    'mnist': {'train': True, 'download': True},
}
dataset_test_args = {
    'custom': {},
    'imagenet': {},
    'cifar10': {'train': False, 'download': True},
    'mnist': {'train': False, 'download': True},
}
dataset_n_channels = {
    'custom': 3,
    'imagenet': 3,
    'cifar10': 3,
    'mnist': 1,
}

dataset_transforms = {
    'custom': transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'imagenet': transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'cifar10': transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'mnist': transforms.ToTensor()
}
default_hyperparams = {
    'custom': {'lr': 2e-4, 'k': 128, 'hidden': 128},
    'imagenet': {'lr': 2e-4, 'k': 512, 'hidden': 128},
    'cifar10': {'lr': 2e-4, 'k': 10, 'hidden': 256},
    'mnist': {'lr': 1e-4, 'k': 10, 'hidden': 64}
}


def main(args):
    parser = argparse.ArgumentParser(description='Variational AutoEncoders')

    model_parser = parser.add_argument_group('Model Parameters')
    model_parser.add_argument('--model', default='vae', choices=['vae', 'vqvae'], help='autoencoder variant to use: vae | vqvae')
    model_parser.add_argument('--batch-size', type=int, default=16, metavar='N', help='input batch size for training (default: 128)')
    model_parser.add_argument('--hidden', type=int, metavar='N', help='number of hidden channels')
    model_parser.add_argument('-k', '--dict-size', type=int, dest='k', metavar='K', help='number of atoms in dictionary')
    model_parser.add_argument('--lr', type=float, default=None, help='learning rate')
    model_parser.add_argument('--vq_coef', type=float, default=None, help='vq coefficient in loss')
    model_parser.add_argument('--commit_coef', type=float, default=None, help='commitment coefficient in loss')
    model_parser.add_argument('--kl_coef', type=float, default=None, help='kl-divergence coefficient in loss')

    training_parser = parser.add_argument_group('Training Parameters')
    training_parser.add_argument('--dataset', default='custom', choices=['mnist', 'cifar10', 'imagenet', 'custom'], help='dataset to use: mnist | cifar10 | imagenet | custom')
    training_parser.add_argument('--dataset_dir_name', default='SIDD', help='name of the dir containing the dataset if dataset == custom')
    training_parser.add_argument('--data-dir', default='/home/share/Einstone/Codebook/VQ-Codebook/images', help='directory containing the dataset')
    training_parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 10)')
    training_parser.add_argument('--max-epoch-samples', type=int, default=50000, help='max num of samples per epoch')
    training_parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
    training_parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    training_parser.add_argument('--gpus', default='2', help='gpus used for training - e.g 0,1,3')

    logging_parser = parser.add_argument_group('Logging Parameters')
    logging_parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    logging_parser.add_argument('--results-dir', metavar='RESULTS_DIR', default='./results', help='results dir')
    logging_parser.add_argument('--check_path', default='/home/share/Einstone/Codebook/VQ-VAE-master/checkpoint', help='checkpoint dir')
    logging_parser.add_argument('--resume', action='store_true', default=False, help='load the log')
    logging_parser.add_argument('--save-name', default='', help='saved folder')
    logging_parser.add_argument('--data-format', default='json', help='in which format to save the data')
    args = parser.parse_args(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    dataset_dir_name = args.dataset if args.dataset != 'custom' else args.dataset_dir_name

    lr = args.lr or default_hyperparams[args.dataset]['lr']
    k = args.k or default_hyperparams[args.dataset]['k']
    hidden = args.hidden or default_hyperparams[args.dataset]['hidden']
    num_channels = dataset_n_channels[args.dataset]

    save_path = setup_logging_from_args(args)
    writer = SummaryWriter(save_path)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        cudnn.benchmark = True
        torch.cuda.manual_seed(args.seed)

    model = models[args.dataset][args.model](hidden, k=k, num_channels=num_channels)

    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10 if args.dataset == 'imagenet' else 30, 0.5,)

    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
    dataset_train_dir = os.path.join(args.data_dir, dataset_dir_name)
    dataset_test_dir = os.path.join(args.data_dir, dataset_dir_name)

    # use custom 数据集
    if args.dataset in ['imagenet', 'custom']:
        dataset_train_dir = os.path.join(dataset_train_dir, 'train')
        dataset_test_dir = os.path.join(dataset_test_dir, 'test')
   
     
    train_loader = torch.utils.data.DataLoader(datasets_classes[args.dataset](dataset_train_dir, transform=dataset_transforms[args.dataset], **dataset_train_args[args.dataset]), batch_size=args.batch_size, shuffle=True, **kwargs)
 
    testdata = getTestData(os.path.join(args.data_dir, dataset_dir_name))
    test_loader = torch.utils.data.DataLoader(testdata, batch_size=1, shuffle=False, **kwargs)
    print("Loading the datasets!")

    # 只进行训练
    for epoch in range(1, args.epochs + 1):
        train_losses = train(epoch, model, train_loader, optimizer, args.cuda, args.log_interval, save_path, args, writer)
        for k in train_losses.keys():
            name = k.replace('_train', '')
            train_name = k
            writer.add_scalars(name, {'train': train_losses[train_name]})
        scheduler.step()
 
    # save the model
    save_model = {
       'model_state': model.state_dict(),
       'epoch': epoch,
       'optim_state': optimizer.state_dict()
    }
    torch.save(save_model, args.check_path + '/best.pth')

    # 训练完开始统一进行测试
    test_losses = test_net(model, test_loader, args.cuda, save_path, args, writer)
    # save the model
    print("The Test Ending!")


def train(epoch, model, train_loader, optimizer, cuda, log_interval, save_path, args, writer):
    model.train()
    loss_dict = model.latest_losses()
    losses = {k + '_train': 0 for k, v in loss_dict.items()}
    epoch_losses = {k + '_train': 0 for k, v in loss_dict.items()}
    start_time = time.time()
    batch_idx, data = None, None
    for batch_idx, (data, _) in enumerate(train_loader):
        if cuda:
            data = data.cuda()
        optimizer.zero_grad()
        outputs = model(data)
        loss = model.loss_function(data, *outputs)
        loss.backward()
        optimizer.step()
        latest_losses = model.latest_losses()
        for key in latest_losses:
            losses[key + '_train'] += float(latest_losses[key])
            epoch_losses[key + '_train'] += float(latest_losses[key])

        if batch_idx % log_interval == 0:
            for key in latest_losses:
                losses[key + '_train'] /= log_interval
            loss_string = ' '.join(['{}: {:.6f}'.format(k, v) for k, v in losses.items()])
            logging.info('Train Epoch: {epoch} [{batch:5d}/{total_batch} ({percent:2d}%)]   time:' ' {time:3.2f}   {loss}'
                         .format(epoch=epoch, batch=batch_idx * len(data), total_batch=len(train_loader) * len(data),
                                 percent=int(100. * batch_idx / len(train_loader)),
                                 time=time.time() - start_time,
                                 loss=loss_string))
            start_time = time.time()
            for key in latest_losses:
                losses[key + '_train'] = 0

        if args.dataset in ['imagenet', 'custom'] and batch_idx * len(data) > args.max_epoch_samples:
            break

    for key in epoch_losses:
        if args.dataset != 'imagenet':
            epoch_losses[key] /= (len(train_loader.dataset) / train_loader.batch_size)
        else:
            epoch_losses[key] /= (len(train_loader.dataset) / train_loader.batch_size)
    loss_string = '\t'.join(['{}: {:.6f}'.format(k, v) for k, v in epoch_losses.items()])
    logging.info('====> Epoch: {} {}'.format(epoch, loss_string))
    #writer.add_histogram('dict frequency', outputs[3], bins=range(args.k + 1))
    #model.print_atom_hist(outputs[3])
    return epoch_losses


def test_net(model, test_loader, cuda, save_path, args, writer):
    model.eval()
    loss_dict = model.latest_losses()
    losses = {k + '_test': 0 for k, v in loss_dict.items()}
    i, data = None, None
    with torch.no_grad():
        for i, (data, idx) in enumerate(test_loader):
            if cuda:
                data = data.cuda()
            outputs = model(data)
            model.loss_function(data, *outputs)
            latest_losses = model.latest_losses()
            for key in latest_losses:
                losses[key + '_test'] += float(latest_losses[key])
             
            # 保存测试图片 
            #print(idx.item())        
            save_reconstructed_images(idx.item(), data, outputs[0], outputs[-1], save_path, 'reconstruction_test')
            if i % 100 == 0:
                print("{0} is reconstructed!".format(i))

    return losses


def write_images(data, outputs, writer, suffix):
    original = data.mul(0.5).add(0.5)
    original_grid = make_grid(original[:6])
    writer.add_image(f'original/{suffix}', original_grid)
    reconstructed = outputs[0].mul(0.5).add(0.5)
    reconstructed_grid = make_grid(reconstructed[:6])
    writer.add_image(f'reconstructed/{suffix}', reconstructed_grid)

def resize_no_new_pixel(src_img, out_h, out_w):
    dst_img = np.zeros((out_h, out_w))

    height = src_img.shape[0]
    width = src_img.shape[1]

    w_scale = float(width / out_w)
    h_scale = float(height / out_h)

    for j in range(out_h):
        for i in range(out_w):
            raw_w = int(i * w_scale)
            raw_h = int(j * h_scale)
            dst_img[j][i] = src_img[raw_h][raw_w]

    return dst_img

def save_reconstructed_images(i, data, outputs, Map, save_path, name):
    """
      semantic_map: [32,32]
    """
    size = data.size()
    n = min(data.size(0), 8)
    batch_size = data.size(0)
    Map = Map.cpu()
    Map = Map.reshape(64, 64)
    if int(i) < 12000:
        Map = resize_no_new_pixel(Map, 512, 512)
        np.save(os.path.join("/home/share/Einstone/Codebook/VQ-VAE-master/Smap_train", 'Smap_' + str(int(i)) + '.npy'), Map)
    else:
        Map = resize_no_new_pixel(Map, 512, 512)
        np.save(os.path.join("/home/share/Einstone/Codebook/VQ-VAE-master/Smap_test", 'Smap_' + str(int(i)) + '.npy'), Map)

def save_checkpoint(model, epoch, save_path):
    os.makedirs(os.path.join(save_path, 'checkpoints'), exist_ok=True)
    checkpoint_path = os.path.join(save_path, 'checkpoints', f'model_{epoch}.pth')
    torch.save(model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    main(sys.argv[1:])
