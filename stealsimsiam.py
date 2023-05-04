import getpass

import argparse
import builtins
import logging
import math
import os
import random
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import warnings
from torch.utils.data import DataLoader
from loss import soft_nn_loss, pairwise_euclid_distance
from data_aug.gaussian_blur import GaussianBlur

from models.resnet_simclr import ResNetSimCLRV2
from utils import print_args

user = getpass.getuser()

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',
                    help='path to imagenet dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=20, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', metavar='N', type=int,
                    # default=4096,
                    default=128,
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial (base) learning rate',
                    dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
# parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
parser.add_argument('--resume', action='store_true',
                    help='resume from checkpoint (if present)')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset for downstream task')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# additional configs:
parser.add_argument('--pretrained', default='', type=str,
                    help='path to simsiam pretrained checkpoint')
parser.add_argument('--lars', action='store_true',
                    help='Use LARS')
parser.add_argument('--epochstrain', default=200, type=int, metavar='N',
                    help='number of epochs victim was trained with')
parser.add_argument('--epochssteal', default=100, type=int, metavar='N',
                    help='number of epochs stolen model was trained with')
parser.add_argument('--num_queries', default=50000, type=int, metavar='N',
                    help='number of queries to steal with with')
parser.add_argument('--losstype', default='mse', type=str,
                    help='Loss function to use.')
parser.add_argument('--useval', default='False', type=str,
                    help='Use validation set for stealing (only with imagenet)')
parser.add_argument('--useaug', default='False', type=str,
                    help='Use augmentations with stealing')
parser.add_argument('--datasetsteal', default='cifar10', type=str,
                    help='dataset used for querying')
parser.add_argument('--temperature', default=0.1, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--temperaturesn', default=1000, type=float,
                    help='temperature for soft nearest neighbors loss')

prefix = '/ssd003'
pathpre = f"/scratch/ssd004/scratch/{os.getenv('USER')}/checkpoint"

best_acc1 = 0


def info_nce_loss(features, args):
    n = int(features.size()[0] / args.batch_size)
    labels = torch.cat(
        [torch.arange(args.batch_size) for i in range(n)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     args.n_views * args.batch_size, args.n_views * args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(
        similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape
    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives
    negatives = similarity_matrix[~labels.bool()].view(
        similarity_matrix.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long)
    logits = logits / args.temperature
    return logits, labels


def main():
    args = parser.parse_args()

    print_args(args=args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    print("# gpus", ngpus_per_node)
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    log_dir = f"{pathpre}/SimCLR/SimSiam/"
    logname = f"stealing{args.datasetsteal}{args.num_queries}{args.losstype}.log"
    logging.basicConfig(
        filename=os.path.join(log_dir, logname),
        level=logging.DEBUG)

    logging.debug(f"args: {args}")
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
    # create models
    print("=> loading model '{}'".format(args.arch))

    victim_model = models.__dict__[args.arch]()
    checkpoint = torch.load(
        "models/checkpoint_0099-batch256.pth.tar",
        map_location="cpu")
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder up to before the embedding layer
        if k.startswith('module.encoder') and not k.startswith(
                'module.encoder.fc'):
            # remove prefix
            state_dict[k[len("module.encoder."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    # print("state dict", state_dict.keys())
    victim_model.load_state_dict(state_dict, strict=False)
    victim_model.fc = torch.nn.Identity()

    # Stolen model initialzied
    model = ResNetSimCLRV2(base_model=args.arch, out_dim=512,
                           loss=args.losstype, include_mlp=False)

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            victim_model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(
                (args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[
                                                                  args.gpu])
            victim_model = torch.nn.parallel.DistributedDataParallel(
                victim_model,
                device_ids=[
                    args.gpu])
        else:
            victim_model.cuda()
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            victim_model = torch.nn.parallel.DistributedDataParallel(
                victim_model)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        victim_model = victim_model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
            victim_model = torch.nn.DataParallel(victim_model).cuda()

    # define loss function (criterion) and optimizer
    if args.losstype == "mse":
        criterion = nn.MSELoss().cuda(args.gpu)
    elif args.losstype == "infonce":
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    elif args.losstype == "softnn":
        criterion = soft_nn_loss

    optimizer = torch.optim.SGD(model.parameters(), init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.lars:
        print("=> use LARS optimizer.")
        from apex.parallel.LARC import LARC
        optimizer = LARC(optimizer=optimizer, trust_coefficient=.001,
                         clip=False)

    # optionally resume from a checkpoint
    if args.resume:
        checkloc = f"{pathpre}/SimCLR/SimSiam/checkpoint_{args.datasetsteal}_{args.losstype}_{args.num_queries}.pth.tar"
        if os.path.isfile(checkloc):
            print("=> loading checkpoint '{}'".format(checkloc))
            if args.gpu is None:
                checkpoint = torch.load(checkloc)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(checkloc, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = 0
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(checkloc, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(checkloc))

    cudnn.benchmark = True

    # Data loading code
    if args.datasetsteal == 'imagenet':
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if prefix == "/ssd003":
            train_dataset = datasets.ImageNet(
                root="/scratch/ssd002/datasets/imagenet_pytorch/",
                split="train",
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))

            indxs = random.sample(range(len(train_dataset)), args.num_queries)
            train_dataset = torch.utils.data.Subset(train_dataset, indxs)

            val_dataset = datasets.ImageNet(
                root="/scratch/ssd002/datasets/imagenet_pytorch/",
                split="val",
                transform=transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              normalize,
                                              ]))
        else:
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))

            val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))



    elif args.datasetsteal == 'cifar10':
        transform_train = transforms.Compose([
            transforms.Resize(224),
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset_raw = datasets.CIFAR10(
            root=f'{prefix}/home/{user}/data/cifar10', train=True,
            download=True, transform=transform_train)

        test_dataset = datasets.CIFAR10(
            root=f'{prefix}/home/{user}/data/cifar10', train=False,
            download=True, transform=transform_train)

        # use both train and test samples for queries:
        train_dataset = torch.utils.data.ConcatDataset(
            [train_dataset_raw, test_dataset])

    elif args.datasetsteal == 'cifar100':
        transform_train = transforms.Compose([
            transforms.Resize(224),
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR100(
            root=f'{prefix}/home/{user}/data/cifar100', train=True,
            download=True, transform=transform_train)

        test_dataset = datasets.CIFAR100(
            root=f'{prefix}/home/{user}/data/cifar100', train=False,
            download=True, transform=transform_test)
        val_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers)

    elif args.datasetsteal == 'svhn':

        transform_svhn = transforms.Compose([
            transforms.Resize(224),
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.42997558, 0.4283771, 0.44269393),
                                 (0.19630221, 0.1978732, 0.19947216)),
        ])

        train_dataset = datasets.SVHN(
            root=f'{prefix}/home/{user}/data/SVHN', split='extra',
            download=False, transform=transform_svhn)

    elif args.datasetsteal == "stl10":

        transform_stl = transforms.Compose([
            transforms.Resize(224),
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
        ])

        train_dataset = datasets.STL10(
            root=f"/checkpoint/{os.getenv('USER')}/SimCLR/stl10",
            split='unlabeled',
            download=False, transform=transform_stl)

    else:
        raise Exception(f"Unknown args.datasetsteal: {args.datasetsteal}.")

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        train_sampler = None

    if args.useval == "True" and args.dataset == "imagenet":
        query_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size,
            shuffle=False,  # (train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler,
            drop_last=True)
    else:
        query_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=False,  # (train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler,
            drop_last=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    victim_model.eval()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch (stealing)
        train(query_loader, model, victim_model, criterion, optimizer, epoch,
              args)

        # # evaluate on validation set (doesnt apply when stealing since a linear classifier is further needed)
        # acc1 = validate(val_loader, model, criterion, args)

        # # remember best acc@1 and save checkpoint
        # is_best = acc1 > best_acc1
        # best_acc1 = max(acc1, best_acc1)

        if (epoch > 0 and epoch % 10 == 0) or epoch == args.epochs - 1:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=True, args=args)


def train(train_loader, model, victim_model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    # top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],  # , top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    num = 0
    model.train()

    size = 224
    s = 1
    color_jitter = transforms.ColorJitter(
        0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    data_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=int(0.1 * size)),
        ])

    tloss = 0

    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        # compute output
        with torch.no_grad():
            victim_features = victim_model(images)

        if args.useaug == "True":
            augment_images = []
            for image in images:
                aug_image = to_pil(image)
                aug_image = data_transforms(aug_image)
                aug_image = to_tensor(aug_image)
                augment_images.append(aug_image)

            augment_images = torch.stack(augment_images)
            if args.gpu is not None:
                augment_images = augment_images.cuda(args.gpu,
                                                     non_blocking=True)
            stolen_features = model(augment_images)
        else:
            stolen_features = model(images)

        if args.losstype == "mse":
            loss = criterion(stolen_features, victim_features)
        elif args.losstype == "infonce":
            all_features = torch.cat([stolen_features, victim_features], dim=0)
            logits, labels = info_nce_loss(all_features, args)
            logits = logits.cuda(args.gpu, non_blocking=True)
            labels = labels.cuda(args.gpu, non_blocking=True)
            loss = criterion(logits, labels)

        elif args.losstype == "softnn":
            all_features = torch.cat([stolen_features, victim_features], dim=0)
            loss = criterion(args, all_features,
                             pairwise_euclid_distance, args.temperaturesn)

        # measure accuracy and record loss
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        tloss += loss.item()
        # top1.update(acc1[0], images.size(0))
        # top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        num += len(images)
        if num > args.num_queries:
            break

        if i % args.print_freq == 0:
            progress.display(i)
    logging.debug(f"Epoch: {epoch}. Loss: {tloss / i}")
    # print average over batch


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        logging.debug(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                      .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, args):
    if is_best:
        torch.save(state,
                   f"{pathpre}/SimCLR/SimSiam/checkpoint_{args.datasetsteal}_{args.losstype}_{args.num_queries}.pth.tar")


def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        k_pre = 'module.encoder.' + k[len('module.'):] \
            if k.startswith('module.') else 'module.encoder.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
