import torch
import sys
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import torchvision
import argparse
from torch.utils.data import DataLoader
from models.resnet import ResNetSimCLR, ResNet18, ResNet34, ResNet50
import torchvision.transforms as transforms
import logging
from torchvision import datasets


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-folder-name', metavar='DIR', default='test',
                    help='path to dataset')
parser.add_argument('--dataset', default='cifar10',
                    help='dataset name', choices=['stl10', 'cifar10', 'svhn', 'imagenet', 'cifar100'])
parser.add_argument('--dataset-test', default='cifar10',
                    help='dataset to run downstream task on', choices=['stl10', 'cifar10', 'svhn'])
parser.add_argument('--datasetsteal', default='cifar10',
                    help='dataset used for querying the victim', choices=['stl10', 'cifar10', 'svhn', 'imagenet', 'tinyimages'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34',
        choices=['resnet18', 'resnet34', 'resnet50'], help='model architecture')
parser.add_argument('-n', '--num-labeled', default=50000,type=int,
                     help='Number of labeled examples to train on')
parser.add_argument('--epochstrain', default=200, type=int, metavar='N',
                    help='number of epochs victim was trained with')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of epochs stolen model was trained with')
parser.add_argument('--num_queries', default=9000, type=int, metavar='N',
                    help='Number of queries to steal the model.')
parser.add_argument('--lr', default=1e-4, type=float, # maybe try other lrs
                    help='learning rate to train the model with.')
parser.add_argument('--modeltype', default='stolen', type=str,
                    help='Type of model to evaluate', choices=['victim', 'stolen', 'random'])
parser.add_argument('--save', default='False', type=str,
                    help='Save final model', choices=['True', 'False'])
parser.add_argument('--losstype', default='infonce', type=str,
                    help='Loss function to use.')
parser.add_argument('--head', default='False', type=str,
                    help='stolen model was trained using recreated head.', choices=['True', 'False'])
parser.add_argument('--sigma', default=0.5, type=float,
                    help='standard deviation used for perturbations')
parser.add_argument('--mu', default=5, type=float,
                    help='mean noise used for perturbations')
parser.add_argument('--clear', default='False', type=str,
                    help='Clear previous logs', choices=['True', 'False'])
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--retrain', action='store_true',
                    help='evaluating a retrained model (use modeltype=victim still)')
parser.add_argument('--array_id', type=int, default=0, help='slurm array id.')
parser.add_argument('--changeepochs', action='store_true',
                    help='use to adjust epochs based on array id instead of queries.')

args = parser.parse_args()
if args.retrain:
    args.modeltype = "victim"
    if args.changeepochs:
        num_epochs = [5, 10, 25, 50]
        args.epochstrain = num_epochs[args.array_id]
        samples = 50000
    else:
        num_samples = [5000, 10000, 20000, 50000]
        samples = num_samples[args.array_id]

pathpre = f"/scratch/ssd004/scratch/{os.getenv('USER')}/checkpoint"
datapath = f"/ssd003/home/{os.getenv('USER')}/data"

def load_victim(epochs, dataset, model, loss, device, retrain=False):

    print("Loading victim model: ")

    if retrain:
        print("Evaluating retrained model")
        checkpoint = torch.load(
        f"{pathpre}/SimCLR/102resnet34infonceSTEAL/retrain{dataset}_checkpoint_{epochs}_{loss}_{samples}.pth.tar", map_location=device)
    else:
        checkpoint = torch.load(
            f"{pathpre}/SimCLR/{epochs}{args.arch}{loss}TRAIN/{dataset}_checkpoint_{epochs}_{loss}.pth.tar",
            map_location=device)
    try:
        state_dict = checkpoint['state_dict']
    except:
        state_dict = checkpoint

    new_state_dict = {}
    # Remove head.
    for k in list(state_dict.keys()):
        if k.startswith('backbone.'):
            if k.startswith('backbone') and not k.startswith('backbone.fc'):
                # remove prefix
                new_state_dict[k[len("backbone."):]] = state_dict[k]
        elif k.startswith('module.backbone.'):
            if k.startswith('module.backbone') and not k.startswith('module.backbone.fc'):
                # remove prefix
                new_state_dict[k[len("module.backbone."):]] = state_dict[k]
        else:
            new_state_dict[k] = state_dict[k]

    log = model.load_state_dict(new_state_dict, strict=False)
    assert log.missing_keys == ['fc.weight', 'fc.bias']
    return model

def load_stolen(epochs, loss, model, dataset, queries, device):

    print("Loading stolen model: ")
    checkpoint = torch.load(
        f"{pathpre}/SimCLR/{epochs}{args.arch}{loss}STEAL/stolen_checkpoint_{queries}_{loss}_{dataset}.pth.tar",
        map_location=device)


    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    # Remove head.
    if loss == "symmetrized":
        for k in list(state_dict.keys()):
            if k.startswith('encoder.'):
                if k.startswith('encoder') and not k.startswith('encoder.fc'):
                    # remove prefix
                    new_state_dict[k[len("encoder."):]] = state_dict[k]
            else:
                new_state_dict[k] = state_dict[k]
    else:
        for k in list(state_dict.keys()):
            if k.startswith('backbone.'):
                if k.startswith('backbone') and not k.startswith('backbone.fc'):
                    # remove prefix
                    new_state_dict[k[len("backbone."):]] = state_dict[k]
            elif k.startswith('module.backbone.'):
                if k.startswith('module.backbone') and not k.startswith(
                        'module.backbone.fc'):
                    # remove prefix
                    new_state_dict[k[len("module.backbone."):]] = state_dict[k]
            else:
                new_state_dict[k] = state_dict[k]

    log = model.load_state_dict(new_state_dict, strict=False)
    assert log.missing_keys == ['fc.weight', 'fc.bias']
    return model

def load_victim_linear(epochs, loss, model, dataset, queries, datasetvic, device):
    # load victim to compute fidelity accuracy
    print("Loading victim model: ")

    state_dict = torch.load(
        f"{pathpre}/SimCLR/downstream/victim/{datasetvic}_down_{dataset}.pth.tar",   # Add support for other victims
        map_location=device)

    log = model.load_state_dict(state_dict, strict=False)
    #assert log.missing_keys == ['fc.weight', 'fc.bias']
    return model

def load_random(model, device, args):

    print(f"Loading {args.dataset} encoder as random model: ")

    checkpoint = torch.load(
        f"{pathpre}/SimCLR/{args.epochstrain}{args.arch}infonceTRAIN/{args.dataset}_checkpoint_{args.epochstrain}_infonce.pth.tar",
        map_location=device)

    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for k in list(state_dict.keys()):
        if k.startswith('backbone.'):
            if k.startswith('backbone') and not k.startswith('backbone.fc'):
                # remove prefix
                new_state_dict[k[len("backbone."):]] = state_dict[k]
        elif k.startswith('module.backbone.'):
            if k.startswith('module.backbone') and not k.startswith('module.backbone.fc'):
                # remove prefix
                new_state_dict[k[len("module.backbone."):]] = state_dict[k]
        else:
            new_state_dict[k] = state_dict[k]

    log = model.load_state_dict(new_state_dict, strict=False)
    assert log.missing_keys == ['fc.weight', 'fc.bias']
    return model

def get_stl10_data_loaders(download, shuffle=False, batch_size=args.batch_size, dim=32):
    train_dataset = datasets.STL10(f"{pathpre}/SimCLR/stl10", split='train', download=download,
                                  transform=transforms.Compose([transforms.Resize(dim), transforms.ToTensor()]))
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)
    test_dataset = datasets.STL10(f"{pathpre}/SimCLR/stl10", split='test', download=download,
                                  transform=transforms.Compose([transforms.Resize(dim), transforms.ToTensor()]))
    test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=2, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader

def get_cifar10_data_loaders(download, shuffle=False, batch_size=args.batch_size, dim=32):
    train_dataset = datasets.CIFAR10(datapath, train=True, download=download,
                                  transform=transforms.Compose([transforms.Resize(dim), transforms.ToTensor()]))
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)
    test_dataset = datasets.CIFAR10(datapath, train=False, download=download,
                                  transform=transforms.Compose([transforms.Resize(dim), transforms.ToTensor()]))
    indxs = list(range(len(test_dataset) - 1000, len(test_dataset)))
    test_dataset = torch.utils.data.Subset(test_dataset,
                                           indxs)  # only select last 1000 samples to prevent overlap with queried samples.
    test_loader = DataLoader(test_dataset, batch_size=64,
                            num_workers=2, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader

def get_cifar100_data_loaders(download, shuffle=False, batch_size=args.batch_size, dim=32):
    train_dataset = datasets.CIFAR100(datapath, train=True, download=download,
                                  transform=transforms.Compose([transforms.Resize(dim), transforms.ToTensor()]))
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)
    test_dataset = datasets.CIFAR100(datapath, train=False, download=download,
                                  transform=transforms.Compose([transforms.Resize(dim), transforms.ToTensor()]))
    test_loader = DataLoader(test_dataset, batch_size=64,
                            num_workers=2, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader

def get_svhn_data_loaders(download, shuffle=False, batch_size=args.batch_size):
    train_dataset = datasets.SVHN(datapath + "/SVHN", split='train', download=download,
                                  transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)
    test_dataset = datasets.SVHN(datapath + "/SVHN", split='test', download=download,
                                  transform=transforms.ToTensor())
    indxs = list(range(len(test_dataset) - 1000, len(test_dataset)))
    test_dataset = torch.utils.data.Subset(test_dataset,
                                           indxs)  # only select last 1000 samples to prevent overlap with queried samples.
    test_loader = DataLoader(test_dataset, batch_size=64,
                            num_workers=2, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader

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





if args.modeltype == "stolen":
    log_dir = f"{pathpre}/SimCLR/{args.epochs}{args.arch}{args.losstype}STEAL/"  # save logs here.
    logname = f'testing{args.modeltype}{args.dataset_test}{args.num_queries}.log'
else:
    if args.dataset == "imagenet":
        args.arch = "resnet50"
    log_dir = f"{pathpre}/SimCLR/{args.epochstrain}{args.arch}{args.losstype}TRAIN/"
    logname = f'testing{args.modeltype}{args.dataset_test}.log'
    if args.retrain:
        log_dir = f"{pathpre}/SimCLR/102resnet34infonceSTEAL/"  # manually set
        logname = f'testingretrained{args.dataset_test}{samples}_e{args.epochstrain}.log'

if args.clear == "True":
    if os.path.exists(os.path.join(log_dir, logname)):
        os.remove(os.path.join(log_dir, logname))
logging.basicConfig(
    filename=os.path.join(log_dir, logname),
    level=logging.DEBUG)

if args.arch == 'resnet18':
    model = ResNet18(num_classes=10).to(device)
elif args.arch == 'resnet34':
    model = ResNet34( num_classes=10).to(device)
elif args.arch == 'resnet50':
    if args.dataset_test == "cifar100":
        model = torchvision.models.resnet50(pretrained=False, num_classes=100).to(device)
    else:
        model = torchvision.models.resnet50(pretrained=False,
                                            num_classes=10).to(device)

if args.modeltype == "victim":
    model = load_victim(args.epochstrain, args.dataset, model, args.losstype,
                                         device=device, retrain=args.retrain)
    print("Evaluating victim")
elif args.modeltype == "random":
    model = load_random(model,device=device, args=args)
    print("Evaluating random model")
else:
    model = load_stolen(args.epochs, args.losstype, model, args.datasetsteal, args.num_queries,
                        device=device)
    print("Evaluating stolen model")

    victim_linear = ResNet18(num_classes=10).to(device)
    victim_linear = load_victim_linear(args.epochs, args.losstype, victim_linear, args.dataset_test, args.num_queries, args.dataset,
                        device=device)

if args.dataset_test == 'cifar10':
    if args.arch == "resnet50":
        train_loader, test_loader = get_cifar10_data_loaders(download=False, dim=224)
    else:
        train_loader, test_loader = get_cifar10_data_loaders(download=False)
elif args.dataset_test == 'cifar100':
    if args.arch == "resnet50":
        train_loader, test_loader = get_cifar100_data_loaders(download=False,
                                                             dim=224)
elif args.dataset_test == 'stl10':
    if args.arch == "resnet50":
        train_loader, test_loader = get_stl10_data_loaders(download=False,dim=224)
    else:
        train_loader, test_loader = get_stl10_data_loaders(download=False)
elif args.dataset_test == "svhn":
    train_loader, test_loader = get_svhn_data_loaders(download=False)

# freeze all layers but the last fc (can try by training all layers)
for name, param in model.named_parameters():
    if name not in ['fc.weight', 'fc.bias', 'fc.0.weight', 'fc.0.bias']: # the imagenet model has fc.0 for the last layer
        param.requires_grad = False

parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
assert len(parameters) == 2  # fc.weight, fc.bias

if args.modeltype == "victim":
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.0008)
    criterion = torch.nn.CrossEntropyLoss().to(device)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=0.0008)
    criterion = torch.nn.CrossEntropyLoss().to(device)
epochs = 100

## Trains the representation model with a linear classifier to measure the accuracy on the test set labels of the victim/stolen model

logging.info(f"Evaluating {args.modeltype} model on {args.dataset_test} dataset. Model trained using {args.losstype}.")
logging.info(f"Args: {args}")
for epoch in range(epochs):
    top1_train_accuracy = 0
    for counter, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        top1 = accuracy(logits, y_batch, topk=(1,))
        top1_train_accuracy += top1[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (counter+1) * x_batch.shape[0] >= args.num_labeled:
            break

    top1_train_accuracy /= (counter + 1)
    top1_accuracy = 0
    top5_accuracy = 0
    fidelity_accuracy = 0
    for counter, (x_batch, y_batch) in enumerate(test_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        if args.modeltype == "stolen":
            victim_logits = victim_linear(x_batch)
            victim_targets = victim_logits.argmax(axis=1)
            fid = accuracy(logits, victim_targets, topk=(1,))
            fidelity_accuracy += fid[0]

        top1, top5 = accuracy(logits, y_batch, topk=(1,5))
        top1_accuracy += top1[0]
        top5_accuracy += top5[0]

    top1_accuracy /= (counter + 1)
    top5_accuracy /= (counter + 1)
    fidelity_accuracy /= (counter + 1)
    if args.modeltype == "stolen":
        print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}\tFidelity: {fidelity_accuracy.item()}")
        logging.debug(
            f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}\tFidelity: {fidelity_accuracy.item()}")
    else:
        print(
            f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")
        logging.debug(
            f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")

if args.save == "True":
    if args.modeltype == "stolen":
        torch.save(model.state_dict(), f"{pathpre}/SimCLR/downstream/stolen/{args.dataset}enc/{args.datasetsteal}_down_{args.dataset_test}.pth.tar")
    elif args.modeltype == "random":
        if args.arch == "resnet50":
            torch.save(model.state_dict(),
                       f"{pathpre}/SimCLR/downstream/random/{args.dataset}_down_{args.dataset_test}_r50.pth.tar")
        else:
            torch.save(model.state_dict(),
                       f"{pathpre}/SimCLR/downstream/random/{args.dataset}_down_{args.dataset_test}.pth.tar")
    else:
        torch.save(model.state_dict(), f"{pathpre}/SimCLR/downstream/victim/{args.dataset}_down_{args.dataset_test}.pth.tar")
