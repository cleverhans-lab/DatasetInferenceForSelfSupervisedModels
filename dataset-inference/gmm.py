import torch
import sys
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Subset
from sklearn.mixture import GaussianMixture
import pickle
from scipy import stats
import argparse


parser = argparse.ArgumentParser(description='dataset_inference')
parser.add_argument('--model_file', type=str, help='location of the model')
parser.add_argument('--imagenet_location', type=str, help='location of the ImageNet dataset')
parser.add_argument('--index_file_1', type=str, help='location of index file for training 1')
parser.add_argument('--index_file_2', type=str, help='location of index file for training 2')
parser.add_argument('--num_dim', type=int, default=2048)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--num_component', type=int, default=20)



args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


#Load Model
model = models.resnet50(pretrained=False, num_classes=10)
checkpoint = torch.load(args.model_file, map_location=device)
state_dict = checkpoint['state_dict']

for k in list(state_dict.keys()):
        # retain only encoder up to before the embedding layer
    if k.startswith('module.encoder') and not k.startswith('module.encoder.fc'):
            # remove prefix
        state_dict[k[len("module.encoder."):]] = state_dict[k]
        # delete renamed or unused k
    del state_dict[k]


log = model.load_state_dict(state_dict, strict=False)
assert log.missing_keys == ['fc.weight', 'fc.bias']

model.fc = torch.nn.Identity()

for name, param in model.named_parameters():
    param.requires_grad = False

model = model.to(device)

model.eval()

print("Finish loading models")


data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=224),transforms.ToTensor()])


#Load ImageNet 

def get_imagenet_data_loaders(batch_size):
  index_1 = np.loadtxt(args.index_file_1, dtype=int)
  index_2 = np.loadtxt(args.index_file_2, dtype=int)
  train_dataset = datasets.ImageNet(args.imagenet_location, split="train", transform=data_transforms)
  train_dataset_1 = Subset(train_dataset, index_1)
  train_dataset_2 = Subset(train_dataset, index_2)

  train_loader_1 = DataLoader(train_dataset_1, batch_size=batch_size,
                            num_workers=8, drop_last=False, shuffle=True)

  train_loader_2 = DataLoader(train_dataset_2, batch_size=batch_size,
                            num_workers=8, drop_last=False, shuffle=True)

  test_dataset = datasets.ImageNet(args.imagenet_location, split="val", transform=data_transforms)

  test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            num_workers=8, drop_last=False, shuffle=True)
  return train_loader_1, train_loader_2, test_loader


train_loader_1, train_loader_2, test_loader = get_imagenet_data_loaders(args.batch_size)

print("Finish loading data")

training_representations_1 = torch.zeros(100000, args.num_dim)
training_representations_2 = torch.zeros(50000, args.num_dim)
test_representations = torch.zeros(50000, args.num_dim)


for i, (x_batch, _) in enumerate(train_loader_1):
  x_batch = x_batch.to(device)
  r = model(x_batch)
  training_representations_1[i * args.batch_size: (i+1)*args.batch_size] = r

print("Finish loading training representations 1")

for i, (x_batch, _) in enumerate(train_loader_2):
  x_batch = x_batch.to(device)
  r = model(x_batch)
  training_representations_2[i * args.batch_size: (i+1)*args.batch_size] = r

print("Finish loading training representations 2")


for i, (x_batch, _) in enumerate(test_loader):
  x_batch = x_batch.to(device)
  r = model(x_batch)
  test_representations[i * args.batch_size: (i+1)*args.batch_size] = r

training_representations_1 = F.normalize(training_representations_1)
training_representations_2 = F.normalize(training_representations_2)
test_representations = F.normalize(test_representations)

training_representations_1 = training_representations_1.cpu().detach().numpy()
training_representations_2 = training_representations_2.cpu().detach().numpy()
test_representations = test_representations.cpu().detach().numpy()

print("Finish loading representations")


gm = GaussianMixture(n_components=args.num_component, max_iter=1000, covariance_type="diag")
gm.fit(training_representations_1)


print("Finish fitting GMM")

training_likelihood_1 = gm.score_samples(training_representations_1)
training_likelihood_2 = gm.score_samples(training_representations_2)
test_likelihood = gm.score_samples(test_representations)

print("training likelihood 1: " + str(np.mean(training_likelihood_1)))
print("training likelihood 2: " + str(np.mean(training_likelihood_2)))
print("test likelihood: " + str(np.mean(test_likelihood)))
print(str(np.mean(training_likelihood_2) - np.mean(test_likelihood)))
print(stats.ttest_ind(training_likelihood_2, test_likelihood)[1]/2)

















