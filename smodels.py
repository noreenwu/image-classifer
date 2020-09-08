import torch
from torchvision import datasets, transforms, models
from torch import nn
from collections import OrderedDict

def load_train_data(root_path, batch_size):
    train_transforms = transforms.Compose(
            [transforms.RandomResizedCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
             ])
    test_transforms = transforms.Compose(
            [transforms.Resize(224),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
             ])
    
    train_dir = root_path + "/train"
    test_dir = root_path + "/test"

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    return trainloader, testloader 


def create_classifier():

  classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear (25088, 1024)),
                          ('relu', nn.ReLU ()),
                          ('fc2', nn.Linear (1024, 102)),
                          ('dropout', nn.Dropout (p = 0.2)),  # down from 0.3
                          ('output', nn.LogSoftmax (dim =1))
                      ]))

  return classifier                      