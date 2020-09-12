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
            [transforms.Resize(255),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
             ])
    
    train_dir = root_path + "/train"
    test_dir = root_path + "/test"
    print("train_dir is ", train_dir)
    print("test_dir is ", test_dir)    
    
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    return trainloader, testloader, train_data.class_to_idx



def create_classifier():
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear (25088, 1024)),
        ('relu', nn.ReLU ()),
        ('dropout', nn.Dropout (p = 0.2)),  # down from 0.3    
        ('fc2', nn.Linear (1024, 102)),     # swap dropout and fc2 
        ('output', nn.LogSoftmax (dim =1))
    ]))
    
    return classifier
    

def train(model, device, trainloader, testloader, optimizer, criterion, epochs=1):
    
    steps = 0
    running_loss = 0
    print_every = 5
    model.to(device)

    for epoch in range(epochs):
        for images, labels in trainloader:
            steps += 1

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()   

            if steps % print_every == 0:
                model.eval()
                test_loss = 0
                accuracy = 0 

                for images, labels in testloader:
                    images, labels = images.to(device), labels.to(device)

                    logps = model(images)
                    loss = criterion(logps, labels)
                    test_loss += loss.item()         

                    # calculate accuracy
                    ps = torch.exp(logps)
                    top_ps, top_class = ps.topk(1, dim=1)
                    equality = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equality.type(torch.FloatTensor)).item()                 

                print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

                running_loss = 0
                model.train()
                                 

def save_checkpoint(epochs, model, optimizer):
    
    checkpoint = {
        'epochs': epochs,
        'state_dict': model.state_dict(),
        'map': model.class_to_idx,
        'optimizer': optimizer.state_dict(),
        'classifier': model.classifier
    }

    torch.save(checkpoint, 'checkpoint.pth')    
    
    
def load_checkpoint(filepath, device):
    if not torch.cuda.is_available():
        print("GPU is not available. Please enable and re-run script.")
        exit(1)
                  
    checkpoint = torch.load(filepath)    
    model = models.vgg16(pretrained=True)
    # freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['map']
    model.state_dict = checkpoint['state_dict']
    model.optimizer = checkpoint['optimizer']
    model.eval()
    
    return model
  