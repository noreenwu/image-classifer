import argparse
import os
import torch
from torchvision import datasets, transforms, models

from torch import nn, optim
from collections import OrderedDict

from smodels import load_train_data, create_classifier, train


parser = argparse.ArgumentParser(
  description='Train a specified architecture on specified data'
)
parser.add_argument("train_data", help="training data directory")

parser.add_argument("-a", "--arch", default="vgg16", help="pretrained architecture")
parser.add_argument("-s", "--save_dir", default="checkpoints", help="checkpoint save directory")
parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("-hu", "--hidden_units", type=int, default=256, help="hidden_units")
parser.add_argument("-e", "--epochs", type=int, default=1, help="increase output verbosity")
parser.add_argument("-g", "--gpu", action="store_true", help="use GPU")

args = parser.parse_args()

train_dir = args.train_data
train_dir_exists = os.path.isdir(train_dir)

if not train_dir_exists:
  print("Could not find training data directory, {}".format(train_dir))
  exit(1)
else:
  print("Training directory {}: ok".format(train_dir))

checkpoints_dir = args.save_dir
checkpoints_dir_exists = os.path.isdir(checkpoints_dir)
if not checkpoints_dir_exists:
  print("Could not find checkpoints directory, {}".format(checkpoints_dir))
  exit(1)
print("Checkpoints directory {}: ok".format(checkpoints_dir))

print("Will train with {} architecure".format(args.arch))
print("Learning rate is {}".format(args.learning_rate))
print("Number of hidden units: {}".format(args.hidden_units))
print("Epochs: {}".format(args.epochs))

if args.gpu:
  print("Use GPU")
else:
  print("Use CPU")


###
trainloader, testloader, class_to_idx = load_train_data("flowers", 32) 

print(class_to_idx)

model = models.vgg16(pretrained=True)           ## default; offer other choices
# model = models.densenet121(pretrained=True)

classifier = create_classifier()

# classifier = nn.Sequential(OrderedDict([
#     ('fc1', nn.Linear (25088, 1024)),
#     ('relu', nn.ReLU ()),
#     ('dropout', nn.Dropout (p = 0.2)),  # down from 0.3    
#     ('fc2', nn.Linear (1024, 102)),     # swap dropout and fc2 
#     ('output', nn.LogSoftmax (dim =1))
# ]))

model.classifier = classifier

print(model)

# exit()
# for param in model.parameters():
#    param.requires_grad = False

# for name, params in model.named_children():
#     print(name)
    


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

criterion = nn.NLLLoss()

optimizer= optim.Adam(model.classifier.parameters(), lr=0.001)    ## learning rate customizable

    
train(model, device, trainloader, testloader, optimizer, criterion, args.epochs)
    
# epochs = args.epochs
# steps = 0
# running_loss = 0
# print_every = 5
# model.to(device)

# for epoch in range(epochs):
#     for images, labels in trainloader:
#         steps += 1

#         images, labels = images.to(device), labels.to(device)

#         optimizer.zero_grad()

#         logps = model(images)
#         loss = criterion(logps, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()   

#         if steps % print_every == 0:
#             model.eval()
#             test_loss = 0
#             accuracy = 0 

#             for images, labels in testloader:
#                 images, labels = images.to(device), labels.to(device)

#                 logps = model(images)
#                 loss = criterion(logps, labels)
#                 test_loss += loss.item()         

#                 # calculate accuracy
#                 ps = torch.exp(logps)
#                 top_ps, top_class = ps.topk(1, dim=1)
#                 equality = top_class == labels.view(*top_class.shape)
#                 accuracy += torch.mean(equality.type(torch.FloatTensor)).item()                 

#                 print("Epoch: {}/{}.. ".format(epoch+1, epochs),
#                       "Training Loss: {:.3f}.. ".format(running_loss/print_every),
#                       "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
#                       "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

#                 running_loss = 0
#                 model.train()

                

model.class_to_idx = class_to_idx    # saved above

checkpoint = {
    'epochs': args.epochs,
    'state_dict': model.state_dict(),
    'map': model.class_to_idx,
    'optimizer': optimizer.state_dict(),
    'classifier': model.classifier
}

torch.save(checkpoint, 'checkpoint.pth')        


print("end")