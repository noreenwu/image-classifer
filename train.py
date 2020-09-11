import argparse as ap
import os
import torch
from torchvision import datasets, transforms, models

from torch import nn, optim
from collections import OrderedDict

from smodels import load_train_data, create_classifier, train


def define_args():    
    parser = ap.ArgumentParser(
      description='Train a specified architecture on specified data'
    )
    parser.add_argument("train_data_dir", help="training data directory")

    parser.add_argument("-a", "--arch", default="vgg16", help="pretrained architecture")
    parser.add_argument("-s", "--save_dir", default="checkpoints", help="checkpoint save directory")
    parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("-hu", "--hidden_units", type=int, default=256, help="hidden_units")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="increase output verbosity")
    parser.add_argument("-g", "--gpu", action="store_true", help="use GPU")
    
    return parser


def get_options():
    parser = define_args()
    args = parser.parse_args()    

    train_dir = args.train_data_dir

    if not os.path.isdir(train_dir):
      print("Could not find training data directory, {}".format(train_dir))
      exit(1)
    else:
      print("Training data directory: {}".format(train_dir))

    checkpoints_dir = args.save_dir
    checkpoints_dir_exists = os.path.isdir(checkpoints_dir)
    if not checkpoints_dir_exists:
      print("Could not find checkpoints directory: {}".format(checkpoints_dir))
      exit(1)
    print("Save checkpoint to directory: {}".format(checkpoints_dir))

    print("Train with architecure: {}".format(args.arch))
    print("Learning rate: {}".format(args.learning_rate))
    print("Number of hidden units: {}".format(args.hidden_units))
    print("Epochs: {}".format(args.epochs))    
    
    if args.gpu:
        specified_device = "cuda"
    else:
        specified_device = "cpu"


    if specified_device == "cuda":
        if not torch.cuda.is_available():
            print("GPU specified but not available. Sorry")
            exit(1)

    print("Device: {}".format(specified_device))
    
    return train_dir, checkpoints_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, specified_device


def main(raw_args=None):
    
    train_dir, checkpoints_dir, arch, learning_rate, hidden_units, epochs, specified_device = get_options()
    
    
    # set device to cuda, if specified
    if specified_device == "cuda":
        device = torch.device("cuda")
    
    
    # load training and test data; get the class_to_idx mapping
    trainloader, testloader, class_to_idx = load_train_data(train_dir, 32) 
    
    print(class_to_idx)
    exit()
    
    
if __name__ == '__main__':
    main()
    

    

    

print(class_to_idx)

model = models.vgg16(pretrained=True)           ## default; offer other choices
# model = models.densenet121(pretrained=True)

classifier = create_classifier()

model.classifier = classifier

print(model)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

criterion = nn.NLLLoss()

optimizer= optim.Adam(model.classifier.parameters(), lr=0.001)    ## learning rate customizable

    
train(model, device, trainloader, testloader, optimizer, criterion, args.epochs)
    

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


