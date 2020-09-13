import argparse as ap
import os
import torch
from torchvision import datasets, transforms, models

from torch import nn, optim
from collections import OrderedDict

from smodels import (load_train_data, create_classifier_vgg, create_classifier_densenet, 
                     train, save_checkpoint)

from utils import check_device

def define_args():
    LEARN_RATE_DEFAULT = 0.001
    EPOCHS_DEFAULT = 1
    HIDDEN_UNITS_DEFAULT = 256
    
    parser = ap.ArgumentParser(
      description='Train a specified architecture on specified data'
    )
    parser.add_argument("train_data_dir", help="training data directory")

    parser.add_argument("-a", "--arch", default="vgg16", help="pretrained architecture: vgg or densenet")
    parser.add_argument("-s", "--save_dir", default="checkpoints", help="checkpoint save directory")
    parser.add_argument("-l", "--learning_rate", type=float, default=LEARN_RATE_DEFAULT, help="learning rate")
    parser.add_argument("-hu", "--hidden_units", type=int, default=HIDDEN_UNITS_DEFAULT, help="hidden_units")
    parser.add_argument("-e", "--epochs", type=int, default=EPOCHS_DEFAULT, help="increase output verbosity")
    parser.add_argument("-g", "--gpu", action="store_true", help="use GPU")
    
    return parser


def get_options():
    parser = define_args()
    args = parser.parse_args()    

    # a valid training data directory must be specified
    train_dir = args.train_data_dir    
    if not os.path.isdir(train_dir):
      print("Could not find training data directory: {}".format(train_dir))
      exit(1)
    else:
      print("Training data directory: {}".format(train_dir))

    # a valid checkpoints directory may be specified, otherwise "checkpoints" will be used
    checkpoints_dir = args.save_dir
    if not os.path.isdir(checkpoints_dir):
      print("Could not find checkpoints directory: {}".format(checkpoints_dir))
      exit(1)
    print("Save checkpoint to directory: {}".format(checkpoints_dir))
    print("Will save checkpoint to {}".format(checkpoints_dir + '/' + args.arch + '-checkpoint.pth'))

    
    # TODO: alternative architectures may be specified
    if args.arch != "vgg16" and args.arch != "densenet121":
        print ("Specified architecture, {}, is not supported. Please specify either vgg16 or densenet121".format(args.arch))
        exit(1)
    print("Train with architecure: {}".format(args.arch))        
            
    # the learning rate may be specified; the default is 0.001
    print("Learning rate: {}".format(args.learning_rate))    
    
    print("Number of hidden units: {}".format(args.hidden_units))
    print("Epochs: {}".format(args.epochs))    
    
    specified_device = check_device(args.gpu)

    print("Device: {}".format(specified_device))
    
    
    return (
        train_dir, checkpoints_dir, args.arch, args.learning_rate, 
        args.hidden_units, args.epochs, specified_device )        



def main(raw_args=None):
    BATCH_SIZE = 32
    train_dir, save_dir, arch, learning_rate, hidden_units, epochs, specified_device = get_options()
        
    # set device to cuda, if specified
    if specified_device == "cuda":
        device = torch.device("cuda")
        
    # load training and test data; get the class_to_idx mapping
    trainloader, testloader, class_to_idx = load_train_data(train_dir, BATCH_SIZE) 
    

    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
        classifier = create_classifier_vgg() 
    elif arch == "densenet121":
        model = models.densenet121(pretrained=True)
        classifier = create_classifier_densenet()
    else:
        print("Fatal: other models not supported")
        exit(1)


    model.classifier = classifier

    print(model)    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
        
    criterion = nn.NLLLoss()

    optimizer= optim.Adam(model.classifier.parameters(), lr=learning_rate)    ## learning rate customizable
    
    train(model, device, trainloader, testloader, optimizer, criterion, epochs)
    
    model.class_to_idx = class_to_idx    # saved above    
    
    model.to("cpu")           # better to save checkpoint for cpu
    save_checkpoint(epochs, model, optimizer, save_dir, arch)

    
if __name__ == '__main__':
    main()
    


