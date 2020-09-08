import argparse
import os


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



# print ("data_dir (required): where train data is : {}".format(args.train_data))

## optional arguments
# print("where to save checkpoints {}".format(args.save_dir))



# check whether specified training directory and checkpoint save directory exist

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


