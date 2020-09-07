
import argparse

parser = argparse.ArgumentParser(
  description='Train a specified architecture on specified data'
)
parser.add_argument("train_data", help="training data directory")

parser.add_argument("-a", "--arch", default="vgg16", help="pretrained architecture")
parser.add_argument("-s", "--save_dir", default="checkpoints", help="checkpoint save directory")
parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("-hu", "--hidden_units", type=int, default=256, help="hidden_units")
parser.add_argument("-e", "--epochs", type=int, help="increase output verbosity")
parser.add_argument("-g", "--gpu", action="store_true", help="use GPU")

args = parser.parse_args()



print ("data_dir (required): where train data is : {}".format(args.train_data))

## optional arguments
print("where to save checkpoints {}".format(args.save_dir))
print("architecure is {}".format(args.arch))
print("learning rate is {}".format(args.learning_rate))
print("hidden units is {}".format(args.hidden_units))
print("epochs {}".format(args.epochs))
print("use GPU? {}".format(args.gpu))

