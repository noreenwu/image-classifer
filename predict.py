import argparse

parser = argparse.ArgumentParser(
  description=
    'Using a saved trained network, make a classification prediction on a given image'
)

parser.add_argument("image_path", help="full path to image")
parser.add_argument("checkpoint", help="saved checkpoint file")
parser.add_argument("-tk", "--top_k", type=int, default=3, help="top k likely classifications")
parser.add_argument("-c", "--category_names", default="cat_to_name.json",
                    help="json mapping of names to classifications")


args = parser.parse_args()

# required
print("image to predict is {}".format(args.image_path))
print("checkpoint file is {}".format(args.checkpoint))

# optional
print("top_k is {}".format(args.top_k))
print("category names file is {}".format(args.category_names))
