import argparse
import os

def get_options():
    TOP_K_DEFAULT = 3
    
    parser = argparse.ArgumentParser(
      description=
        'Using a saved trained network, make a classification prediction on a given image'
    )

    parser.add_argument("image_path", help="full path to image")
    parser.add_argument("checkpoint", help="saved checkpoint file")
    parser.add_argument("-tk", "--top_k", type=int, default=TOP_K_DEFAULT, help="top k likely classifications")
    parser.add_argument("-c", "--category_names_file", default="cat_to_name.json",
                        help="json mapping of names to classifications")


    args = parser.parse_args()
    if not os.path.isfile(args.image_path):
      print("Could not find image file: {}".format(args.image_path))
      exit(1)
        
    if not os.path.isfile(args.checkpoint):
        print("Could not find checkpoint file: {}".format(args.checkpoint))
        exit(1)
    
    if not os.path.isfile(args.category_names_file):
        print("Could not find category names file: {}".format(args.category_names_file))
        exit(1)
    
    return(args.image_path, args.checkpoint, args.top_k, args.category_names_file)



def main(raw_args=None):
    
      image_path, checkpoint_file, k, cat_names = get_options()
    
      print("main got:\n {}, {}, {}, {}".format(image_path, checkpoint_file, k, cat_names))
    
          
if __name__ == '__main__':
    main()