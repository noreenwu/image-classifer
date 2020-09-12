import argparse
import os
import torch
from simage import process_image
from smodels import load_checkpoint

def predict(image_path, model, k=3):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''    
    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path)
    image.to("cpu")  
    model.to("cpu")
    image.unsqueeze_(0)
    image = image.float()
  
    logps = model(image)
    ps = torch.exp(logps)
    top_ps, top_class = ps.topk(k, dim=1)
    print(top_ps)
    print(top_class)
    

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
    
      model = load_checkpoint(checkpoint_file)
        
      predict(image_path, model, k)
        
if __name__ == '__main__':
    main()