import argparse
import os
import torch

from simage import process_image
from smodels import (load_checkpoint, get_idx_to_class, 
                     get_flower_name, get_flowername_mapping, preds_to_flower_names)
from utils import check_device



def predict(image_path, model, device, k=3):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''    
    if device == "cuda":
        image.to(device)
        model.to(device)
        
    image = process_image(image_path)

    image.unsqueeze_(0)
    image = image.float()
         
    logps = model(image)
    ps = torch.exp(logps)
    top_ps, top_class = ps.topk(k, dim=1)
    
    return top_ps, top_class

    

def get_options():
    TOP_K_DEFAULT = 1
    
    parser = argparse.ArgumentParser(
      description=
        'Using a saved trained network, make a classification prediction on a given image'
    )

    parser.add_argument("image_path", help="full path to image")
    parser.add_argument("checkpoint", help="saved checkpoint file")
    parser.add_argument("-tk", "--top_k", type=int, default=TOP_K_DEFAULT, help="top k likely classifications")
    parser.add_argument("-c", "--category_names_file", default="cat_to_name.json",
                        help="json mapping of names to classifications")
    parser.add_argument("-g", "--gpu", action="store_true", help="use GPU")

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
        
    specified_device = check_device(args.gpu)

    print("Device: {}".format(specified_device))        
    
    return(args.image_path, args.checkpoint, args.top_k, args.category_names_file, specified_device)



def main(raw_args=None):
    
      image_path, checkpoint_file, k, cat_names_file, specified_device = get_options()
    
#       print("main got:\n {}, {}, {}, {}".format(image_path, checkpoint_file, k, cat_names))
    
      model = load_checkpoint(checkpoint_file, specified_device)
        
      top_ps, top_class = predict(image_path, model, specified_device, k)
    
      confidence = top_ps.tolist()[0][0] * 100
      
      predicted_cat = top_class.tolist()[0][0]        
        
      predicted_set = top_class.tolist()[0]
      probability_set = top_ps.tolist()[0]
      print(probability_set)
      print(predicted_set)
        
      idx_to_class = get_idx_to_class(model.class_to_idx)
        
      cat_map = get_flowername_mapping(cat_names_file)
    
      flower_name = get_flower_name(idx_to_class, cat_map, predicted_cat)
    
      print(preds_to_flower_names(predicted_set, idx_to_class, cat_map))    
      print("I can say with {} confidence that this flower\'s name is: \"{}\"".format(round(confidence,2), flower_name))  

        
if __name__ == '__main__':
    main()