import torch
from PIL import Image
import numpy as np

def get_new_dim(width, height, min_length=256):
    percent_scale =  min_length / min(width, height) 
    new_width = round(percent_scale * width)
    new_height = round(percent_scale * height)
    
    return new_width, new_height
    
def thumb_image(image):
    
    im = Image.open(image)
    orig_width, orig_height = im.size
    new_width, new_height = get_new_dim(orig_width, orig_height)
    
    newsize = (new_width, new_height)
    im.thumbnail(newsize)
    left = round((new_width - 224)/6)

    top = round((new_height - 224)/6)
    right = left + 224
    bottom = top + 224
    cropped = im.crop((left, top, right, bottom)) 
#     cropped.show()
    return cropped
 
    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch    model,
     returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model

    cropped = thumb_image(image)
    
    np_image = np.array(cropped)

    np_image = np.array(np_image) / 255
    
    # normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
                
    np_image = np_image.transpose((2, 0, 1))        
    
    final_image = torch.from_numpy(np_image)
    return final_image    