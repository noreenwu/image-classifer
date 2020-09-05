from PIL import Image
import numpy as np


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    
    width, height = im.size
    left = (width - 224)/2
    top = (height - 224)/2
    right = left + 224
    bottom = top + 224
    cropped = im.crop((left, top, right, bottom))
    
    np_image = np.array(cropped)

    np_image = np.array(np_image) / 255
    
    # normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
                
    np_image = np_image.transpose((2, 0, 1))        
    
    final_image = torch.from_numpy(np_image)
    return final_image

# img_filename = '/Users/noreen/aipnd-project/flowers/test' + '/1/image_06743.jpg'
# img_filename = '/Users/noreen/aipnd-project/flowers/test' + '/1/image_06752.jpg'
# img_filename = '/Users/noreen/aipnd-project/flowers/test' + '/1/image_06754.jpg'
# img_filename = '/Users/noreen/aipnd-project/flowers/test' + '/1/image_06760.jpg'
# img_filename = '/Users/noreen/aipnd-project/flowers/test' + '/1/image_06764.jpg'
# img_filename = '/Users/noreen/aipnd-project/flowers/test' + '/3/image_06641.jpg'
# img_filename = '/Users/noreen/aipnd-project/flowers/test' + '/1/image_06764.jpg'
# img_filename = '/Users/noreen/aipnd-project/flowers/test' + '/101/image_07988.jpg'
# img_filename = '/Users/noreen/aipnd-project/flowers/test' + '/101/image_07952.jpg'
# img_filename = '/Users/noreen/aipnd-project/flowers/test' + '/14/image_06091.jpg'
img_filename = '/Users/noreen/aipnd-project/flowers/train' + '/1/image_06735.jpg'

print(img_filename)
myimage = process_image(img_filename)
imshow(myimage)
