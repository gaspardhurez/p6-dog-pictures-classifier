from PIL import Image
import numpy as np

def load_image_to_ndarray(image_path):
    img = Image.open(image_path)
    img_array = np.array(img)
    return img_array

def resize_image(img, output_size=(224, 224)):

    img_resized = img.resize(output_size)
    
    return img_resized
   
def apply_whitening(img_array):

    img_centered = img_array - np.mean(img_array, axis=(0, 1), keepdims=True)
    
    img_whitened = img_centered / (np.std(img_centered, axis=(0, 1), keepdims=True) + 1e-5)  
    
    return img_whitened

import cv2

def apply_histogram_equalization(img_array):
    img_equalized = np.zeros_like(img_array)
    
    for i in range(3): 
        img_equalized[..., i] = cv2.equalizeHist(img_array[..., i].astype(np.uint8))
    
    return img_equalized

