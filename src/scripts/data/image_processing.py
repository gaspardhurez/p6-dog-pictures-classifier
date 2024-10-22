from PIL import Image
import numpy as np
import cv2 

def load_image_to_ndarray(image_path):
    img = Image.open(image_path)
    img_array = np.array(img)
    return img_array

def resize_image(img, output_size=(224, 224)):
    img_resized = cv2.resize(img, output_size, interpolation=cv2.INTER_AREA)
    return img_resized

def normalize_image(img):
    img_normalized = img / 255.0
    return img_normalized
   
def whiten_image(img_array):

    img_centered = img_array - np.mean(img_array, axis=(0, 1), keepdims=True)
    
    img_whitened = img_centered / (np.std(img_centered, axis=(0, 1), keepdims=True) + 1e-5)  
    
    return img_whitened


def equalize_image(img_array):
    img_equalized = np.zeros_like(img_array)
    
    for i in range(3): 
        img_equalized[..., i] = cv2.equalizeHist(img_array[..., i].astype(np.uint8))
    
    return img_equalized

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,      # Rotation aléatoire des images jusqu'à 20 degrés
    width_shift_range=0.2,  # Décalage horizontal jusqu'à 20%
    height_shift_range=0.2, # Décalage vertical jusqu'à 20%
    shear_range=0.2,        # Cisaillement jusqu'à 20%
    zoom_range=0.2,         # Zoom avant ou arrière jusqu'à 20%
    horizontal_flip=True,   # Inverser horizontalement les images
    fill_mode='nearest'     # Mode de remplissage pour les pixels manquants après une transformation
)

def augment_image(img_array):

    img_array = img_array.reshape((1,) + img_array.shape) 
    augmented_images = []
    
    i = 0
    for batch in datagen.flow(img_array, batch_size=1):
        augmented_images.append(batch[0])
        i += 1
        if i >= 5:
            break
    
    return augmented_images

