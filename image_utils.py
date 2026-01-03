from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(file_path):
    image = Image.open(file_path)
    np_image = np.asarray(image)
    return np_image 
    
def edge_detection(image_array):
    gray_image = np.mean(image_array, axis=2)
    
    kernelY = np.array([[ 1,  0, -1],
                        [ 2,  0, -2],
                        [ 1,  0, -1]])
    
  
    kernelX = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])
    
    edgeY = convolve2d(gray_image, kernelY, mode='same', boundary='fill', fillvalue=0)
    edgeX = convolve2d(gray_image, kernelX, mode='same', boundary='fill', fillvalue=0)
  
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    
    return edgeMAG
