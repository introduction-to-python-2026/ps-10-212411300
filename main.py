import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.filters import median
from skimage.morphology import disk
from scipy.signal import convolve2d # Added for edge_detection

from image_utils import load_image # Only import load_image, edge_detection will be defined here

# Redefine edge_detection to ensure correct parameter usage
def edge_detection(image_array):
    # If the image_array is already grayscale (2D), use it directly.
    # Otherwise, convert 3-channel image to grayscale.
    if image_array.ndim == 3:
        gray_image = np.mean(image_array, axis=2)
    else:
        gray_image = image_array

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


path = '/content/IMG-20260103-WA0013.jpg'
original_np = load_image(path)

gray_image = np.mean(original_np, axis=2).astype(np.uint8)
clean_image = median(gray_image, disk(3))

edgeMAG = edge_detection(clean_image)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(edgeMAG.ravel(), bins=256, color='black')
plt.title("Histogram of edgeMAG")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

threshold_value = 50
edge_binary = edgeMAG > threshold_value

plt.subplot(1, 2, 2)
plt.imshow(edge_binary, cmap='gray')
plt.title("Binary Edge Map")
plt.show()

edge_image = Image.fromarray((edge_binary * 255).astype(np.uint8))
edge_image.save('my_edges.png')
