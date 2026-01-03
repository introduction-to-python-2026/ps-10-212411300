import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.filters import median
from skimage.morphology import disk

from image_utils import load_image 


path = '/content/sample_data/IMG-20260103-WA0013.jpg'
original_img = load_image(path)

gray_img = np.mean(original_img, axis=2)

clean_image = median(gray_img, disk(3))

edgeMAG = edge_detection(original_img)


plt.figure(figsize=(10, 4))
plt.hist(edgeMAG.ravel(), bins=256, range=(0, 100))
plt.title("Histogram of Edge Intensities")
plt.show()

threshold = 40
edge_binary = edgeMAG > threshold

plt.imshow(edge_binary, cmap='gray')
plt.title(f"Edges with Threshold = {threshold}")
plt.axis('off')
plt.show()

edge_image = Image.fromarray((edge_binary * 255).astype(np.uint8))
edge_image.save('my_edges.png')

