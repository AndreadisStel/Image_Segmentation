import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from image_to_graph import image_to_graph
from spectral_clustering import spectral_clustering

# Load data
data = loadmat("dip_hw_3.mat")
d2a = data["d2a"]  # (50, 50, 3)
d2b = data["d2b"]

# reshape to plot 
def visualize_segmentation(labels, image_shape, k, title):
    labels = labels.reshape(image_shape)
    plt.imshow(labels, cmap='tab10')
    plt.title(f"{title} - k={k}")
    plt.axis('off')

# Fix the way is implemented
for img_id, img in enumerate([d2a, d2b], start=1):
    affinity = image_to_graph(img)
    print(f"Affinity matrix for d2{chr(ord('a')+img_id-1)} shape: {affinity.shape}")
    
    plt.figure(figsize=(12, 4))
    for i, k in enumerate([2, 3, 4], start=1):
        labels = spectral_clustering(affinity, k)
        plt.subplot(1, 3, i)
        visualize_segmentation(labels, (50, 50), k, f"d2{chr(ord('a')+img_id-1)}")
    plt.suptitle(f"Spectral Clustering Results on d2{chr(ord('a')+img_id-1)}")
    plt.tight_layout()
    plt.show()
