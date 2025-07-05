import numpy as np
from scipy.spatial.distance import cdist


def image_to_graph(
    img_array: np.ndarray,
) -> np.ndarray:

    # Reshape the array for calculations
    M, N, C = img_array.shape
    pixels = img_array.reshape(-1, C)  

    # Find the distances of every pixel from the others (including itself)
    distances = cdist(pixels, pixels, metric='euclidean')

    # Find Affinity matrix 
    affinity_matrix = 1 / np.exp(distances)

    return affinity_matrix