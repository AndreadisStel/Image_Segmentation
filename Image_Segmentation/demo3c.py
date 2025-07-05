import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from image_to_graph import image_to_graph
from spectral_clustering import spectral_clustering
from n_cuts import n_cuts
from calculate_n_cut_value import calculate_n_cut_value
from n_cuts_recursive import n_cuts_recursive

# Threshold parameters for recursion
T1 = 100
T2 = 1

# Set K for non-recursive n-cuts and spectral clustering to compare
K = 3

# Load RGB images
data = loadmat("dip_hw_3.mat")
d2a = data["d2a"]
d2b = data["d2b"]
images = {'Image d2a': d2a, 'Image d2b': d2b}


# repeat for both images
for img_name, img in images.items():
    print(f"Processing {img_name}")
    affinity_mat = image_to_graph(img)
    M, N, _ = img.shape

    # Non-recursive
    cluster_idx_nc = n_cuts(affinity_mat, K)
    ncut_value = calculate_n_cut_value(affinity_mat, cluster_idx_nc)

    # Spectral clustering
    cluster_idx_sc = spectral_clustering(affinity_mat, K)

    # Recursive n-cuts
    cluster_idx_recursive = n_cuts_recursive(affinity_mat, T1=T1, T2=T2)
    unique_clusters = np.unique(cluster_idx_recursive)
    
    # Plotting for comparison
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(img)
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    axs[1].imshow(cluster_idx_nc.reshape(M, N), cmap='jet')
    axs[1].set_title(f"n-cuts (k = {K})")
    axs[1].axis('off')

    axs[2].imshow(cluster_idx_sc.reshape(M, N), cmap='jet')
    axs[2].set_title(f"Spectral Clustering (k={K})")
    axs[2].axis('off')

    axs[3].imshow(cluster_idx_recursive.reshape(M, N), cmap='nipy_spectral')
    axs[3].set_title(f"Recursive n-cuts\nClusters: {len(unique_clusters)}")
    axs[3].axis('off')

    plt.suptitle(f"Clustering Comparison on {img_name}")
    plt.tight_layout()
    plt.show()

    print(f"{img_name} - Ncut (k = {K})")
    print(f"{img_name} - Recursive n-cuts: {len(unique_clusters)} clusters")
    print()
