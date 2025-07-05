import matplotlib.pyplot as plt
from scipy.io import loadmat
from image_to_graph import image_to_graph
from spectral_clustering import spectral_clustering
from n_cuts import n_cuts
from calculate_n_cut_value import calculate_n_cut_value

# Load Data
data = loadmat("dip_hw_3.mat")
d2a = data["d2a"]
d2b = data["d2b"]
images = {'Image d2a': d2a, 'Image d2b': d2b}

ks = 2

#repeat for both images
for img_name, img in images.items():
    print(f"Processing {img_name}")
    affinity_mat = image_to_graph(img)
    M, N, _ = img.shape

    # recursive n-cut for one step, so we just break for k = 2
    cluster_idx_nc = n_cuts(affinity_mat, k = 2)
    ncut_value = calculate_n_cut_value(affinity_mat, cluster_idx_nc)

    # Spectral clustering k = 2
    cluster_idx_sc = spectral_clustering(affinity_mat, k = 2)

    # plotting / show side by side to compare
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img)
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    axs[1].imshow(cluster_idx_nc.reshape(M, N), cmap='jet')
    axs[1].set_title(f"Recursive n-cuts (1 step)\nNcut value: {ncut_value:.4f}")
    axs[1].axis('off')

    axs[2].imshow(cluster_idx_sc.reshape(M, N), cmap='jet')
    axs[2].set_title("Spectral Clustering (k=2)")
    axs[2].axis('off')

    plt.suptitle(f"Comparison on {img_name}")
    plt.show()

    print(f"{img_name} - Ncut value: {ncut_value:.4f}")
    print()
