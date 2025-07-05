import matplotlib.pyplot as plt
from scipy.io import loadmat
from image_to_graph import image_to_graph
from n_cuts import n_cuts

# Load RGB images
data = loadmat("dip_hw_3.mat")
d2a = data["d2a"]  # shape (50,50,3)
d2b = data["d2b"]  # shape (50,50,3)

images = {'Image d2a': d2a, 'Image d2b': d2b}

#k values
ks = [2, 3, 4]

for img_name, img in images.items():
    print(f"Processing {img_name}")
    affinity_mat = image_to_graph(img)
    M, N, _ = img.shape
    
    #plot
    fig, axs = plt.subplots(1, len(ks)+1, figsize=(15, 5))
    axs[0].imshow(img)
    axs[0].set_title("Original Image")
    axs[0].axis('off')
    
    for i, k in enumerate(ks):
        labels = n_cuts(affinity_mat, k)
        segmented_img = labels.reshape(M, N)
        
        axs[i+1].imshow(segmented_img, cmap='jet')
        axs[i+1].set_title(f"n-cuts k={k}")
        axs[i+1].axis('off')
    
    plt.suptitle(f"Νormalized cuts - {img_name}")
    plt.show()
