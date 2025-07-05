import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from spectral_clustering import spectral_clustering

#load data (affinity matrix 12x12)
data = loadmat("dip_hw_3.mat")
affinity_matrix = data["d1a"]

ks = [2, 3, 4]
cluster_labels = {}

#repeat for ks
for k in ks:
    labels = spectral_clustering(affinity_matrix, k)
    cluster_labels[k] = labels

#Create barplot 
fig, axs = plt.subplots(1, len(ks), figsize=(12, 4))
for i, k in enumerate(ks):
    axs[i].bar(np.arange(12), cluster_labels[k])
    axs[i].set_title(f"Spectral Clustering (k={k})")
    axs[i].set_xlabel("Node index")
    axs[i].set_ylabel("Cluster label")
    axs[i].set_xticks(np.arange(12))
    axs[i].set_ylim(-0.5, max(cluster_labels[k]) + 0.5)

plt.tight_layout()
plt.show()
