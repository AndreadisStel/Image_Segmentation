import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans


def n_cuts(affinity_mat: np.ndarray, k: int) -> np.ndarray:
    
    D = np.diag(affinity_mat.sum(axis=1))
    L = D - affinity_mat
    
    # !!! Lx = λDx
    eigenvals, eigenvecs = eigsh(L, M=D, k=k, which='SM')

    U = eigenvecs
    
    # k-means
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(U)
    cluster_idx = kmeans.labels_
    
    return cluster_idx
