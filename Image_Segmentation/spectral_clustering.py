import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

def spectral_clustering(
    affinity_mat: np.ndarray,
    k: int
) -> np.ndarray:

    #Calculate the D matrix
    D = np.diag(affinity_mat.sum(axis=1))

    #Calc the Laplacian
    L = D - affinity_mat
    
    #Find eigenvalues and eigenvectors (using eigsh cause Laplacian is symmetric and has only real part, smallest k in magnitude) 
    eigenvals, eigenvecs = eigsh(L, k = k, which='SM')      

    #if really needed use eigs() and use the real part of eigenvectors
    #eigenvals, eigenvecs = eigs(L, k = k, which='SM')
    #eigenvecs = eigenvecs.real
    U = eigenvecs
    
    #KMeans
    kmeans = KMeans(n_clusters = k, random_state = 1) #Same for every test
    kmeans.fit(U)
    labels = kmeans.labels_

    return labels
    