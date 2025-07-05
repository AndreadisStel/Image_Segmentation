import numpy as np


def assoc(W: np.ndarray, X: np.ndarray, Y: np.ndarray) -> float:
    
    submatrix = W[X[:, None], Y]  # i belongs in X, j belongs in Y, submatrix with all of those 
    return np.sum(submatrix)



def calculate_n_cut_value(affinity_mat: np.ndarray, cluster_idx: np.ndarray) -> float:
    
    # just followning paper's formula for nasssoc calc
    W = affinity_mat
    A_idx = np.where(cluster_idx == 0)[0]
    B_idx = np.where(cluster_idx == 1)[0]
    V_idx = np.arange(W.shape[0])
    
   

    assoc_AA = assoc(W, A_idx, A_idx)
    assoc_BB = assoc(W, B_idx, B_idx)
    assoc_AV = assoc(W, A_idx, V_idx)
    assoc_BV = assoc(W, B_idx, V_idx)
    
    # check denominator = 0
    nassoc = (assoc_AA / assoc_AV if assoc_AV > 0 else 0) + (assoc_BB / assoc_BV if assoc_BV > 0 else 0)
    n_cut = 2 - nassoc
    return n_cut
