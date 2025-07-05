import numpy as np
from n_cuts import n_cuts
from calculate_n_cut_value import calculate_n_cut_value


def n_cuts_recursive(affinity_mat: np.ndarray, T1: int, T2: float) -> np.ndarray:
    
    n = affinity_mat.shape[0]
    cluster_idx = np.zeros(n, dtype=int)  # initialize on 0 cluster
    
    # minimum cluster size threshold T1
    def recursive_partition(indices, current_label):
        if len(indices) < T1:
            return
        
        # sub affinity matrix 
        W_sub = affinity_mat[np.ix_(indices, indices)]
        
        # cut in 2
        labels_sub = n_cuts(W_sub, k=2)
        
        # find ncut_values
        ncut_val = calculate_n_cut_value(W_sub, labels_sub)
        #print(ncut_val)

        # cut quality threshold T2
        if ncut_val > T2:
            return
        
        cluster_0_idx = [indices[i] for i in range(len(indices)) if labels_sub[i] == 0]
        cluster_1_idx = [indices[i] for i in range(len(indices)) if labels_sub[i] == 1]
        
        #print(len(cluster_0_idx))
        #print(len(cluster_1_idx))
        if len(cluster_0_idx) < T1 or len(cluster_1_idx) < T1:
            return
        
        
        cluster_idx[cluster_1_idx] = current_label + 1  
        
        # Recursion
        recursive_partition(cluster_0_idx, current_label)
        recursive_partition(cluster_1_idx, current_label + 1)
    
    # repeat until < T1 or > T2
    recursive_partition(np.arange(n), 0)
    
    
    return cluster_idx