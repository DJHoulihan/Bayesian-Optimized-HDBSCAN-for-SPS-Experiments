from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd

def density_aware_resample(data, n_samples, k):
    """
    Resample the dataset in a density-aware manner.
    
    Parameters:
    - data: ndarray, the full dataset (shape: [n_samples, n_features])
    - n_samples: int, number of points per subset
    - n_subsets: int, number of subsets to create
    - k: int, number of nearest neighbors for density estimation

    Returns:
    - subsets: list of ndarray, resampled subsets
    """
    
    # Estimate local density using k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k).fit(data)
    distances, _ = nbrs.kneighbors(data) # Returns indices of and distances to the neighbors of each point.
    local_density = 1 / (np.mean(distances, axis=1) + 1e-10)  # Avoid divide-by-zero
    
    # Invert densities to give higher weights to sparse regions
    # weights = 1 / (local_density + 1e-10)
    # max_weight = np.percentile(weights, 95)  # Use the 95th percentile as a threshold
    # weights = np.minimum(weights, max_weight)
    # Blend density-aware and uniform sampling
    density_weights = 1 / (local_density + 1e-10)
    density_weights /= np.sum(density_weights)  # Normalize density-based weights
    
    uniform_weights = np.ones_like(density_weights) / len(density_weights)  # Uniform weights
    alpha = 0.3  # Weight for density-aware sampling
    weights = alpha * density_weights + (1 - alpha) * uniform_weights
    weights /= np.sum(weights)  # Normalize combined weights
    # weights /= np.sum(weights)  # Normalize weights
    # scaler = StandardScaler()
    
    # Generate subsets
    subsets_unscaled = [] #;subsets_scaled = []
    
    sampled_indices = np.random.choice(len(data), size=n_samples, p=weights)
    # subsets_scaled.append(pd.DataFrame(scaler.fit_transform(data.iloc[sampled_indices]), columns = ['ScintLeft', 'AnodeBack']))
    subsets_unscaled.append(pd.DataFrame(data.iloc[sampled_indices], columns = ['ScintLeft', 'AnodeBack']))
    return  subsets_unscaled
   

