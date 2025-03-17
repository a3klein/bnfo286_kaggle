import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

import torch
import torch.nn as nn

def nmf_impute_pytorch(df, rank=10, max_epochs=500, lr=1e-2, device='cuda'):

    # ---------------------------
    # 1) PREPARE DATA & MASK
    # ---------------------------

    # Convert to numpy array
    X_numpy = df.values.astype(np.float32)  # shape (N, 72)
    # Create a mask of observed entries (1 where not NaN, 0 where NaN)
    mask_numpy = ~np.isnan(X_numpy)  # True for observed, False for missing
    # Replace NaNs with 0 just as a placeholder (won't affect the masked loss)
    X_numpy_noNaN = np.nan_to_num(X_numpy, nan=0.0)

    N, F = X_numpy_noNaN.shape  # F should be 72

    # Move data & mask to Torch Tensors on the chosen device
    X_torch = torch.tensor(X_numpy_noNaN, device=device)
    mask_torch = torch.tensor(mask_numpy, dtype=torch.float32, device=device)

    # ---------------------------
    # 2) INITIALIZE W & H
    # ---------------------------
    # We'll define them as unconstrained parameters but then enforce non-negativity in forward pass
    # Another approach: initialize them directly as positive, e.g. torch.rand(...), but
    # we'll want them to remain non-negative each iteration.

    # "Requires grad" parameters:
    W_param = nn.Parameter(torch.randn(N, rank, device=device) * 0.1)
    H_param = nn.Parameter(torch.randn(rank, F, device=device) * 0.1)

    # Alternatively, we could do "W_param.data.clamp_(min=0)" after each update or use ReLU in forward pass.

    # ---------------------------
    # 3) SETUP OPTIMIZER & LOSS
    # ---------------------------
    optimizer = torch.optim.Adam([W_param, H_param], lr=lr)

    # We'll define a small function that computes the masked reconstruction loss
    def masked_mse(X, W, H, mask):
        # Reconstruct
        X_hat = W @ H
        # Mask out missing positions => only compute MSE where mask=1
        diff = (X - X_hat) * mask
        mse = torch.sum(diff ** 2) / torch.sum(mask)
        return mse, X_hat

    # ---------------------------
    # 4) TRAINING LOOP
    # ---------------------------
    for epoch in range(max_epochs):
        optimizer.zero_grad()

        # Enforce non-negativity via ReLU or exponent
        W = torch.relu(W_param)    # shape (N, rank)
        H = torch.relu(H_param)    # shape (rank, F)

        loss, _ = masked_mse(X_torch, W, H, mask_torch)

        loss.backward()
        optimizer.step()

        # Optional: print progress every 50 epochs
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}/{max_epochs}, Loss = {loss.item():.4f}")

    # ---------------------------
    # 5) IMPUTE MISSING ENTRIES
    # ---------------------------
    # Final W, H
    W_final = torch.relu(W_param).detach()
    H_final = torch.relu(H_param).detach()

    X_hat_final = W_final @ H_final  # shape (N, 72)

    # Move back to CPU for constructing the DataFrame
    X_hat_final_cpu = X_hat_final.cpu().numpy()

    # Construct a new numpy array that includes original observed values,
    # but replaces NaN with the reconstruction for missing entries:
    X_imputed = X_numpy.copy()
    X_imputed[~mask_numpy] = X_hat_final_cpu[~mask_numpy]

    # ---------------------------
    # 6) RETURN AS DATAFRAME
    # ---------------------------
    imputed_df = pd.DataFrame(X_imputed, index=df.index, columns=df.columns)
    return imputed_df

def dtw_alignment(x, y):
    """
    A simple DTW implementation returning both the minimal cost path and the total cost.
    x and y are 1D numpy arrays.
    """
    n, m = len(x), len(y)
    cost = np.zeros((n+1, m+1), dtype=float)
    cost[0, 1:] = np.inf
    cost[1:, 0] = np.inf

    # Populate cost matrix
    for i in range(1, n+1):
        for j in range(1, m+1):
            dist = (x[i-1] - y[j-1])**2
            cost[i, j] = dist + min(cost[i-1, j],    # deletion
                                   cost[i, j-1],    # insertion
                                   cost[i-1, j-1])  # match

    # Backtrack to find the alignment path
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        directions = [cost[i-1, j], cost[i, j-1], cost[i-1, j-1]]
        tb = np.argmin(directions)
        if tb == 0:
            i -= 1
        elif tb == 1:
            j -= 1
        else:
            i -= 1
            j -= 1

    path.reverse()
    return path, cost[n, m]


def warp_rep2_to_rep1(rep2_matrix, path, n_ref):
    """
    Given:
      - rep2_matrix: shape (N, M2), replicate 2 data for N proteins across M2 fractions
      - path: list of (i, j) indices from DTW (i in [0..n_ref-1], j in [0..M2-1])
      - n_ref: the length of replicate 1's fraction axis

    Returns:
      - rep2_warped: shape (N, n_ref), replicate 2 data aligned (warped) to replicate 1's fraction indices.
    """
    from collections import defaultdict
    # Map each i in replicate 1 to all j's from replicate 2 that aligned via DTW
    i_to_js = defaultdict(list)
    for (i, j) in path:
        i_to_js[i].append(j)

    N = rep2_matrix.shape[0]
    rep2_warped = np.zeros((N, n_ref), dtype=float)

    for i in range(n_ref):
        js = i_to_js[i]
        if len(js) == 0:
            # If no j matches this i, set to 0 or np.nan
            rep2_warped[:, i] = 0.0
        else:
            # Average across all matched j's
            rep2_warped[:, i] = rep2_matrix[:, js].mean(axis=1)

    return rep2_warped


def align_and_merge_with_preDTW_smoothing(replicate1_df, replicate2_df, sigma=1.0):
    """
    Smooth replicate data (pandas DataFrames) with a Gaussian filter *before* DTW,
    then align replicate 2 to replicate 1, and finally average them.
    
    Parameters
    ----------
    replicate1_df : pd.DataFrame
        Co-fractionation data for N proteins (rows) x F fractions (columns). 
        Index = protein names, columns = fraction labels (e.g., "repl1_1", "repl1_2", ...).
    replicate2_df : pd.DataFrame
        Co-fractionation data for the same N proteins (rows) x F2 fractions (columns). 
        Index = protein names, columns = fraction labels (e.g., "repl2_1", "repl2_2", ...).
    sigma : float
        Standard deviation for the Gaussian kernel in smoothing.
        
    Returns
    -------
    merged_preDTW_df : pd.DataFrame
        The final merged profiles for each protein (rows) after smoothing + DTW alignment + averaging,
        having the same shape as replicate1_df and the same row/column labels.
    """

    # Check that replicate1_df and replicate2_df have the same index (same proteins in the same order)
    # If not the same, you may want to intersect or reorder. For now, let's assume they match.
    if not replicate1_df.index.equals(replicate2_df.index):
        raise ValueError("Protein indices (rows) in replicate1_df and replicate2_df do not match. "
                         "Reindex or merge them before alignment.")

    # Convert to numpy arrays for processing
    repl1_arr = replicate1_df.values  # shape (N, F)
    repl2_arr = replicate2_df.values  # shape (N, F2)

    # 1) Smooth each replicate across the fraction axis
    repl1_smoothed = gaussian_filter1d(repl1_arr, sigma=sigma, axis=1)
    repl2_smoothed = gaussian_filter1d(repl2_arr, sigma=sigma, axis=1)

    # 2) Create representative 1D profiles from the smoothed data
    # Using np.nanmean to be robust if there are any NaNs. Adjust as needed.
    profile1 = np.nanmean(repl1_smoothed, axis=0)  # shape (F,)
    profile2 = np.nanmean(repl2_smoothed, axis=0)  # shape (F2,)

    # 3) Perform DTW on these smoothed representative profiles
    path, dtw_cost = dtw_alignment(profile1, profile2)
    print(f"DTW alignment cost (smoothed): {dtw_cost:.2f}")

    # 4) Warp the entire replicate-2 matrix (already smoothed) to replicate 1's fraction axis
    rep2_warped = warp_rep2_to_rep1(repl2_smoothed, path, len(profile1))

    # 5) Average replicate 1 (smoothed) and replicate 2 (smoothed + warped)
    merged_arr = 0.5 * (repl1_smoothed + rep2_warped)  # shape (N, F)

    # Convert back to a pandas DataFrame, preserving the same index and columns as replicate1_df
    merged_preDTW_df = pd.DataFrame(
        data=merged_arr,
        index=replicate1_df.index,
        columns=replicate1_df.columns
    )

    return merged_preDTW_df


# ----------------------------------------------------------------------------
# Example usage:
if __name__ == "__main__":
    # Let's create dummy data with an index of protein names:
    df_all = pd.read_csv('data/coelution/repl1_repl2_combined.tsv', sep = '\t', index_col = 0)

    replicate_1 = [c for c in df_all.columns if 'repl1' in c]
    replicate_2 = [c for c in df_all.columns if 'repl2' in c]
    df_repl1 = df_all[replicate_1]
    df_repl2 = df_all[replicate_2]

    # Downsampling to the one_cluster size
    df_repl1 = df_repl1.loc[prot_list]
    df_repl2 = df_repl2.loc[prot_list]

    # Getting the proteins with all NaN values in either replicates
    all_null_prots_1 = df_repl1[df_repl1.isnull().all(axis=1)].index
    all_null_prots_2 = df_repl2[df_repl2.isnull().all(axis=1)].index
    all_null_both = all_null_prots_1.intersection(all_null_prots_2)
    
    replicate1_df = df_repl1.drop(all_null_both)
    replicate2_df = df_repl2.drop(all_null_both)

    # Align and merge with pre-DTW smoothing:
    merged_preDTW_df = align_and_merge_with_preDTW_smoothing(replicate1_df, replicate2_df, sigma=1.0)

    print("Final merged DataFrame:\n", merged_preDTW_df)
    print("\nMerged DataFrame shape:", merged_preDTW_df.shape)

    # Imputed Aligned data using NMF
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    df_imputed = nmf_impute_pytorchmerged_preDTW_df, rank=128, max_epochs=50000, lr=1e-2, device=device)

    df_imputed.to_csv("imputed_cleaned_coelution.csv", index = True)

    print("\n--- Original DF with NaNs ---")
    print(df_nx72.head())

    print("\n--- Imputed DF ---")
    print(df_imputed.head())