import torch

def compute_recall_at_k(h_hat, h, ks=[1, 5, 10]):
    N, D = h.shape
    # Ensure h_hat and h are PyTorch tensors
    if not isinstance(h_hat, torch.Tensor):
        h_hat = torch.tensor(h_hat)
    if not isinstance(h, torch.Tensor):
        h = torch.tensor(h)
    
    # Compute pairwise L2 distances between predicted and ground truth features
    # distances: N x N tensor
    distances = torch.cdist(h_hat, h, p=2)  # Shape: N x N

    # Get the indices that sort the distances for each predicted feature
    sorted_indices = torch.argsort(distances, dim=1)  # Shape: N x N

    recalls = {}
    for k in ks:
        # Get the top-k indices for each predicted feature
        top_k_indices = sorted_indices[:, :k]  # Shape: N x k

        # Ground truth indices (from 0 to N-1)
        ground_truth_indices = torch.arange(N, device=h_hat.device).unsqueeze(1)  # Shape: N x 1

        # Check if the ground truth index is among the top-k predictions
        correct = (top_k_indices == ground_truth_indices).any(dim=1).float()  # Shape: N

        # Calculate recall@k
        recalls[k] = correct.mean().item()
    return recalls