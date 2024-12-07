import scipy.ndimage as ndi
import torch
import numpy as np
import torch.nn.functional as F

# Probably everything should be in the realm of PyTorch tensors no? To not interrupt the flow of gradients


def basic_threshold(predicted_heatmaps, threshold=0.1):
    """
    Apply thresholding and local maxima detection to a batch of predicted heatmaps.
    This version keeps the operations in PyTorch to retain gradients.

    Args:
        predicted_heatmaps (torch.Tensor): Tensor of shape (batch_size x 1 x height x width).
        threshold (float): Threshold value to filter out low-confidence predictions.

    Returns:
        list of torch.Tensor: A list of tensors containing the coordinates of detected points for each heatmap.
    """
    batch_size, _, height, width = predicted_heatmaps.shape

    coordinates_list = []

    for i in range(batch_size):
        # Step 1: Thresholding
        binary_heatmap = (predicted_heatmaps[i, 0] > threshold).float()

        # Step 2: Find Local Maxima (using max pooling to emulate maximum filter)
        max_pooled_heatmap = F.max_pool2d(predicted_heatmaps[i, 0].unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1,
                                          padding=1)
        local_maxima = (predicted_heatmaps[i, 0] == max_pooled_heatmap.squeeze(0).squeeze(0)).float()

        # Step 3: Extract Coordinates
        final_points = local_maxima * binary_heatmap

        # Use torch.nonzero to find the coordinates of non-zero elements
        coordinates = torch.nonzero(final_points, as_tuple=False)

        coordinates_list.append(coordinates)

    return coordinates_list


def soft_thresholding(heatmap):
    assert heatmap.dim() == 2, "Heatmap must be a 2D tensor"

    # Get the dimensions of the heatmap
    height, width = heatmap.shape

    # Create coordinate grids
    x = torch.arange(width, dtype=torch.float32, device=heatmap.device)
    y = torch.arange(height, dtype=torch.float32, device=heatmap.device)

    # Expand coordinates to match the heatmap shape
    x = x.unsqueeze(0).expand(height, width)
    y = y.unsqueeze(1).expand(height, width)

    # Flatten the heatmap and coordinates for softmax
    heatmap_flat = heatmap.view(-1)
    x_flat = x.view(-1)
    y_flat = y.view(-1)

    # Apply softmax to the heatmap
    softmax = torch.nn.functional.softmax(beta * heatmap_flat, dim=0)

    # Compute the expected x and y coordinates
    expected_x = torch.sum(x_flat * softmax)
    expected_y = torch.sum(y_flat * softmax)

    return torch.tensor([expected_x, expected_y], device=heatmap.device)
