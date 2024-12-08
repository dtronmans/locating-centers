import torch
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


if __name__ == "__main__":
    pass
