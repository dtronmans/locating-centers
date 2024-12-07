import torch
import torch.nn as nn


def mse_loss(predicted, ground_truth):
    criterion = nn.MSELoss()

    loss = criterion(predicted, ground_truth)

    return loss


def chamfer_distance(predicted_points, target_points):
    pred_to_target_dist = torch.cdist(predicted_points, target_points, p=2)
    target_to_pred_dist = torch.cdist(target_points, predicted_points, p=2)

    loss_pred_to_target = pred_to_target_dist.min(dim=1)[0].mean()
    loss_target_to_pred = target_to_pred_dist.min(dim=1)[0].mean()

    return loss_pred_to_target + loss_target_to_pred