import torch
import numpy as np
from torch_scatter import scatter_add


def weighting_DSC(y_pred, y_true, class_weights, smooth=1.0):
    smooth = 1.0
    mdsc = 0.0
    n_classes = y_pred.shape[-1]

    # convert probability to one-hot code
    max_idx = torch.argmax(y_pred, dim=-1, keepdim=True)
    one_hot = torch.zeros_like(y_pred)
    one_hot.scatter_(-1, max_idx, 1)

    for c in range(0, n_classes):
        pred_flat = one_hot[:, c].reshape(-1)
        true_flat = y_true[:, c].reshape(-1)
        intersection = (pred_flat * true_flat).sum()
        w = class_weights[c] / class_weights.sum()
        mdsc += w * (
            (2.0 * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)
        )

    return mdsc


def weighting_SEN(y_pred, y_true, class_weights, smooth=1.0):
    smooth = 1.0
    msen = 0.0
    n_classes = y_pred.shape[-1]

    # convert probability to one-hot code
    max_idx = torch.argmax(y_pred, dim=-1, keepdim=True)
    one_hot = torch.zeros_like(y_pred)
    one_hot.scatter_(-1, max_idx, 1)

    for c in range(0, n_classes):
        pred_flat = one_hot[:, c].reshape(-1)
        true_flat = y_true[:, c].reshape(-1)
        intersection = (pred_flat * true_flat).sum()
        w = class_weights[c] / class_weights.sum()
        msen += w * ((intersection + smooth) / (true_flat.sum() + smooth))

    return msen


def weighting_PPV(y_pred, y_true, class_weights, smooth=1.0):
    smooth = 1.0
    mppv = 0.0
    n_classes = y_pred.shape[-1]

    # convert probability to one-hot code
    max_idx = torch.argmax(y_pred, dim=-1, keepdim=True)
    one_hot = torch.zeros_like(y_pred)
    one_hot.scatter_(-1, max_idx, 1)

    for c in range(0, n_classes):
        pred_flat = one_hot[:, c].reshape(-1)
        true_flat = y_true[:, c].reshape(-1)
        intersection = (pred_flat * true_flat).sum()
        w = class_weights[c] / class_weights.sum()
        mppv += w * ((intersection + smooth) / (pred_flat.sum() + smooth))

    return mppv


def Generalized_Dice_Loss(y_pred, y_true, class_weights, smooth=1.0):
    smooth = 1.0
    loss = 0.0
    n_classes = y_pred.shape[-1]
    for c in range(0, n_classes):
        pred_flat = y_pred[:, c].reshape(-1)
        true_flat = y_true[:, c].reshape(-1)
        intersection = (pred_flat * true_flat).sum()

        # with weight
        w = class_weights[c] / class_weights.sum()
        loss += w * (
            1
            - (
                (2.0 * intersection + smooth)
                / (pred_flat.sum() + true_flat.sum() + smooth)
            )
        )

    return loss


def DSC(y_pred, y_true, ignore_background=True, smooth=1.0):
    """
    inputs:
        y_pred [npts, n_classes] one-hot code
        y_true [npts, n_classes] one-hot code
    """
    smooth = 1.0
    n_classes = y_pred.shape[-1]
    dsc = []
    if ignore_background:
        for c in range(1, n_classes):  # pass 0 because 0 is background
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            dsc.append(
                (
                    (2.0 * intersection + smooth)
                    / (pred_flat.sum() + true_flat.sum() + smooth)
                )
            )

        dsc = np.asarray(dsc)
    else:
        for c in range(0, n_classes):
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            dsc.append(
                (
                    (2.0 * intersection + smooth)
                    / (pred_flat.sum() + true_flat.sum() + smooth)
                )
            )

        dsc = np.asarray(dsc)

    return dsc


def SEN(y_pred, y_true, ignore_background=True, smooth=1.0):
    """
    inputs:
        y_pred [npts, n_classes] one-hot code
        y_true [npts, n_classes] one-hot code
    """
    smooth = 1.0
    n_classes = y_pred.shape[-1]
    sen = []
    if ignore_background:
        for c in range(1, n_classes):  # pass 0 because 0 is background
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            sen.append(((intersection + smooth) / (true_flat.sum() + smooth)))

        sen = np.asarray(sen)
    else:
        for c in range(0, n_classes):
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            sen.append(((intersection + smooth) / (true_flat.sum() + smooth)))

        sen = np.asarray(sen)

    return sen


def PPV(y_pred, y_true, ignore_background=True, smooth=1.0):
    """
    inputs:
        y_pred [npts, n_classes] one-hot code
        y_true [npts, n_classes] one-hot code
    """
    smooth = 1.0
    n_classes = y_pred.shape[-1]
    ppv = []
    if ignore_background:
        for c in range(1, n_classes):  # pass 0 because 0 is background
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            ppv.append(((intersection + smooth) / (pred_flat.sum() + smooth)))

        ppv = np.asarray(ppv)
    else:
        for c in range(0, n_classes):
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            ppv.append(((intersection + smooth) / (pred_flat.sum() + smooth)))

        ppv = np.asarray(ppv)

    return ppv


def weighted_mse_loss(inputs, targets, weights="balanced", roi_mask=None):
    """
    Computes the weighted mean squared error (MSE) loss between the predicted `inputs` and the target `targets`.
    The weights can be specified as a tensor of the same shape as `targets`, or as the string "balanced", in which
    case the weights are set to 1 for all elements, except for those in the region of interest (ROI), which are
    multiplied by a factor that balances the number of elements inside and outside the ROI.

    Args:
        inputs (torch.Tensor): The predicted values, with shape (batch_size, num_channels, height, width).
        targets (torch.Tensor): The target values, with shape (batch_size, num_channels, height, width).
        weights (Union[str, torch.Tensor]): The weights to apply to each element of the loss. If "balanced", the
            weights are set to 1 for all elements, except for those in the ROI, which are multiplied by a factor
            that balances the number of elements inside and outside the ROI. If a tensor is provided, it must have
            the same shape as `targets`.
        roi_mask (Optional[torch.Tensor]): A binary mask indicating the region of interest (ROI), with the shape as `targets`.
            If not provided, all elements are considered part of the ROI.

    Returns:
        The weighted mean squared error (MSE) loss between the predicted `inputs` and the target `targets`.
    """
    if weights == "balanced":
        weights = torch.ones_like(targets)
        roi_weight = roi_mask.numel() / roi_mask.sum()
        weights[roi_mask] *= roi_weight
    elif isinstance(weights, torch.Tensor):
        if weights.shape != targets.shape:
            raise ValueError(
                "The shape of the weights tensor must match the shape of the targets tensor."
            )
    else:
        raise ValueError("The weights argument must be either 'balanced' or a tensor.")

    return torch.mean(weights * (inputs - targets) ** 2)


def get_offset_gt(offset_pred, graph, num_classes):
    position = graph.x[:, :3]
    teeth_centroids = graph.teeth_centroids
    cell_label = graph.cell_label.detach().cpu()
    batch = graph.batch.detach().cpu()
    centroids_idx = cell_label.flatten() + batch * num_classes
    centroids = teeth_centroids[centroids_idx]
    offset_gt = centroids - position
    gingiva_idx = offset_gt.sum(1) > 1e8  # gingiva centroid is torch.ones([3]) * 1e8
    offset_gt[gingiva_idx] = 0  # expect zero offset for gingiva cells
    return offset_gt


def offset_direction_loss(offset_pred, offset_gt):
    cos_sim = torch.nn.CosineSimilarity(dim=1)(offset_pred, offset_gt)
    return -cos_sim.mean()


def distance_loss(pred, gt):
    loss = (pred - gt).pow(2).sum(1).sqrt().mean(0)
    return loss


def centroid_loss(prob, pos, centroid, batch, low=0.3, up=0.6):
    # TODO: fix bug
    prob[prob > up] = 1
    prob[prob < low] = 0
    batch_sum = scatter_add(prob, batch, dim=0, dim_size=batch.max().item() + 1)
    prob_sum = batch_sum[batch]
    prob = prob / prob_sum
    centroid_pred = torch.matmul(prob.T, pos)
    valid_mask = centroid.mean(dim=1) < 1e3
    return distance_loss(centroid_pred[valid_mask], centroid[valid_mask])


def boundary_loss(prob, graph):
    cell_region_label = graph.cell_region_label.detach().cpu().flatten()
    boundary_cell_idx = torch.where(cell_region_label >= 2)[0]
    boundary_prob = prob[boundary_cell_idx]
    boundary_gt = graph.cell_label[boundary_cell_idx]
    loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
    return loss_func(boundary_prob, boundary_gt.flatten())
