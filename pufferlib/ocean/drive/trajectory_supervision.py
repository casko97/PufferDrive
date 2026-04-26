import torch


def masked_trajectory_loss(
    predictions,
    targets,
    valid_threshold=0.5,
    horizon_decay=0.97,
    position_weight=1.0,
    heading_weight=0.5,
    speed_weight=0.25,
    valid_weight=0.25,
):
    predictions = predictions.view(predictions.shape[0], -1, 5)
    targets = targets.view(targets.shape[0], -1, 5)

    valid_target = targets[..., 4]
    valid_mask = (valid_target > valid_threshold).float()
    horizon_weights = horizon_decay ** torch.arange(targets.shape[1], device=targets.device, dtype=targets.dtype)
    horizon_weights = horizon_weights.view(1, -1)

    regression_weight = valid_mask * horizon_weights
    regression_norm = regression_weight.sum().clamp_min(1.0)

    position_loss = ((predictions[..., :2] - targets[..., :2]) ** 2).sum(dim=-1)
    heading_loss = (predictions[..., 2] - targets[..., 2]) ** 2
    speed_loss = (predictions[..., 3] - targets[..., 3]) ** 2
    valid_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        predictions[..., 4], valid_target, reduction="none"
    )

    total = 0.0
    total = total + position_weight * (position_loss * regression_weight).sum() / regression_norm
    total = total + heading_weight * (heading_loss * regression_weight).sum() / regression_norm
    total = total + speed_weight * (speed_loss * regression_weight).sum() / regression_norm
    total = total + valid_weight * (valid_loss * horizon_weights).mean()
    return total


def trajectory_metrics(predictions, targets, valid_threshold=0.5):
    predictions = predictions.view(predictions.shape[0], -1, 5)
    targets = targets.view(targets.shape[0], -1, 5)
    valid_mask = targets[..., 4] > valid_threshold

    if not torch.any(valid_mask):
        zero = predictions.new_tensor(0.0)
        return {
            "ade": zero,
            "fde": zero,
            "heading_error": zero,
            "speed_error": zero,
            "valid_accuracy": zero,
        }

    position_error = torch.linalg.norm(predictions[..., :2] - targets[..., :2], dim=-1)
    heading_error = (predictions[..., 2] - targets[..., 2]).abs()
    speed_error = (predictions[..., 3] - targets[..., 3]).abs()
    valid_prediction = torch.sigmoid(predictions[..., 4]) > valid_threshold

    ade = position_error[valid_mask].mean()
    heading = heading_error[valid_mask].mean()
    speed = speed_error[valid_mask].mean()
    valid_accuracy = (valid_prediction == valid_mask).float().mean()

    last_valid_idx = valid_mask.int().sum(dim=1).clamp_min(1) - 1
    batch_idx = torch.arange(predictions.shape[0], device=predictions.device)
    fde = position_error[batch_idx, last_valid_idx].mean()

    return {
        "ade": ade,
        "fde": fde,
        "heading_error": heading,
        "speed_error": speed,
        "valid_accuracy": valid_accuracy,
    }
