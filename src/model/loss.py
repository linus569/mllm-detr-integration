import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.loss.loss_for_object_detection import (
    ForObjectDetectionLoss,
    ImageLoss,
    _set_aux_loss,
)


class SizeBasedMatcher(nn.Module):
    """
    This class matches predictions to targets based on bounding box size.
    Instead of using Hungarian matching, it sorts both predicted and target boxes
    by their area and matches them in order.
    """

    def __init__(self, descending: bool = True):
        super().__init__()

        self.descending = descending

    def _calculate_box_area(self, boxes):
        """
        Calculate the area of bounding boxes.
        Args:
            boxes: Tensor of shape (N, 4) where N is the number of boxes
        Returns:
            Tensor of shape (N,) containing the area of each box
        """
        x1, y1, x2, y2 = boxes.unbind(1)
        return (x2 - x1) * (y2 - y1)

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs (dict): Dictionary containing:
                * "logits": Tensor of dim [batch_size, num_queries, num_classes]
                * "pred_boxes": Tensor of dim [batch_size, num_queries, 4] (cx, cy, w, h)
            targets (List[dict]): List of dictionaries containing:
                * "class_labels": Tensor of dim [num_target_boxes]
                * "boxes": Tensor of dim [num_target_boxes, 4] (cx, cy, w, h)

        Returns:
            List[Tuple]: List of tuples (index_i, index_j) where:
                * index_i are indices of selected predictions
                * index_j are indices of corresponding target objects
        """
        batch_size = len(targets)
        results = []

        for b in range(batch_size):
            pred_boxes = outputs["pred_boxes"][b]  # [num_queries, 4]
            target_boxes = targets[b]["boxes"]  # [num_targets, 4]

            if len(target_boxes) == 0:
                # No targets, return empty matching
                results.append(
                    (
                        torch.tensor([], dtype=torch.int64),
                        torch.tensor([], dtype=torch.int64),
                    )
                )
                continue

            # Calculate areas
            pred_areas = self._calculate_box_area(pred_boxes)  # [num_queries]
            target_areas = self._calculate_box_area(target_boxes)  # [num_targets]

            # Sort boxes by area
            pred_indices = torch.argsort(pred_areas, descending=self.descending)
            target_indices = torch.argsort(target_areas, descending=self.descending)

            # Match first N boxes, where N is min(num_queries, num_targets)
            num_to_match = min(len(pred_indices), len(target_indices))
            matched_pred_indices = pred_indices[:num_to_match]
            matched_target_indices = target_indices[:num_to_match]

            results.append((matched_pred_indices, matched_target_indices))

        return results


def detr_loss(
    logits,
    labels,
    device,
    pred_boxes,
    config,
    outputs_class=None,
    outputs_coord=None,
    sized_based_matching=False,
    **kwargs,
):
    """
    Compute the loss for DETR model.

    Args:
        logits: Model logits
        labels: Target labels
        device: Device to use
        pred_boxes: Predicted bounding boxes
        config: Configuration object
        outputs_class: Class outputs (optional)
        outputs_coord: Coordinate outputs (optional)
        **kwargs: Additional arguments

    Returns:
        The computed loss
    """
    if sized_based_matching:
        """copied from transformers.loss.loss_for_object_detection.py and adapted with SizeBasedMatcher"""
        # First: create the matcher
        matcher = SizeBasedMatcher()

        # Second: create the criterion
        losses = ["labels", "boxes", "cardinality"]
        criterion = ImageLoss(
            matcher=matcher,
            num_classes=config.num_labels,
            eos_coef=config.eos_coefficient,
            losses=losses,
        )
        criterion.to(device)
        # Third: compute the losses, based on outputs and labels
        outputs_loss = {}
        auxiliary_outputs = None
        outputs_loss["logits"] = logits
        outputs_loss["pred_boxes"] = pred_boxes
        if config.auxiliary_loss:
            auxiliary_outputs = _set_aux_loss(outputs_class, outputs_coord)
            outputs_loss["auxiliary_outputs"] = auxiliary_outputs

        loss_dict = criterion(outputs_loss, labels)
        # Fourth: compute total loss, as a weighted sum of the various losses
        weight_dict = {"loss_ce": 1, "loss_bbox": config.bbox_loss_coefficient}
        weight_dict["loss_giou"] = config.giou_loss_coefficient
        if config.auxiliary_loss:
            aux_weight_dict = {}
            for i in range(config.decoder_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        loss = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )
        return loss  # , loss_dict, auxiliary_outputs
    else:
        loss, loss_dict, auxiliary_outputs = ForObjectDetectionLoss(
            logits=logits,
            labels=labels,
            device=device,
            pred_boxes=pred_boxes,
            config=config,
            outputs_class=outputs_class,
            outputs_coord=outputs_coord,
            **kwargs,
        )
        return loss  # loss_dict


# currently same as in transformers.loss.loss_utils
def masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    num_items_in_batch: int = None,
    ignore_index: int = -100,
):
    """
    Compute cross-entropy loss while ignoring padding tokens.

    Args:
        logits: Logits from the model
        labels: Target labels
        vocab_size: Size of the vocabulary
        num_items_in_batch: Number of items in the batch
        ignore_index: Index to ignore

    Returns:
        The computed loss
    """

    # upcast to float to avoid precision issues
    logits = logits.float()
    labels = labels.to(logits.device)

    # shift so that next token is predicted
    shift_logits = logits[
        ..., :-1, :
    ].contiguous()  # removes last token as it is not predicted
    shift_labels = labels[..., 1:].contiguous()  # moves labels one step forward

    # flatten the tokens
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)

    shift_labels = shift_labels.to(shift_logits.device)

    # compute loss
    # TODO: reduction correct?
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = F.cross_entropy(
        shift_logits, shift_labels, ignore_index=ignore_index, reduction=reduction
    )
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss
