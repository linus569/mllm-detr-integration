import torch
import torch.nn.functional as F

from transformers.loss.loss_for_object_detection import ForObjectDetectionLoss

def detr_loss(
    logits, labels, device, pred_boxes, config, outputs_class=None, outputs_coord=None, **kwargs
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
    loss, loss_dict, auxiliary_outputs = ForObjectDetectionLoss(logits=logits, labels=labels, device=device, pred_boxes=pred_boxes, config=config, outputs_class=outputs_class, outputs_coord=outputs_coord, **kwargs)
    return loss # loss_dict

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
