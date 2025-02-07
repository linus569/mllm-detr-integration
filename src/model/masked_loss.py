import torch.nn.functional as F


# currently same as in transformers.loss.loss_utils
def masked_cross_entropy(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: int = None,
    ignore_index: int = -100,
):
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
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = F.cross_entropy(
        shift_logits, shift_labels, ignore_index=ignore_index, reduction=reduction
    )
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss
