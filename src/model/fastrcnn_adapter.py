import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from transformers.modeling_outputs import CausalLMOutputWithPast


class FastRCNNAdapter(torch.nn.Module):
    """Adapter for Faster R-CNN model that can be used to replace VLM in training pipeline."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fastrcnn = fasterrcnn_resnet50_fpn() #weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)  #
        print(self.fastrcnn)
        self.device = torch.device(config.device)
        self.fastrcnn.to(self.device)

    def _convert_labels_to_targets(self, labels):
        """Convert labels to Fast RCNN targets."""
        pass

    def _format_predictions(self, predictions):
        """Format Fast RCNN predictions."""
        pass

    def forward(self, input_ids=None, images=None, attention_mask=None, labels=None):
        self.train()
        images = images.to(self.device)

        if labels is not None:
            #targets = self._convert_labels_to_targets(labels)
            targets = labels
            loss_dict = self.fastrcnn(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            return CausalLMOutputWithPast(loss=loss, logits=None)
        outputs = self.fastrcnn(images)
        return CausalLMOutputWithPast(logits=torch.zeros(1))

    @torch.no_grad()
    def generate(self, input_ids=None, attention_mask=None, image=None, **kwargs):
        self.eval()
        if image is None:
            raise ValueError("Image is required for generation.")
        predictions = self.fastrcnn(image)
        # formatted_preds = self._format_predictions(predictions)
        return predictions
