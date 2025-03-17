from collections import OrderedDict

import torch
import torch.nn as nn
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
from torchvision.models.detection import (
    FasterRCNN,
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from transformers.modeling_outputs import CausalLMOutputWithPast


class LLaVAFeatureExtractor(nn.Module):
    def __init__(self, llava_encoder, output_dim=256):
        super().__init__()
        self.llava_encoder = llava_encoder

        # Ensure the encoder is frozen or trainable based on your needs
        for param in self.llava_encoder.parameters():
            param.requires_grad = False  # Set True if fine-tuning

        with torch.no_grad():
            # Test forward pass to get the output dimension
            test_input = torch.rand(1, 3, 384, 384)
            test_output = self.llava_encoder(test_input)
            hidden_dim = test_output.shape[-1]
        # Projection layer to reshape embeddings into spatial feature maps
        self.projection = nn.Conv2d(
            in_channels=hidden_dim, out_channels=output_dim, kernel_size=1
        )
        self.out_channels = output_dim

    def forward(self, images):
        """
        Extract features from LLaVA and reshape for Faster R-CNN.
        :param images: Tensor of shape (batch, C, H, W)
        :return: Dictionary of feature maps suitable for Faster R-CNN
        """
        # transform image to 384x384
        images = nn.functional.interpolate(
            images, size=(384, 384), mode="bilinear", align_corners=False
        )

        with torch.no_grad():
            features = self.llava_encoder(images)  # Shape: (batch, seq_len, hidden_dim)

        # LLaVA outputs a sequence (batch, seq_len, dim), reshape to (batch, channels, height, width)
        batch_size, seq_len, dim = features.shape
        spatial_size = int(seq_len**0.5)  # Assuming square patch embedding
        features = features.permute(0, 2, 1).view(
            batch_size, dim, spatial_size, spatial_size
        )

        # Reduce dimensionality and adapt for Faster R-CNN
        features = self.projection(features)

        # Return as OrderedDict with named features as expected by FasterRCNN
        return OrderedDict([("0", features)])


image_encoder = LlavaQwenForCausalLM.from_pretrained(
    "lmms-lab/llava-onevision-qwen2-0.5b-si"
).get_vision_tower()
image_encoder.eval()


# Wrap it as a backbone for Faster R-CNN
class FasterRCNNWithLLaVA(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Custom backbone using LLaVA
        self.backbone = LLaVAFeatureExtractor(image_encoder)

        # Set up anchor generator params
        anchor_sizes = ((32, 64, 128, 256, 512),)
        aspect_ratios = ((0.5, 1.0, 2.0),)

        # Define Faster R-CNN model with custom backbone
        self.model = FasterRCNN(
            backbone=self.backbone,
            num_classes=num_classes,
            rpn_anchor_generator=AnchorGenerator(
                sizes=anchor_sizes, aspect_ratios=aspect_ratios
            ),
            box_roi_pool=MultiScaleRoIAlign(
                featmap_names=["0"],  # Should match keys returned by backbone
                output_size=7,
                sampling_ratio=2,
            ),
        )

    def forward(self, images, targets=None):
        return self.model(images, targets)


class FastRCNNAdapter(torch.nn.Module):
    """Adapter for Faster R-CNN model that can be used to replace VLM in training pipeline."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.fastrcnn = (
        #     fasterrcnn_resnet50_fpn()
        # )  # weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

        self.fastrcnn = FasterRCNNWithLLaVA(num_classes=91)
        print(self.fastrcnn)
        self.device = torch.device(config.device)
        self.fastrcnn.to(self.device)

    def forward(self, input_ids=None, images=None, attention_mask=None, labels=None):
        self.fastrcnn.train()
        images = images.to(self.device)

        if labels is not None:
            targets = labels
            loss_dict = self.fastrcnn(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            return CausalLMOutputWithPast(loss=loss, logits=None)
        return CausalLMOutputWithPast(logits=torch.zeros(1))

    @torch.no_grad()
    def generate(self, input_ids=None, attention_mask=None, image=None, **kwargs):
        self.fastrcnn.eval()
        if image is None:
            raise ValueError("Image is required for generation.")

        # Ensure image is a list if it's just a single tensor
        if isinstance(image, torch.Tensor) and image.dim() == 4:
            # If it's a batch of images, make sure it's on the right device
            image = image.to(self.device)
        elif isinstance(image, torch.Tensor) and image.dim() == 3:
            # If it's a single image (C,H,W), add batch dimension and move to device
            image = image.unsqueeze(0).to(self.device)

        predictions = self.fastrcnn(image)
        return predictions
