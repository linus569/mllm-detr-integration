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


class SigLipFeatureExtractor(nn.Module):
    def __init__(
        self, fpn_imitation, out_channels, trainable_backbone=False
    ):  # TODO: can be of any size!?
        super().__init__()
        self.fpn_imitation = fpn_imitation

        self.image_encoder = LlavaQwenForCausalLM.from_pretrained(
            "lmms-lab/llava-onevision-qwen2-0.5b-si"
        ).get_vision_tower()
        # self.image_encoder.eval()

        # Ensure the encoder is frozen or trainable based on your needs
        for param in self.image_encoder.parameters():
            param.requires_grad = trainable_backbone  # TODO: Set True if fine-tuning

        hidden_dim = self.image_encoder.config.hidden_size
        self.out_channels = out_channels
        self.input_size = 384

        # Projection layer to reshape embeddings into spatial feature maps
        # self.feature_projection = nn.Conv2d(
        #     in_channels=hidden_dim, out_channels=output_dim, kernel_size=1
        # )
        self.feature_projection = nn.Sequential(
            nn.Conv2d(hidden_dim, self.out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )

        # Get vision transformer layers for access to intermediate features
        # print(self.image_encoder)
        self.transformer_layers = (
            self.image_encoder.vision_tower.vision_model.encoder.layers
        )
        self.num_transformer_layers = len(self.transformer_layers)

        # Extract features from different layers of the transformer
        # We'll use the last 4 layers for the FPN
        self.layer_indices = [
            self.num_transformer_layers - 4,
            self.num_transformer_layers - 3,
            self.num_transformer_layers - 2,
            self.num_transformer_layers - 1,
        ]

        # Project features from different layers to a common dimension
        self.projections = nn.ModuleList(
            [nn.Conv2d(hidden_dim, self.out_channels, kernel_size=1) for _ in range(4)]
        )

        # FPN lateral connections (from top-down)
        self.fpn_laterals = nn.ModuleList(
            [
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1)
                for _ in range(3)
            ]
        )

        # FPN predictors
        self.fpn_outputs = nn.ModuleList(
            [
                nn.Conv2d(
                    self.out_channels, self.out_channels, kernel_size=3, padding=1
                )
                for _ in range(4)
            ]
        )

    def _extract_layer_features(self, x, layer_idx):
        # Extract features from a specific transformer layer
        hidden_states = x

        # Process embeddings
        if layer_idx == -1:
            return hidden_states

        # Create attention mask in the correct format for SigLip
        # SigLip expects a 4D attention mask: (batch_size, num_heads, seq_len, seq_len)
        batch_size, seq_len = hidden_states.shape[:2]
        # Create a square mask where all positions can attend to all others
        attention_mask = torch.ones(
            (batch_size, 1, seq_len, seq_len), device=hidden_states.device
        )

        # Process through transformer layers up to the specified index
        for i, layer_module in enumerate(self.transformer_layers):
            hidden_states = layer_module(hidden_states, attention_mask=attention_mask)[
                0
            ]
            if i == layer_idx:
                return hidden_states

        return hidden_states

    def _reshape_features(self, features):
        # Remove class token and reshape to 2D feature map
        b, n, c = features.shape
        h = w = int(n**0.5)  # Assumes square input
        return features.permute(0, 2, 1).reshape(b, c, h, w)

    def forward(self, x):
        """
        #     Extract features from LLaVA and reshape for Faster R-CNN.
        #     :param images: Tensor of shape (batch, C, H, W)
        #     :return: Dictionary of feature maps suitable for Faster R-CNN
        """
        if self.fpn_imitation:
            # Get batch size and channels
            batch_size = x.shape[0]

            # Resize if needed
            if x.shape[-2:] != (384, self.input_size):
                x = torch.nn.functional.interpolate(
                    x,
                    size=(self.input_size, self.input_size),
                    mode="bilinear",
                    align_corners=False,
                )

            # SigLIP expects normalized inputs
            if x.min() >= 0 and x.max() <= 1:
                x = (x - 0.5) * 2.0

            # Get embeddings
            embeddings = self.image_encoder.vision_tower.vision_model.embeddings(x)

            # Extract features from different transformer layers
            multi_scale_features = []
            for i, layer_idx in enumerate(self.layer_indices):
                # Get features from this layer
                layer_features = self._extract_layer_features(embeddings, layer_idx)
                # Reshape to 2D feature map
                spatial_features = self._reshape_features(layer_features)
                # print(layer_features.shape)
                # print(spatial_features.shape)
                # Project to common dimension
                projected_features = self.projections[i](spatial_features)
                # Add to list
                multi_scale_features.append(projected_features)

            # Apply FPN (top-down pathway with lateral connections)
            fpn_features = {}
            last_feature = multi_scale_features[-1]
            fpn_features["3"] = self.fpn_outputs[3](last_feature)

            # Top-down pathway
            for i in range(2, -1, -1):
                higher_resolution_feature = multi_scale_features[i]
                lateral_connection = self.fpn_laterals[i](higher_resolution_feature)

                # Upsample and add
                upsampled = torch.nn.functional.interpolate(
                    last_feature, size=lateral_connection.shape[-2:], mode="nearest"
                )
                last_feature = lateral_connection + upsampled
                fpn_features[str(i)] = self.fpn_outputs[i](last_feature)

            return fpn_features
        else:
            # Extract features using SigLip image_encoder
            features = self.image_encoder(x)  # Shape: (batch, seq_len, hidden_dim)

            # LLaVA outputs a sequence (batch, seq_len, dim), reshape to (batch, channels, height, width)
            batch_size, seq_len, dim = features.shape
            spatial_size = int(seq_len**0.5)  # Assuming square patch embedding
            features = features.permute(0, 2, 1).reshape(
                batch_size, dim, spatial_size, spatial_size
            )

            # Reduce dimensionality and adapt for Faster R-CNN
            features = self.feature_projection(features)

            # Return as OrderedDict with named features as expected by FasterRCNN
            return OrderedDict([("0", features)])

    # def forward(self, images):
    #


def create_siglip_fasterrcnn(num_classes, trainable_backbone=False):
    fpn_imitation = True
    out_channels = 512
    backbone = SigLipFeatureExtractor(
        fpn_imitation=fpn_imitation, out_channels=out_channels
    )

    if fpn_imitation:

        anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,)), aspect_ratios=((0.5, 1.0, 2.0),) * 4
        )

        # Create ROI pooler
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=7,
            sampling_ratio=2,  # Only one feature level
        )
    else:
        # For single feature map: all anchor sizes on one level
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256),),  # All sizes in single tuple
            aspect_ratios=((0.5, 1.0, 2.0),),  # One tuple for single feature map
        )

        roi_pooler = MultiScaleRoIAlign(
            featmap_names=["0"], output_size=7, sampling_ratio=2  # Only one feature map
        )

    model = FasterRCNN(
        min_size=384,
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
    )

    return model


class FastRCNNAdapter(torch.nn.Module):
    """Adapter for Faster R-CNN model that can be used to replace VLM in training pipeline."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        if "resnet" in config.model_name:
            self.fastrcnn = fasterrcnn_resnet50_fpn(
                weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            )
            # weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            # print(self.fastrcnn_resnet)
        elif "siglip" in config.model_name:
            self.fastrcnn = create_siglip_fasterrcnn(num_classes=91)
            # print(self.fastrcnn)
        else:
            raise ValueError(
                f"Model {config.model_name} not supported. Please use resnet or siglip."
            )

        self.device = torch.device(config.device)
        self.fastrcnn.to(self.device)

    def forward(self, input_ids=None, images=None, attention_mask=None, labels=None):
        self.fastrcnn.train()
        if isinstance(images, list):
            images = [img.to(self.device) for img in images]
        elif isinstance(images, torch.Tensor):
            images = [images.to(self.device)]
        else:
            raise ValueError("Images should be a list of tensors or single tensor.")

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

        # image is list of tensors
        if isinstance(image, list):
            image = [img.to(self.device) for img in image]
        elif isinstance(image, torch.Tensor):
            image = [image.to(self.device)]
        else:
            raise ValueError("Image should be a list of tensors or single tensor.")

        predictions = self.fastrcnn(image)
        return predictions
