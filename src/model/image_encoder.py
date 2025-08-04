import logging
from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
from transformers import AutoModel, DetrForObjectDetection

from utils.config import ExperimentConfig

log = logging.getLogger(__name__)


class BaseImageEncoder(ABC, torch.nn.Module):
    """
    Base class for image encoders.
    """

    def __init__(self, model_name: str, config: ExperimentConfig):
        super().__init__()

        self.model_name = model_name
        self.experiment_config = config
        self.encoder = None

    @abstractmethod
    def encode(
        self, pixel_values: torch.FloatTensor, image_sizes: List[Tuple[int, int]] = None
    ) -> torch.FloatTensor:
        """
        Extract features from images.
        """
        pass

    @abstractmethod
    def get_output_dim(self):
        """Return output dimension of features"""
        pass

    def load_model(self):
        """
        Load the model from the specified model name.
        """
        # log.info(f"Loading model {self.model_name}")
        # self.model = AutoModel.from_pretrained(self.model_name)
        pass


class SiglipImageEncoder(BaseImageEncoder):
    """
    Image encoder using SIGLIP.
    """

    def __init__(
        self,
        model_name: str,
        config: ExperimentConfig,
        model: LlavaQwenForCausalLM,
        **kwargs,
    ):
        log.info("Using SIGLIP image encoder")
        super().__init__(model_name, config)

        self.encoder = model.get_vision_tower()

        # Freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

    def encode(self, pixel_values, image_sizes=None):
        if pixel_values.ndim != 4:
            raise ValueError(
                f"Expected pixel_values to have 4 dimensions, but got {pixel_values.ndim}"
            )

        image_features = self.encoder(pixel_values)
        return image_features

    def get_output_dim(self):
        return self.encoder.config.hidden_size


class DinoV2ImageEncoder(BaseImageEncoder):
    """
    Image encoder using DINOv2.
    """

    def __init__(self, model_name: str, config: ExperimentConfig, **kwargs):
        log.info("Using DINOv2 image encoder")
        super().__init__(model_name, config)

        self.encoder = AutoModel.from_pretrained(self.model_name)
        # Freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

    def encode(self, pixel_values, image_sizes):
        if pixel_values.ndim != 4:
            raise ValueError(
                f"Expected pixel_values to have 4 dimensions, but got {pixel_values.ndim}"
            )

        outputs = self.encoder(pixel_values)

        # Extract the [CLS] token representation (first token) or
        # use the last hidden state
        if self.experiment_config.image_encoder.use_pooler_output:
            image_features = outputs.pooler_output.unsqueeze(1)
        else:
            image_features = outputs.last_hidden_state
        return image_features

    def get_output_dim(self):
        return self.encoder.config.hidden_size


class Resnet50ImageEncoder(BaseImageEncoder):
    """
    Image encoder using ResNet50 from DETR.
    """

    def __init__(self, model_name: str, config: ExperimentConfig, **kwargs):
        log.info("Using ResNet50 image encoder")
        super().__init__(model_name, config)

        # Load DETR model and extract backbone
        model = DetrForObjectDetection.from_pretrained(
            self.model_name, revision="no_timm"
        )
        self.encoder = model.model.backbone
        self.model_detr = model.model

        # Freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

    def encode(self, pixel_values, image_sizes):
        # use image_sizes to create pixel_mask
        pixel_mask = torch.zeros(
            pixel_values.shape[0],
            pixel_values.shape[2],
            pixel_values.shape[3],
            dtype=torch.bool,
        )
        # TODO: is this needed?
        for i, size in enumerate(image_sizes):
            pixel_mask[i, : size[0], : size[1]] = 1

        pixel_mask = pixel_mask.to(pixel_values.device)
        features, object_queries_list = self.encoder(pixel_values, pixel_mask)

        # get final feature map and downsampled mask
        feature_map, mask = features[-1]
        if mask is None:
            raise ValueError("Backbone does not return downsampled pixel mask")

        if "encoder" in self.experiment_config.image_encoder.name:
            # Second, apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
            projected_feature_map = self.model_detr.input_projection(feature_map)

            # Third, flatten the feature map + position embeddings of shape NxCxHxW to NxCxHW, and permute it to NxHWxC
            # In other words, turn their shape into (batch_size, sequence_length, hidden_size)
            flattened_features = projected_feature_map.flatten(2).permute(0, 2, 1)
            object_queries = object_queries_list[-1].flatten(2).permute(0, 2, 1)

            flattened_mask = mask.flatten(1)

            # Fourth, sent flattened_features + flattened_mask + position embeddings through encoder
            # flattened_features is a Tensor of shape (batch_size, heigth*width, hidden_size)
            # flattened_mask is a Tensor of shape (batch_size, heigth*width)
            encoder_outputs = self.model_detr.encoder(
                inputs_embeds=flattened_features,
                attention_mask=flattened_mask,
                object_queries=object_queries,
                output_attentions=None,
                output_hidden_states=True,
                return_dict=False,
            )

            # encoder_outputs = (last_hidden_state, hidden_states, attentions)
            image_features = encoder_outputs[0]
        else:
            image_features = feature_map.flatten(2).permute(0, 2, 1)

        return image_features

    def get_output_dim(self):
        if "encoder" in self.experiment_config.image_encoder.name:
            # The output dimension is the same as the input dimension of the encoder
            return self.model_detr.config.d_model
        return self.encoder.conv_encoder.intermediate_channel_sizes[-1]
