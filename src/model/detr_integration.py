import logging
from typing import List, Optional, Tuple

import torch
from transformers import (
    DabDetrConfig,
    DabDetrForObjectDetection,
    DetrConfig,
    DetrForObjectDetection,
)

from utils.config import ExperimentConfig

log = logging.getLogger(__name__)


class DETRIntegration(torch.nn.Module):
    """
    Module handling DETR object detection integration with Language Model.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        llm_hidden_size: int,
        image_encoder_hidden_size: int,
        batch_size: int,
        device: str,
    ):
        """
        Initialize the DETR components.

        Args:
            config: Experiment configuration.
            llm_hidden_size: Hidden size of the language model.
            image_encoder_hidden_size: Hidden size of the image encoder.
            batch_size: Batch size for processing.
            device: Device to run the model on (e.g., 'cuda' or 'cpu').
        """
        super().__init__()

        self.config = config
        self.device = device
        self.batch_size = batch_size

        # Load pretrained DETR model
        # detr_model = DetrForObjectDetection.from_pretrained(
        #     "facebook/detr-resnet-50", revision="no_timm"
        # )

        # Use DetrConfig to create a new configuration with custom num_queries
        self.detr_config = DetrConfig.from_pretrained(
            "facebook/detr-resnet-50", revision="no_timm"
        )
        self.detr_config.num_queries = config.num_query_tokens
        detr_model = DetrForObjectDetection(config=self.detr_config)

        self.class_labels_classifier = detr_model.class_labels_classifier
        self.bbox_predictor = detr_model.bbox_predictor
        self.detr_config = detr_model.config

        # Create DETR layer to use on LLM hidden states
        self.decoder = detr_model.model.decoder
        self.encoder = detr_model.model.encoder

        self.query_position_embeddings = detr_model.model.query_position_embeddings

        # currently shape (batch_size, mm_hidden_size, vocab_size)
        # project to shape (batch_size, mm_hidden_size, d_model)

        self.input_projection = torch.nn.Sequential(
            torch.nn.Linear(llm_hidden_size, self.detr_config.d_model),
            torch.nn.GELU(),
            torch.nn.Linear(self.detr_config.d_model, self.detr_config.d_model),
        )

        self.cross_attention_projection = torch.nn.Sequential(
            torch.nn.Linear(image_encoder_hidden_size, self.detr_config.d_model),
            torch.nn.GELU(),
            torch.nn.Linear(self.detr_config.d_model, self.detr_config.d_model),
        )

        # Make all parameters trainable
        for component in [
            self.input_projection,
            self.cross_attention_projection,
            self.class_labels_classifier,
            self.bbox_predictor,
        ]:
            for param in component.parameters():
                param.requires_grad = True

        if self.config.bbox_ordering == "size_desc":
            log.warning(
                "Bbox ordering is set to size_desc - DETR loss will be using size based matching instead of hungarian matching."
            )

        log.info(f"DETR model loaded for integration with LLM.")

    def forward(
        self,
        input_ids: torch.Tensor,
        outputs: torch.Tensor,
        image_features: torch.Tensor,
        query_tokens_id: torch.Tensor,
        num_query_tokens: int,
    ):
        """Process hidden states with DETR head."""

        # Extract sequence output from hidden states
        # if output of lm_heads is wanted, use outputs[0]
        sequence_output = outputs.hidden_states[-1]

        # Extract query tokens
        sequence_output_queries = []
        for ids in input_ids:
            query_position = torch.where(ids == query_tokens_id[0])[0][0]
            sequence_output_queries.append(
                sequence_output[
                    0, query_position : query_position + num_query_tokens, :
                ].unsqueeze(0)
            )
        sequence_output_queries = torch.cat(sequence_output_queries, dim=0)

        # Project queries to DETR dimension
        projected_queries = self.input_projection(sequence_output_queries)

        if not self.config.add_detr_layers:
            # Simple classification and box prediction
            logits = self.class_labels_classifier(projected_queries)
            pred_boxes = self.bbox_predictor(projected_queries).sigmoid()
        else:
            # Use DETR layers for processing
            # Project image features to DETR dimension,
            # either llm projection + detr input projection or extra detr cross attention layer
            # (need to switch in forward and generate method too image_features_proj or just image_features)
            # image_features = self.detr_input_projection(image_features)
            image_features = self.cross_attention_projection(image_features)

            # Create attention mask - all 1s -> attent to all positions
            encoder_attention_mask = torch.ones(
                image_features.shape[0],
                image_features.shape[1],
                device=projected_queries.device,
                dtype=torch.bool,
            )

            # Get position embeddings
            query_position_embeddings = self.query_position_embeddings.weight.unsqueeze(
                0
            ).repeat(self.batch_size, 1, 1)[:, :num_query_tokens, :]

            # Create object queries
            # added to queries and keys in each cross-attention layer
            object_queries = torch.zeros(
                image_features.shape[0],
                image_features.shape[1],  # Match the sequence length of image_features
                image_features.shape[2],  # Match the feature dimension
                device=projected_queries.device,
                dtype=projected_queries.dtype,
            )

            # Run decoder
            decoder_outputs = self.decoder(
                inputs_embeds=projected_queries,  # Query embeddings from LLM
                attention_mask=None,  # No attention mask for queries
                encoder_hidden_states=image_features,  # Cross-attention features (siglip)
                encoder_attention_mask=encoder_attention_mask,  # Cross-attention mask
                object_queries=object_queries,  # Object queries (zeros)
                query_position_embeddings=query_position_embeddings,  # Position embeddings for self-attention
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False,
            )

            # Get decoder output (tuple with hidden states as first element)
            decoder_output = decoder_outputs[0]

            # Generate predictions
            logits = self.class_labels_classifier(decoder_output)
            pred_boxes = self.bbox_predictor(decoder_output).sigmoid()

        return logits, pred_boxes
