import logging
from typing import List, Optional, Tuple

import torch
from transformers import (
    DabDetrConfig,
    DabDetrForObjectDetection,
    DetrConfig,
    DetrForObjectDetection,
)

# from transformers.models.detr.modeling_detr import (
#     DetrSinePositionEmbedding,
# )
from transformers.models.dab_detr.modeling_dab_detr import (
    DabDetrSinePositionEmbedding,
    inverse_sigmoid,
)

from model.loss import dab_detr_loss, detr_loss
from utils.config import ExperimentConfig

log = logging.getLogger(__name__)


class DETROutput:
    """
    Class to hold DETR model outputs.
    """

    def __init__(
        self,
        logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        last_hidden_state: Optional[torch.Tensor] = None,
    ):
        """
        Initialize the DETROutput.

        Args:
            logits: Class logits from the DETR model.
            pred_boxes: Predicted bounding boxes from the DETR model.
            last_hidden_state: Optional last hidden state from the model, if available.
        """
        self.logits = logits
        self.pred_boxes = pred_boxes
        self.last_hidden_state = last_hidden_state


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

        for param in self.decoder.parameters():
            param.requires_grad = self.config.train_detr
        for param in self.encoder.parameters():
            param.requires_grad = self.config.train_detr

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
        **kwargs,
    ):
        """Process hidden states with DETR head."""

        # Extract sequence output from hidden states
        # if output of lm_heads is wanted, use outputs[0]
        sequence_output = outputs.hidden_states[-1]

        # Extract query tokens
        sequence_output_queries = []
        for idx, ids in enumerate(input_ids):
            query_position = torch.where(ids == query_tokens_id[0])[0][0]
            sequence_output_queries.append(
                sequence_output[
                    idx, query_position : query_position + num_query_tokens, :
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
            ).repeat(
                self.batch_size, 1, 1
            )  # [:, :num_query_tokens, :]

            # Create object queries
            # added to queries and keys in each cross-attention layer
            object_queries = torch.zeros(
                image_features.shape[0],
                image_features.shape[1],  # Match the sequence length of image_features
                image_features.shape[2],  # Match the feature dimension
                device=projected_queries.device,
                dtype=projected_queries.dtype,
            )

            # inputs_embeds = torch.zeros_like(projected_queries)

            # object_queries = DetrSinePositionEmbedding(128, normalize=True)(
            #     projected_queries, torch.ones_like(image_features[:, :, 0])
            # )

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

        return DETROutput(
            logits=logits,
            pred_boxes=pred_boxes,
            last_hidden_state=decoder_output,
        )

    def loss(
        self,
        logits,
        labels,
        pred_boxes,
        outputs_class=None,
        outputs_coord=None,
        **kwargs,
    ):
        return detr_loss(
            logits=logits,
            labels=labels,
            device=self.config.device,
            pred_boxes=pred_boxes,
            config=self.detr_config,
            outputs_class=outputs_class,
            outputs_coord=outputs_coord,
            sized_based_matching=self.config.bbox_ordering == "size_desc",
        )


class FullDETRIntegration(torch.nn.Module):
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
        self.detr_model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50", revision="no_timm"
        ).to(device)

        # Use DetrConfig to create a new configuration with custom num_queries
        self.detr_config = self.detr_model.config

        # currently shape (batch_size, mm_hidden_size, vocab_size)
        # project to shape (batch_size, mm_hidden_size, d_model)

        self.input_projection = torch.nn.Sequential(
            torch.nn.Linear(llm_hidden_size, self.detr_config.d_model),
            torch.nn.GELU(),
            torch.nn.Linear(self.detr_config.d_model, self.detr_config.d_model),
        )

        # Only freeze backbone, not the full model
        for name, param in self.detr_model.named_parameters():
            if "backbone" in name:
                param.requires_grad = False
            else:
                param.requires_grad = self.config.train_detr

        if self.config.bbox_ordering == "size_desc":
            log.warning(
                "Bbox ordering is set to size_desc - DETR loss will be using size based matching instead of hungarian matching."
            )

        log.info(f"Full DETR model incl. RestNet50 loaded for integration with LLM.")

    def forward(
        self,
        input_ids: torch.Tensor,
        outputs: torch.Tensor,
        image_features: torch.Tensor,
        query_tokens_id: torch.Tensor,
        num_query_tokens: int,
        pixel_values: Optional[torch.Tensor] = None,
        image_sizes: Optional[List[Tuple[int, int]]] = None,
        **kwargs,
    ):
        """Process hidden states with DETR head."""

        # Extract sequence output from hidden states
        # if output of lm_heads is wanted, use outputs[0]
        sequence_output = outputs.hidden_states[-1]

        # Extract query tokens
        sequence_output_queries = []
        for idx, ids in enumerate(input_ids):
            query_position = torch.where(ids == query_tokens_id[0])[0][0]
            sequence_output_queries.append(
                sequence_output[
                    idx, query_position : query_position + num_query_tokens, :
                ].unsqueeze(0)
            )
        sequence_output_queries = torch.cat(sequence_output_queries, dim=0)

        # Project queries to DETR dimension
        projected_queries = self.input_projection(sequence_output_queries)

        pixel_values = (
            pixel_values.to(self.device) if pixel_values is not None else None
        )
        output = self.detr_model.forward(
            pixel_values=pixel_values,
            inputs_embeds=projected_queries,  # Query embeddings from LLM
            return_dict=True,
        )

        self.loss_value = output.loss if hasattr(output, "loss") else None

        return DETROutput(
            logits=output.logits,
            pred_boxes=output.pred_boxes,
            last_hidden_state=output.last_hidden_state,
        )

    def loss(
        self,
        logits,
        labels,
        pred_boxes,
        outputs_class=None,
        outputs_coord=None,
        **kwargs,
    ):
        # return self.loss_value
        return detr_loss(
            logits=logits,
            labels=labels,
            device=self.config.device,
            pred_boxes=pred_boxes,
            config=self.detr_config,
            outputs_class=outputs_class,
            outputs_coord=outputs_coord,
            sized_based_matching=self.config.bbox_ordering == "size_desc",
        )


class DabDETRIntegration(torch.nn.Module):
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

        # DabDETR model
        # detr_model = DetrForObjectDetection.from_pretrained(
        #     "IDEA-Research/dab-detr-resnet-50"
        # )
        # dab_config = DabDetrConfig.from_pretrained("IDEA-Research/dab-detr-resnet-50")
        # dab_config.use_timm_backbone = False
        # detr_model = DabDetrModel.from_pretrained(
        #     "IDEA-Research/dab-detr-resnet-50", use_pretrained_backbone=False, use_timm_backbone=False#, backbone=None
        # )
        detr_config = DabDetrConfig.from_pretrained("IDEA-Research/dab-detr-resnet-50")
        detr_config.use_pretrained_backbone = False
        detr_config.use_timm_backbone = False
        detr_config.backbone = "microsoft/resnet-50"
        # config.use_backbone = False
        # config.backbone = None

        detr_model = DabDetrForObjectDetection.from_pretrained(
            "IDEA-Research/dab-detr-resnet-50",
            config=detr_config,
            ignore_mismatched_sizes=True,
        )

        self.detr_config = detr_model.config

        # Create DETR layer to use on LLM hidden states
        self.decoder = detr_model.model.decoder
        self.encoder = detr_model.model.encoder

        self.position_embedding = DabDetrSinePositionEmbedding(self.detr_config)

        # self.query_refpoint_embeddings = detr_model.model.query_refpoint_embeddings

        self.query_refpoint_embeddings = torch.nn.Embedding(
            self.config.num_query_tokens, self.detr_config.query_dim
        )

        # currently shape (batch_size, mm_hidden_size, vocab_size)
        # project to shape (batch_size, mm_hidden_size, d_model)

        self.input_projection = torch.nn.Sequential(
            torch.nn.Linear(llm_hidden_size, self.detr_config.hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(self.detr_config.hidden_size, self.detr_config.hidden_size),
        )

        self.cross_attention_projection = torch.nn.Sequential(
            torch.nn.Linear(image_encoder_hidden_size, self.detr_config.hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(self.detr_config.hidden_size, self.detr_config.hidden_size),
        )

        self.class_embed = detr_model.class_embed
        self.bbox_predictor = detr_model.bbox_predictor

        # Make all parameters trainable
        for component in [
            self.input_projection,
            self.cross_attention_projection,
            self.class_embed,
            self.bbox_predictor,
        ]:
            for param in component.parameters():
                param.requires_grad = True

        if self.config.bbox_ordering == "size_desc":
            log.warning(
                "Bbox ordering is set to size_desc - DETR loss will be using size based matching instead of hungarian matching."
            )

        log.info(f"DAB-DETR model loaded for integration with LLM.")

    def forward(
        self,
        input_ids: torch.Tensor,
        outputs: torch.Tensor,
        image_features: torch.Tensor,
        query_tokens_id: torch.Tensor,
        num_query_tokens: int,
        **kwargs,
    ):
        """Process hidden states with DETR head."""

        # Extract sequence output from hidden states
        # if output of lm_heads is wanted, use outputs[0]
        sequence_output = outputs.hidden_states[-1]

        # Extract query tokens
        sequence_output_queries = []
        for idx, ids in enumerate(input_ids):
            query_position = torch.where(ids == query_tokens_id[0])[0][0]
            sequence_output_queries.append(
                sequence_output[
                    idx, query_position : query_position + num_query_tokens, :
                ].unsqueeze(0)
            )
        sequence_output_queries = torch.cat(sequence_output_queries, dim=0)

        # Project queries to DETR dimension
        projected_queries = self.input_projection(sequence_output_queries)

        if not self.config.add_detr_layers:
            # error not implemented
            raise NotImplementedError(
                "Simple classification and box prediction is not implemented for DabDETRIntegration."
            )
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
                device=self.config.device,
                dtype=torch.bool,
            )

            # Get position embeddings
            reference_position_embeddings = (
                self.query_refpoint_embeddings.weight.unsqueeze(0).repeat(
                    self.batch_size, 1, 1
                )
            )

            # Create object queries
            # added to queries and keys in each cross-attention layer
            # object_queries = torch.zeros(
            #     image_features.shape[0],
            #     image_features.shape[1],  # Match the sequence length of image_features
            #     image_features.shape[2],  # Match the feature dimension
            #     device=self.query_refpoint_embeddings.device,
            #     dtype=self.query_refpoint_embeddings.dtype,
            # )
            # Generate object queries with spatial information for SIGLIP features
            # # Convert SIGLIP features to a spatial representation
            # batch_size, seq_len, hidden_dim = image_features.shape

            # # Calculate approximate spatial dimensions (assuming square feature map)
            # feature_map_size = int(seq_len**0.5)

            # # Create 2D position embeddings
            # y_embed = torch.arange(feature_map_size, device=image_features.device).float()
            # x_embed = torch.arange(feature_map_size, device=image_features.device).float()

            # # Normalize to [0, 1]
            # y_embed = y_embed / (feature_map_size - 1)
            # x_embed = x_embed / (feature_map_size - 1)

            # # Create grid
            # y_embed, x_embed = torch.meshgrid(y_embed, x_embed, indexing="ij")

            # # Reshape to match feature sequence
            # pos_embed = torch.cat([
            #     y_embed.flatten().unsqueeze(-1),
            #     x_embed.flatten().unsqueeze(-1)
            # ], dim=-1)

            # # Expand to batch size
            # pos_embed = pos_embed.unsqueeze(0).repeat(batch_size, 1, 1)

            # # Scale to match image_features dimensions
            # pos_embed = torch.nn.functional.pad(pos_embed, (0, hidden_dim - 2), "constant", 0)

            # # Object queries are position embeddings for the image features
            # object_queries = pos_embed.to(
            #     device=reference_position_embeddings.device,
            #     dtype=reference_position_embeddings.dtype
            # )
            # Create a pseudo 2D representation of your features to use with DabDetrSinePositionEmbedding
            batch_size, seq_len, hidden_dim = image_features.shape
            feature_map_size = int(seq_len**0.5)

            # Reshape features to 2D spatial layout (assuming square feature map)
            reshaped_features = image_features.reshape(
                batch_size, feature_map_size, feature_map_size, hidden_dim
            ).permute(
                0, 3, 1, 2
            )  # [B, C, H, W]

            # Create a pixel mask for all valid positions
            pixel_mask = torch.ones(
                batch_size,
                feature_map_size,
                feature_map_size,
                device=image_features.device,
            )

            # Generate position embeddings using DabDetrSinePositionEmbedding
            pos_embeddings = self.position_embedding(reshaped_features, pixel_mask)

            # Flatten back to match image_features shape
            object_queries = pos_embeddings.flatten(2).permute(0, 2, 1)  # [B, H*W, C]

            # Make sure object_queries has the right shape
            if object_queries.shape[1] < image_features.shape[1]:
                # Pad if needed
                padding = torch.zeros(
                    batch_size,
                    image_features.shape[1] - object_queries.shape[1],
                    hidden_dim,
                    device=object_queries.device,
                    dtype=object_queries.dtype,
                )
                object_queries = torch.cat([object_queries, padding], dim=1)
            elif object_queries.shape[1] > image_features.shape[1]:
                # Truncate if needed
                object_queries = object_queries[:, : image_features.shape[1], :]

            queries = torch.zeros(
                self.batch_size,
                self.config.num_query_tokens,
                self.detr_config.hidden_size,
                device=self.device,
            )

            return_dict = True
            decoder_outputs = self.decoder(
                inputs_embeds=projected_queries,  # maybe use from llm?, query embeddings passed to decoder
                query_position_embeddings=reference_position_embeddings,  # position embeddings for self-attention
                object_queries=object_queries,  # position embeddings, added to q and k in cross-attention
                encoder_hidden_states=image_features,  # output of encoder, used for cross-attention
                memory_key_padding_mask=~encoder_attention_mask,  # which positions to ignore in encoder outputs
                output_attentions=False,
                output_hidden_states=False,
                return_dict=return_dict,
            )

            # Get decoder output (tuple with hidden states as first element)
            # decoder_output = decoder_outputs[0]

            # Generate predictions
            reference_points = (
                decoder_outputs.reference_points if return_dict else decoder_outputs[-1]
            )
            intermediate_hidden_states = (
                decoder_outputs.intermediate_hidden_states
                if return_dict
                else decoder_outputs[-2]
            )

            # class logits + predicted bounding boxes
            logits = self.class_embed(intermediate_hidden_states[-1])

            reference_before_sigmoid = inverse_sigmoid(reference_points)
            bbox_with_refinement = self.bbox_predictor(intermediate_hidden_states)
            bbox_with_refinement[
                ..., : self.detr_config.query_dim
            ] += reference_before_sigmoid
            outputs_coord = bbox_with_refinement.sigmoid()

            pred_boxes = outputs_coord[-1]

            # loss, loss_dict, auxiliary_outputs = None, None, None
            # if labels is not None:
            #     outputs_class = None
            #     if self.config.auxiliary_loss:
            #         outputs_class = self.class_embed(intermediate_hidden_states)
            #     loss, loss_dict, auxiliary_outputs = self.loss_function(
            #         logits, labels, self.device, pred_boxes, self.config, outputs_class, outputs_coord
            #     )

        # print(logits, pred_boxes)

        return DETROutput(
            logits=logits,
            pred_boxes=pred_boxes,
            last_hidden_state=None,
        )

    def loss(
        self,
        logits,
        labels,
        pred_boxes,
        outputs_class=None,
        outputs_coord=None,
        **kwargs,
    ):
        return dab_detr_loss(
            logits=logits,
            labels=labels,
            device=self.config.device,
            pred_boxes=pred_boxes,
            config=self.detr_config,
            outputs_class=outputs_class,
            outputs_coord=outputs_coord,
        )
