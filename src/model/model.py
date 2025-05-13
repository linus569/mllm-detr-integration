import logging
from typing import List, Optional, Tuple, Union

import torch
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
from transformers import AutoModel, DetrForObjectDetection
from transformers.modeling_outputs import CausalLMOutputWithPast

from model.loss import detr_loss, masked_cross_entropy
from model.partial_frozen_embeddings import (
    PartiallyFrozenEmbedding,
    PartiallyFrozenLMHead,
)
from utils.config import ExperimentConfig

log = logging.getLogger(__name__)


class VisionLanguageModel(torch.nn.Module):
    def __init__(
        self,
        config: ExperimentConfig,
        image_token_index: int,
        num_new_tokens: int = 0,
        tokenizer_size: int = None,
        initializers: list[list[int]] = None,
        do_init: bool = True,
        query_tokens_id: Optional[List[int]] = None,
    ):
        super(VisionLanguageModel, self).__init__()

        if do_init:
            assert (
                initializers is not None
            ), "Initializers should be provided for new tokens"

        self.config = config
        self.query_tokens_id = query_tokens_id

        # Get model components
        # device_map="auto", prints warning, could be ignored https://github.com/huggingface/transformers/issues/31544
        # torch_dtype=self.torch_dtype, attn_implementation=attn_implementation,
        self.model = LlavaQwenForCausalLM.from_pretrained(self.config.model_name)

        if self.config.image_encoder.name == "dinov2":
            log.info("Using DINOv2 image encoder")
            self.image_encoder = AutoModel.from_pretrained("facebook/dinov2-small")
            llm_embed_dim = self.model.config.hidden_size
            dinov2_embed_dim = self.image_encoder.config.hidden_size

            self.projector = torch.nn.Sequential(
                torch.nn.Linear(dinov2_embed_dim, llm_embed_dim),  # 4*llm_embed_dim),
                torch.nn.GELU(),
                torch.nn.Linear(llm_embed_dim, llm_embed_dim),
            )

            # Freeze all image encoder parameters
            for param in self.image_encoder.parameters():
                param.requires_grad = False

            log.info(
                f"Image encoder {self.config.image_encoder.name} initialized with output dim {dinov2_embed_dim} -> {llm_embed_dim}"
            )
        elif "resnet50" in self.config.image_encoder.name:
            log.info("Using ResNet50 image encoder")

            model = DetrForObjectDetection.from_pretrained(
                "facebook/detr-resnet-50", revision="no_timm"
            )
            self.model_detr = model.model
            self.image_encoder = model.model.backbone
            llm_embed_dim = self.model.config.hidden_size

            resnet_embed_dim = (
                self.image_encoder.conv_encoder.intermediate_channel_sizes[-1]
            )
            self.projector = torch.nn.Sequential(
                torch.nn.Linear(resnet_embed_dim, llm_embed_dim),
                torch.nn.GELU(),
                torch.nn.Linear(llm_embed_dim, llm_embed_dim),
            )
            if "encoder" in self.config.image_encoder.name:
                self.projector = torch.nn.Sequential(
                    torch.nn.Linear(256, llm_embed_dim),
                    torch.nn.GELU(),
                    torch.nn.Linear(llm_embed_dim, llm_embed_dim),
                )

            log.info(
                f"Image encoder {self.config.image_encoder.name} initialized with output dim {resnet_embed_dim}"
            )
        else:
            log.info(
                f"Using default image encoder from model: {self.config.image_encoder.name}"
            )
            self.image_encoder = self.model.get_vision_tower()
            self.projector = self.model.get_model().mm_projector

        # freeze all parameters
        if config.freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False

        # check if image_encoder params are frozen
        for param in self.image_encoder.parameters():
            assert not param.requires_grad, "Image encoder parameters should be frozen"

        # unfreeze projector parameters for fine-tuning
        for param in self.projector.parameters():
            param.requires_grad = True

        # Resize embeddings to correct start size if necessary
        if tokenizer_size is None:
            tokenizer_size = self.model.get_input_embeddings().num_embeddings
            log.warning("Tokenizer size not provided. Using model vocab size.")

        embedding_size = self.model.get_input_embeddings().num_embeddings
        if embedding_size != tokenizer_size:
            log.warning(
                f"Tokenizer vocab size {tokenizer_size} does not match model vocab size {embedding_size}. "
                "Resizing model embeddings to match tokenizer size."
            )
            # Resize token embeddings
            self.model.resize_token_embeddings(tokenizer_size)
            assert (
                self.model.get_input_embeddings().num_embeddings == tokenizer_size
            ), f"Model vocab size {self.model.get_input_embeddings().num_embeddings} does not match tokenizer size {tokenizer_size} after resizing."

        # Initialize new token embeddings
        self.model.set_input_embeddings(
            PartiallyFrozenEmbedding(
                frozen_embedding=self.model.get_input_embeddings(),
                new_tokens=num_new_tokens,
                initializers=initializers,
                do_init=do_init,
            )
        )
        self.model.set_output_embeddings(
            PartiallyFrozenLMHead(
                frozen_lm_head=self.model.get_output_embeddings(),
                new_tokens=num_new_tokens,
                initializers=initializers,
                do_init=do_init,
            )
        )

        log.info(
            f"frozen input embed: {self.model.get_input_embeddings().frozen_embedding}, "
            f"trainable input embed: {self.model.get_input_embeddings().trainable_embedding}, "
            f"full input embed size: {self.model.get_input_embeddings().num_embeddings}, "
            f"frozen output embed: {self.model.get_output_embeddings().frozen_lm_head}, "
            f"trainable output embed: {self.model.get_output_embeddings().trainable_lm_head}, "
            f"full output embed size: {self.model.get_output_embeddings().out_features};"
        )

        self.vocab_size = self.model.get_input_embeddings().num_embeddings
        self.image_token_index = image_token_index

        if self.config.detr_loss:
            log.info("Using DETR loss")

            detr_model = DetrForObjectDetection.from_pretrained(
                "facebook/detr-resnet-50", revision="no_timm"
            )
            self.detr_model = detr_model

            self.detr_class_labels_classifier = detr_model.class_labels_classifier
            self.detr_bbox_predictor = detr_model.bbox_predictor
            self.detr_config = detr_model.config

            # create detr layers that I will use on the hidden states
            self.detr_decoder = detr_model.model.decoder
            self.detr_encoder = detr_model.model.encoder

            # currently shape (batch_size, mm_hidden_size, vocab_size)
            # project to shape (batch_size, mm_hidden_size, d_model)

            # hidden_size = self.vocab_size # when using outputs[0], output from lm_head
            hidden_size = self.model.config.hidden_size  # when using hidden states

            self.detr_input_projection = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, self.detr_config.d_model),
                torch.nn.GELU(),
                torch.nn.Linear(self.detr_config.d_model, self.detr_config.d_model),
            )

            self.detr_cross_attention_projection = torch.nn.Sequential(
                torch.nn.Linear(
                    self.model.config.mm_hidden_size, self.detr_config.d_model
                ),
                torch.nn.GELU(),
                torch.nn.Linear(self.detr_config.d_model, self.detr_config.d_model),
            )

            for param in self.detr_input_projection.parameters():
                assert (
                    param.requires_grad == True
                ), "Input projection parameters should be trainable"

            for param in self.detr_class_labels_classifier.parameters():
                assert (
                    param.requires_grad == True
                ), "DETR class_labels_classifier parameters should be trainable"

            for param in self.detr_bbox_predictor.parameters():
                assert (
                    param.requires_grad == True
                ), "DETR bbox_predictor parameters should be trainable"

            if self.config.bbox_ordering == "size_desc":
                log.warning(
                    "Bbox ordering is set to size_desc - DETR loss will be using size based matching instead of hungarian matching."
                )

        if self.config.use_precompute:
            # delete image encoder and projector
            # TODO: delete self.model.model.vision_tower also during normal runs as I use self.image_encoder
            del self.image_encoder
            del self.model.model.vision_tower
            del self.model.model.vision_resampler

        log.info("Model initialized")

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        log.info("Reading state dict")

        projector_state_dict = self.projector.state_dict(
            destination=destination, prefix=prefix + "projector.", keep_vars=keep_vars
        )
        input_embeddings_state_dict = self.model.get_input_embeddings().state_dict(
            destination=destination,
            prefix=prefix + "input_embeddings.",
            keep_vars=keep_vars,
        )
        output_embeddings_state_dict = self.model.get_output_embeddings().state_dict(
            destination=destination,
            prefix=prefix + "output_embeddings.",
            keep_vars=keep_vars,
        )

        if not self.config.freeze_model:
            # add model state dict to the output
            model_state_dict = self.model.state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars
            )
            return {
                **projector_state_dict,
                **input_embeddings_state_dict,
                **output_embeddings_state_dict,
                **model_state_dict,
            }

        return {
            **projector_state_dict,
            **input_embeddings_state_dict,
            **output_embeddings_state_dict,
        }

    def load_state_dict(self, state_dict, strict=True):
        log.info("Loading state dict")
        is_torch_compile = next(iter(state_dict.keys())).startswith("_orig_mod.")
        if is_torch_compile:
            assert all(
                [k.startswith("_orig_mod.") for k in state_dict.keys()]
            ), "State dict should be from torch compile"
            state_dict = {k[len("_orig_mod.") :]: v for k, v in state_dict.items()}
        projector_state_dict = {
            k[len("projector.") :]: v
            for k, v in state_dict.items()
            if k.startswith("projector.")
        }
        input_embeddings_state_dict = {
            k[len("input_embeddings.") :]: v
            for k, v in state_dict.items()
            if k.startswith("input_embeddings.")
        }
        output_embeddings_state_dict = {
            k[len("output_embeddings.") :]: v
            for k, v in state_dict.items()
            if k.startswith("output_embeddings.")
        }

        missing1, unexpected1 = self.projector.load_state_dict(
            projector_state_dict, strict=strict
        )
        missing2, unexpected2 = self.model.get_input_embeddings().load_state_dict(
            input_embeddings_state_dict, strict=strict
        )
        missing3, unexpected3 = self.model.get_output_embeddings().load_state_dict(
            output_embeddings_state_dict, strict=strict
        )
        missing = missing1 + missing2 + missing3
        unexpected = unexpected1 + unexpected2 + unexpected3

        return missing, unexpected

    def tie_weights(self):
        # TODO: is never called, should be called in __init__ if tie_word_embeddings is True
        # TODO: if self.language_config.tie_word_embeddings:
        output_embeddings: PartiallyFrozenLMHead = self.model.get_output_embeddings()
        input_embeddings: PartiallyFrozenEmbedding = self.model.get_input_embeddings()

        output_embeddings.frozen_lm_head.weight = (
            input_embeddings.frozen_embedding.weight
        )
        output_embeddings.trainable_lm_head.weight = (
            input_embeddings.trainable_embedding.weight
        )

    def _get_image_features(
        self, pixel_values: torch.FloatTensor, image_sizes: List[Tuple[int, int]]
    ) -> torch.FloatTensor:
        assert pixel_values.dim() in (
            4,
            5,
        ), "Image should be of shape [batch_size, channels, height, width] or [batch_size, num_patches, channels, height, width]"

        # Image feature extraction
        pixel_values = pixel_values.to(self.config.device)
        encoder_name = self.config.image_encoder.name

        if pixel_values.ndim == 5:  # If patches are used
            image_features = []
            for img in pixel_values:
                if "dinov2" in encoder_name:
                    outputs = self.image_encoder(img)
                    if self.config.image_encoder.use_pooler_output:
                        # Extract the [CLS] token representation (first token)
                        img_features = outputs.pooler_output.unsqueeze(1)
                    else:
                        # Use last hidden state
                        img_features = outputs.last_hidden_state
                elif "siglip" in encoder_name:
                    img_features = self.image_encoder(img)
                else:
                    raise ValueError(
                        f"Image encoder {encoder_name} not yet supported for patches"
                    )
                image_features.append(img_features)
            # convert list to tensor
            image_features = torch.stack(image_features)
        else:
            # # Ensure image is in correct format [batch_size, channels, height, width]
            # if pixel_values.shape[-1] == 3:  # If channels are last
            #     pixel_values = pixel_values.permute(0, 3, 1, 2)
            # image_features = self.image_encoder(pixel_values)
            if encoder_name == "dinov2":
                outputs = self.image_encoder(pixel_values)
                if self.config.image_encoder.use_pooler_output:
                    # Extract the [CLS] token representation (first token)
                    image_features = outputs.pooler_output.unsqueeze(1)
                else:
                    image_features = outputs.last_hidden_state
            elif "resnet50" in encoder_name:
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

                # pixel_mask = torch.ones(
                #     pixel_values.shape[0],
                #     pixel_values.shape[2],
                #     pixel_values.shape[3],
                #     dtype=torch.bool,
                # )

                pixel_mask = pixel_mask.to(pixel_values.device)
                # pixel_values = pixel_values.flatten(0, 1)
                features, object_queries_list = self.image_encoder(
                    pixel_values, pixel_mask
                )
                # print(image_sizes, pixel_values.shape)

                # get final feature map and downsampled mask
                feature_map, mask = features[-1]

                if mask is None:
                    raise ValueError("Backbone does not return downsampled pixel mask")

                if "encoder" in self.config.image_encoder.name:
                    # Second, apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
                    projected_feature_map = self.model_detr.input_projection(
                        feature_map
                    )

                    # Third, flatten the feature map + position embeddings of shape NxCxHxW to NxCxHW, and permute it to NxHWxC
                    # In other words, turn their shape into (batch_size, sequence_length, hidden_size)
                    flattened_features = projected_feature_map.flatten(2).permute(
                        0, 2, 1
                    )
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
                    # # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
                    # elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
                    #     encoder_outputs = BaseModelOutput(
                    #         last_hidden_state=encoder_outputs[0],
                    #         hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                    #         attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                    #     )
                    # print(encoder_outputs)
                    # tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)

                    image_features = encoder_outputs[0]
                    # image_features = image_features.permute(0, 2, 1)
                    # print(image_features.shape)

                    # image_features = encoder_outputs[1]
                    # print(image_features.shape)
                else:
                    image_features = feature_map.flatten(2).permute(0, 2, 1)
                    # print(image_features.shape)
            else:
                # use siglip, already in correct format
                image_features = self.image_encoder(pixel_values)

        image_features = image_features.to(self.model.device, self.model.dtype)
        # print("image_features:", image_features.shape)
        # Project image features to token size
        image_features_proj = self.projector(image_features)
        # print("projected image_features:", image_features.shape)
        # shape of image_features is (batch_size, (num_patches), num_image_tokens, token_size_text_encoder)

        return image_features, image_features_proj

    def _use_detr_head(self, input_ids, outputs, image_features):
        # get the last hidden state or output from lm_head
        # sequence_output = outputs[0]
        sequence_output = outputs.hidden_states[-1]

        # get the last hidden state of the decoder
        # sequence_output = outputs[1][-1]
        # project to d_model size to match the classifier and predictor

        sequence_output_queries = []
        for ids in input_ids:
            query_position = torch.where(ids == self.query_tokens_id[0])[0][0]
            sequence_output_queries.append(
                sequence_output[
                    0, query_position : query_position + self.config.num_query_tokens, :
                ].unsqueeze(0)
            )
        sequence_output_queries = torch.cat(sequence_output_queries, dim=0)
        projected_queries = self.detr_input_projection(sequence_output_queries)

        if not self.config.add_detr_layers:
            logits = self.detr_class_labels_classifier(projected_queries)
            pred_boxes = self.detr_bbox_predictor(projected_queries).sigmoid()

            return logits, pred_boxes
        else:
            # Project image features to detr dimension, either llm projection + detr input projection or extra detr cross attention projection
            # (need to switch in forward and generate method too image_features_proj or just image_features)
            # image_features = self.detr_input_projection(image_features)
            image_features = self.detr_cross_attention_projection(image_features)

            # Create a mask for the encoder outputs - all 1s means attend to all positions
            encoder_attention_mask = torch.ones(
                image_features.shape[0],
                image_features.shape[1],
                device=projected_queries.device,
                dtype=torch.bool,
            )

            # Create position embeddings for the query tokens
            query_position_embeddings = (
                self.detr_model.model.query_position_embeddings.weight.unsqueeze(
                    0
                ).repeat(self.config.batch_size, 1, 1)[
                    :, : self.config.num_query_tokens, :
                ]
            )

            # Create object queries (are added to queries and keys in each cross-attention layer)
            object_queries = torch.zeros(
                image_features.shape[0],
                image_features.shape[1],  # Match the sequence length of image_features
                image_features.shape[2],  # Match the feature dimension
                device=projected_queries.device,
                dtype=projected_queries.dtype,
            )

            # Run the decoder
            decoder_outputs = self.detr_decoder(
                inputs_embeds=projected_queries,  # Query embeddings from LLM
                attention_mask=None,  # No masking needed for queries
                encoder_hidden_states=image_features,  # Cross-attention features (siglip features)
                encoder_attention_mask=encoder_attention_mask,  # Cross-attention mask
                object_queries=object_queries,  # No object queries needed
                query_position_embeddings=query_position_embeddings,  # Position embeddings for self-attention
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False,
            )

            # Get decoder output (tuple with hidden states as first element)
            decoder_output = decoder_outputs[0]

            # Apply classification and box regression heads
            logits = self.detr_class_labels_classifier(decoder_output)
            pred_boxes = self.detr_bbox_predictor(decoder_output).sigmoid()

            return logits, pred_boxes

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        images: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        if inputs_embeds is not None and images is not None:
            raise ValueError("You cannot specify both inputs_embeds and images")

        if inputs_embeds is None:
            # Token embeddings
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        if images is not None:
            if self.config.use_precompute:
                # Project image features to token size
                image_features = images
                image_features_proj = self.projector(image_features).to(
                    self.model.device, self.model.dtype
                )
            else:
                image_features, image_features_proj = self._get_image_features(
                    images, image_sizes
                )

            image_features_detr = image_features.clone()

            # Integrate image features into token embeddings
            n_image_tokens = (input_ids == self.image_token_index).sum()
            n_image_features = (
                image_features_proj.shape[0] * image_features_proj.shape[1]
            )
            if image_features_proj.ndim == 4:  # if patches are used
                n_image_features *= image_features_proj.shape[2]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}. "
                    f"images: {images.shape}, image_sizes: {image_sizes}, image_features: {image_features_proj.shape}, input_ids: {input_ids.shape}"
                )
            # special_image_mask shape is (batch_size, seq_len, token_size_text_encoder)
            # inputs_embeds shape is (batch_size, seq_len, token_size_text_encoder)
            # image_features shape is (batch_size, num_image_tokens, token_size_text_encoder)
            special_image_mask = (input_ids == self.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(
                inputs_embeds.device
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask, image_features_proj
            )

        # Hidden states needed for DETR loss
        if self.config.detr_loss:
            output_hidden_states = True

        # LLM forward pass
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        loss = None
        if labels is not None:
            if self.config.detr_loss:
                logits, pred_boxes = self._use_detr_head(
                    input_ids, outputs, image_features_detr
                )

                outputs_class = outputs_coord = None  # used only with auxiliary outputs

                loss = detr_loss(
                    logits=logits,
                    labels=labels,
                    device=self.config.device,
                    pred_boxes=pred_boxes,
                    config=self.detr_config,
                    outputs_class=outputs_class,
                    outputs_coord=outputs_coord,
                    sized_based_matching=self.config.bbox_ordering == "size_desc",
                )
            else:
                loss = masked_cross_entropy(
                    outputs.logits, labels, vocab_size=self.vocab_size
                )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            # TODO: return also image_hidden states?
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        attention_mask,
        image,
        max_new_tokens=2048,
        stopping_criteria=None,
        image_sizes: Optional[List[List[int]]] = None,
        **kwargs,
    ):
        if image is None:
            raise ValueError("Image needs to be provided for generation.")

        if self.config.use_precompute:
            # Project image features to token size
            image_features = image
            image_features_proj = self.projector(image_features)
            image_features_proj = image_features_proj.to(
                self.model.device, self.model.dtype
            )
        else:
            # Image feature extraction
            image_features, image_features_proj = self._get_image_features(
                image, image_sizes
            )

        image_features_detr = image_features.clone()

        # Token embeddings
        embedding_layer = self.model.get_input_embeddings()
        inputs_embeds = embedding_layer(input_ids)

        # Integrate image features into embeddings
        special_image_mask = (input_ids == self.image_token_index).unsqueeze(-1)
        special_image_mask = special_image_mask.expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(
            special_image_mask, image_features_proj
        )

        # Generate text
        if self.config.detr_loss:
            # LLM forward pass
            assert self.model.eval()
            outputs = self.model(
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=inputs_embeds,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=True,  # needed
                return_dict=None,
            )

            logits, pred_boxes = self._use_detr_head(
                input_ids, outputs, image_features_detr
            )

            outputs = {
                "pred_boxes": pred_boxes,
                "pred_logits": logits,
            }
        else:
            # direcly call generate on superclass as inputs_embeds is already created and class implementation does not support this
            outputs = super(LlavaQwenForCausalLM, self.model).generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_new_tokens,
                attention_mask=attention_mask,
                stopping_criteria=stopping_criteria,
                **kwargs,
            )

        return outputs


if __name__ == "__main__":
    model_name = "lmms-lab/llava-onevision-qwen2-0.5b-si"
    from ..utils.config import ExperimentConfig

    config = ExperimentConfig()

    model = VisionLanguageModel(config=config)
    # get one image and input into forward pass
    image = torch.rand(1, 3, 384, 384)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
    output = model(input_ids, attention_mask, image)
    log.info(output.size())
