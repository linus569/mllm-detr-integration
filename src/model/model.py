import logging
from typing import List, Optional, Tuple, Union

import torch
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
from transformers import AutoModel, DetrForObjectDetection
from transformers.modeling_outputs import CausalLMOutputWithPast

from model.detr_integration import DETRIntegration
from model.image_encoder import (
    BaseImageEncoder,
    DinoV2ImageEncoder,
    Resnet50ImageEncoder,
    SiglipImageEncoder,
)
from model.loss import detr_loss, masked_cross_entropy
from model.partial_frozen_embeddings import (
    PartiallyFrozenEmbedding,
    PartiallyFrozenLMHead,
)
from utils.config import ExperimentConfig

log = logging.getLogger(__name__)

dict_image_encoders = {
    "dinov2": DinoV2ImageEncoder,
    "siglip": SiglipImageEncoder,
    "resnet50": Resnet50ImageEncoder,
    "resnet50_encoder": Resnet50ImageEncoder,
}


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
            msg = "Initializers should be provided for new tokens"
            assert initializers is not None, msg

        self.config = config
        self.query_tokens_id = query_tokens_id

        # Get model components
        # device_map="auto", prints warning, could be ignored https://github.com/huggingface/transformers/issues/31544
        # torch_dtype=self.torch_dtype, attn_implementation=attn_implementation,
        self.model = LlavaQwenForCausalLM.from_pretrained(self.config.model_name)

        # Get image encoder based on config
        self.image_encoder: BaseImageEncoder
        self.image_encoder = dict_image_encoders.get(self.config.image_encoder.name)(
            model_name=self.config.image_encoder.model_path,
            config=self.config,
            model=self.model if self.config.image_encoder.name == "siglip" else None,
        )

        # Check if image encoder is frozen
        for param in self.image_encoder.encoder.parameters():
            assert (
                param.requires_grad == False
            ), "Image encoder parameters should be frozen"

        llm_embed_dim = self.model.config.hidden_size
        image_encoder_embed_dim = self.image_encoder.get_output_dim()

        if "siglip" in self.config.image_encoder.name:
            self.projector = self.model.get_model().mm_projector
        else:
            self.projector = torch.nn.Sequential(
                torch.nn.Linear(image_encoder_embed_dim, llm_embed_dim),
                torch.nn.GELU(),
                torch.nn.Linear(llm_embed_dim, llm_embed_dim),
            )

        log.info(
            f"Image encoder {self.config.image_encoder.name} initialized with output dim {image_encoder_embed_dim} -> {llm_embed_dim}"
        )

        # Freeze all LLM model parameters
        if config.freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False

        # Unfreeze projector parameters for fine-tuning
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
            self.detr_integration = DETRIntegration(
                config=self.config,
                # llm_hidden_size=self.vocab_size when using lm_head and not last hidden states
                llm_hidden_size=self.model.config.hidden_size,
                image_encoder_hidden_size=self.image_encoder.get_output_dim(),
                batch_size=self.config.batch_size,
                device=self.config.device,
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

        destination = destination or {}

        self.projector.state_dict(
            destination=destination,
            prefix=prefix + "projector.",
            keep_vars=keep_vars,
        )
        self.model.get_input_embeddings().state_dict(
            destination=destination,
            prefix=prefix + "input_embeddings.",
            keep_vars=keep_vars,
        )
        self.model.get_output_embeddings().state_dict(
            destination=destination,
            prefix=prefix + "output_embeddings.",
            keep_vars=keep_vars,
        )

        # Add full model state dict if not frozen
        # TODO: maybe change to always return full if training in stages
        if not self.config.freeze_model:
            # add model state dict to the output
            self.model.state_dict(
                destination=destination,
                prefix=prefix + "llm.",
                keep_vars=keep_vars,
            )

        if self.config.detr_loss:
            self.detr_integration.state_dict(
                destination=destination,
                prefix=prefix + "detr_integration.",
                keep_vars=keep_vars,
            )

        return destination

    def load_state_dict(self, state_dict, strict=True):
        log.info("Loading state dict")
        missing_keys = []
        unexpected_keys = []

        is_torch_compile = next(iter(state_dict.keys())).startswith("_orig_mod.")
        if is_torch_compile:
            assert all(
                [k.startswith("_orig_mod.") for k in state_dict.keys()]
            ), "State dict should be from torch compile"
            state_dict = {k[len("_orig_mod.") :]: v for k, v in state_dict.items()}

        components = {
            "projector": self.projector,
            "input_embeddings": self.model.get_input_embeddings(),
            "output_embeddings": self.model.get_output_embeddings(),
        }

        if any(k.startswith("llm.") for k in state_dict.keys()):
            # log.info("Loading llm component to dict")
            components["llm"] = self.model

        if hasattr(self, "detr_integration") and any(
            k.startswith("detr_integration.") for k in state_dict.keys()
        ):
            # log.info("Loading detr_integration component to dict")
            components["detr_integration"] = self.detr_integration

        # Load each component
        for prefix, component in components.items():
            # log.info(f"Loading {prefix} component")
            component_dict = {
                k[len(f"{prefix}.") :]: v
                for k, v in state_dict.items()
                if k.startswith(f"{prefix}.")
            }
            if component_dict:
                _strict = strict and prefix != "llm"
                m, u = component.load_state_dict(component_dict, strict=_strict)
                missing_keys.extend(m)
                unexpected_keys.extend(u)

            # print(missing_keys, unexpected_keys)

        return missing_keys, unexpected_keys

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
        # pixel_values format: [batch_size, channels, height, width]

        # Image feature extraction
        pixel_values = pixel_values.to(self.config.device)

        if pixel_values.ndim == 5:  # If patches are used
            image_features = []
            for img in pixel_values:
                image_features.append(self.image_encoder.encode(img, None))
            # convert list to tensor
            image_features = torch.stack(image_features)
        else:
            image_features = self.image_encoder.encode(pixel_values, image_sizes)

        image_features = image_features.to(self.model.device, self.model.dtype)
        # Project image features to token size
        image_features_proj = self.projector(image_features)

        # shape of image_features is (batch_size, (num_patches), num_image_tokens, token_size_text_encoder)
        return image_features, image_features_proj

    def _use_detr_head(self, input_ids, outputs, image_features):
        return self.detr_integration.forward(
            input_ids=input_ids,
            outputs=outputs,
            image_features=image_features,
            query_tokens_id=self.query_tokens_id,
            num_query_tokens=self.config.num_query_tokens,
        )

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
                    config=self.detr_integration.detr_config,
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
