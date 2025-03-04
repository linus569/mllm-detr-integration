import logging
from typing import List, Optional, Tuple, Union

import torch
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from model.loss import masked_cross_entropy
from model.partial_frozen_embeddings import (
    PartiallyFrozenEmbedding,
    PartiallyFrozenLMHead,
)

log = logging.getLogger(__name__)


class VisionLanguageModel(torch.nn.Module):
    def __init__(
        self,
        config,
        image_token_index: int,
        num_new_tokens: int = 0,
        initializers: list[list[int]] = None,
        do_init: bool = True,
    ):
        super(VisionLanguageModel, self).__init__()

        if do_init:
            assert (
                initializers is not None
            ), "Initializers should be provided for new tokens"

        self.config = config

        # Get model components
        # device_map="auto", prints warning, could be ignored https://github.com/huggingface/transformers/issues/31544
        # torch_dtype=self.torch_dtype, attn_implementation=attn_implementation,
        self.model = LlavaQwenForCausalLM.from_pretrained(self.config.model_name)
        self.image_encoder = self.model.get_vision_tower()
        self.projector = self.model.get_model().mm_projector

        # freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # check if image_encoder params are frozen
        for param in self.image_encoder.parameters():
            assert not param.requires_grad, "Image encoder parameters should be frozen"

        # unfreeze projector parameters for fine-tuning
        for param in self.projector.parameters():
            param.requires_grad = True

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

        self.vocab_size = self.model.get_input_embeddings().num_embeddings
        self.image_token_index = image_token_index

        log.info("Model initialized")

    def _get_image_features(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        # Image feature extraction
        if pixel_values.ndim == 5:  # If patches are used
            image_features = []
            for img in pixel_values:
                image_features.append(self.image_encoder(img))
            # convert list to tensor
            image_features = torch.stack(image_features)
        else:
            image_features = self.image_encoder(pixel_values)
        # TODO: check in detail again if image_features are coorect or if I need to do something with them
        image_features = image_features.to(self.model.device, self.model.dtype)

        # Project image features to token size
        image_features = self.projector(image_features)
        # shape of image_features is (batch_size, (num_patches), num_image_tokens, token_size_text_encoder)

        return image_features

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
            image_features = self._get_image_features(images)
            # Integrate image features into token embeddings
            n_image_tokens = (input_ids == self.image_token_index).sum()
            n_image_features = image_features.shape[0] * image_features.shape[1]
            if image_features.ndim == 4:  # if patches are used
                n_image_features *= image_features.shape[2]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            # special_image_mask shape is (batch_size, seq_len, token_size_text_encoder)
            # inputs_embeds shape is (batch_size, seq_len, token_size_text_encoder)
            # image_features shape is (batch_size, num_image_tokens, token_size_text_encoder)
            special_image_mask = (input_ids == self.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(
                inputs_embeds.device
            )
            image_features = image_features.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask, image_features
            )

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
            loss = masked_cross_entropy(
                outputs.logits,
                labels,
                vocab_size=self.vocab_size,  # TODO: Check if change needd when increasing vocab size
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            # TODO: return also image_hidden states?
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask,
        image,
        max_new_tokens=2048,
        stopping_criteria=None,
        **kwargs,
    ):
        assert image.dim() in (
            4,
            5,
        ), "Image should be of shape [batch_size, channels, height, width] or [batch_size, num_patches, channels, height, width]"
        # assert image.dtype == torch.float32, "Image should be of type float32" # FIXME: not true, can be bfloat16

        # Image feature extraction
        if image.dim() == 5:  # Image with patches
            image_features = []
            for img in image:
                # Ensure correct channel order for each patch
                if img.shape[-1] == 3:  # If channels are last
                    img = img.permute(0, 3, 1, 2)
                image_features.append(self.image_encoder(img))
            image_features = torch.stack(image_features)
        else:  # Handle regular images
            # Ensure image is in correct format [batch_size, channels, height, width]
            if image.shape[-1] == 3:  # If channels are last
                image = image.permute(0, 3, 1, 2)
            image_features = self.image_encoder(image)

        image_features = self.projector(image_features)

        # Token embeddings
        embedding_layer = self.model.get_input_embeddings()
        inputs_embeds = embedding_layer(input_ids)

        # Integrate image features into embeddings
        special_image_mask = (input_ids == self.image_token_index).unsqueeze(-1)
        special_image_mask = special_image_mask.expand_as(inputs_embeds)
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # Generate text
        # outputs = self.text_encoder.generate(
        #     inputs_embeds=inputs_embeds,
        #     attention_mask=attention_mask,
        #     max_new_tokens=max_new_tokens,
        #     num_beams=num_beams,
        #     temperature=temperature,
        #     do_sample=do_sample,
        # )

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
