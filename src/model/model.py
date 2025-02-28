import logging

import torch
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
from transformers import AutoTokenizer


from model.loss import masked_cross_entropy

log = logging.getLogger(__name__)


class VisionLanguageModel(torch.nn.Module):
    def __init__(self, model_name):
        super(VisionLanguageModel, self).__init__()

        self.model_name = model_name

        if torch.cuda.is_available():
            attn_implementation = "flash_attention_2"
            torch_dtype = torch.bfloat16
            #device_map = "auto"
        else:
            attn_implementation = torch_dtype = device_map = None

        # Get model components
        # TODO: device_map="auto", currently give warning, could be ignored https://github.com/huggingface/transformers/issues/31544
        self.model = LlavaQwenForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, attn_implementation=attn_implementation
        )
        self.image_encoder = self.model.get_vision_tower()
        self.projector = self.model.get_model().mm_projector

        # self._tokenizer = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.image_token = "<image>"

        self.image_size = self.image_encoder.config.image_size
        # Get number of tokens of image encoder
        self.num_img_tokens = self.image_encoder.vision_tower.vision_model.embeddings.position_embedding.weight.shape[
            0
        ]

        log.info("Model initialized")

    # @property # create error
    # def tokenizer(self):
    #     if self._tokenizer is None:
    #         from transformers import AutoTokenizer

    #         self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    #         self.model.resize_token_embeddings(len(self._tokenizer))
    #     return self._tokenizer

    def forward(self, input_ids, attention_mask, images, labels=None):
        # Image feature extraction
        if images.ndim == 5:  # If patches are used
            image_features = []
            for img in images:
                image_features.append(self.image_encoder(img))
            # convert list to tensor
            image_features = torch.stack(image_features)
        else:
            image_features = self.image_encoder(images)
        image_features = image_features.to(self.model.device, self.model.dtype)

        image_features = self.projector(image_features)
        # shape of image_features is (batch_size, (num_patches), num_image_tokens, token_size_text_encoder)

        # Token embeddings
        embedding_layer = self.model.get_input_embeddings()
        inputs_embeds = embedding_layer(input_ids)

        # Integrate image features into token embeddings
        image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)

        n_image_tokens = (input_ids == image_token_id).sum().item()
        n_image_features = image_features.shape[0] * image_features.shape[1]
        if image_features.ndim == 4:  # if patches are used
            n_image_features *= image_features.shape[2]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        special_image_mask = (input_ids == image_token_id).unsqueeze(-1)
        special_image_mask = special_image_mask.expand_as(inputs_embeds).to(
            inputs_embeds.device
        )
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # LLM forward pass
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        if labels is None:
            return outputs

        loss = masked_cross_entropy(
            outputs.logits, labels, vocab_size=self.model.vocab_size
        )
        outputs.loss = loss
        return outputs

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
        image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        special_image_mask = (input_ids == image_token_id).unsqueeze(-1)
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
    model = VisionLanguageModel(model_name)
    # get one image and input into forward pass
    image = torch.rand(1, 3, 384, 384)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
    output = model(input_ids, attention_mask, image)
    log.info(output.size())
