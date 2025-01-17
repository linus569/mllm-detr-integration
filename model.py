import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    LlavaForConditionalGeneration,
)
from processor import processor

# load pre-trained model
model_name = "lmms-lab/llava-onevision-qwen2-0.5b-si"


class VisionLanguageModel(torch.nn.Module):
    def __init__(self, model_name):
        super(VisionLanguageModel, self).__init__()
        self.model = AutoModelForImageTextToText.from_pretrained(model_name)
        self.tokenizer = processor.tokenizer
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.image_encoder = self.model.vision_tower
        self.text_encoder = self.model.get_decoder()
        self.projector = torch.nn.Linear(
            self.image_encoder.config.hidden_size, self.text_encoder.config.hidden_size
        )

    def forward(self, input_ids, attention_mask, images):
        # Image feature extraction
        image_features = self.image_encoder(images).last_hidden_state
        projected_features = self.projector(image_features)

        # Token embeddings
        # token_embeddings = self.text_encoder.embeddings(input_ids)
        embedding_layer = self.text_encoder.get_input_embeddings()
        token_embeddings = embedding_layer(input_ids)

        # Integrate image features into token embeddings
        image_token_mask = input_ids == self.tokenizer.convert_tokens_to_ids(
            processor.image_token
        )
        token_embeddings[image_token_mask] = projected_features.view(
            -1, token_embeddings.size(-1)
        )

        # LLM forward pass
        outputs = self.text_encoder(
            inputs_embeds=token_embeddings, attention_mask=attention_mask
        )
        return outputs
