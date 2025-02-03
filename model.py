import torch
from transformers import (
    AutoTokenizer,
    Qwen2ForCausalLM,
    SiglipVisionModel,
)


class VisionLanguageModel(torch.nn.Module):
    def __init__(self, model_name):
        super(VisionLanguageModel, self).__init__()

        # Load model
        self.model = Qwen2ForCausalLM.from_pretrained(
            model_name,
            # torch_dtype=torch.float16, # fp16 for efficiency reasons during dev
            # low_cpu_mem_usage=True,
        )
        model_config = self.model.config
        self.vision_tower = None

        if hasattr(model_config, "mm_vision_tower"):
            vision_tower = getattr(model_config, "mm_vision_tower")
            self.vision_tower = SiglipVisionModel.from_pretrained(
                vision_tower,
                # torch_dtype=torch.float32
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.image_token = "<image>"
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Get model components
        self.image_encoder = self.vision_tower
        self.text_encoder = self.model
        self.projector = torch.nn.Linear(
            self.vision_tower.config.hidden_size, self.text_encoder.config.hidden_size
        )
        print("Model initialized")

    def forward(self, input_ids, attention_mask, images, labels=None):
        # Image feature extraction
        image_features = self.image_encoder(images)
        image_features = image_features.last_hidden_state
        image_features = self.projector(
            image_features
        )  # shape (batch_size, num_image_tokens, token_size_text_encoder)

        # Token embeddings
        embedding_layer = self.text_encoder.get_input_embeddings()
        inputs_embeds = embedding_layer(input_ids)

        # Integrate image features into token embeddings
        image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)

        n_image_tokens = (input_ids == image_token_id).sum().item()
        n_image_features = image_features.shape[0] * image_features.shape[1]
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
        outputs = self.text_encoder(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels
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
    print(output.size())
