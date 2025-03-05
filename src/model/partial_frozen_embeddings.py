import torch
from torch import nn


class PartiallyFrozenEmbedding(nn.Module):
    def __init__(
        self,
        frozen_embedding: nn.Embedding,
        new_tokens: int,
        initializers: list[list[int]],
        do_init: bool = True,
    ):
        super().__init__()
        self.frozen_embedding = frozen_embedding
        self.trainable_embedding = nn.Embedding(
            new_tokens, frozen_embedding.embedding_dim
        )
        self.num_frozen = frozen_embedding.num_embeddings
        self.num_trainable = new_tokens
        self.num_embeddings = self.num_frozen + self.num_trainable

        self.embedding_dim = frozen_embedding.embedding_dim

        if do_init:
            self._init_trainable_embedding(initializers)

        for param in self.frozen_embedding.parameters():
            param.requires_grad = False

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return self.trainable_embedding.state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )

    def load_state_dict(self, state_dict, strict=True):
        return self.trainable_embedding.load_state_dict(state_dict, strict)

    def _init_trainable_embedding(self, initializers: list[list[int]]):
        assert (
            len(initializers) == self.num_trainable
        ), "Initializers should be provided for all new tokens"
        old_embeddings_weight = self.frozen_embedding.weight.data.to(torch.float32)

        new_embeddings_weight = _init_new_embeddings(
            old_embeddings_weight, initializers, add_1std_noise_for_empty_init=True
        )
        self.trainable_embedding.weight.data = new_embeddings_weight

    def forward(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        N, L = input_ids.shape
        frozen_mask = input_ids < self.num_frozen

        # frozen_input_ids = input_ids[frozen_mask]
        # frozen_embed = self.frozen_embedding(frozen_input_ids)
        # trainable_input_ids = input_ids[~frozen_mask] - self.num_frozen
        # trainable_embed = self.trainable_embedding(trainable_input_ids)
        # embeddings = frozen_embed.new_zeros(N, L, self.embedding_dim)
        # embeddings[frozen_mask, :] = frozen_embed
        # embeddings[~frozen_mask, :] = trainable_embed

        frozen_input_ids = input_ids.clamp(max=self.num_frozen - 1)
        trainable_input_ids = input_ids.clamp(min=self.num_frozen) - self.num_frozen
        frozen_embed = self.frozen_embedding(frozen_input_ids)
        trainable_embed = self.trainable_embedding(trainable_input_ids)
        embeddings = torch.where(frozen_mask.unsqueeze(-1), frozen_embed, trainable_embed)

        #assert torch.allclose(embeddings, embeddings2)

        return embeddings


class PartiallyFrozenLMHead(nn.Module):
    def __init__(
        self,
        frozen_lm_head: nn.Linear,
        new_tokens: int,
        initializers: list[list[int]],
        do_init: bool = True,
    ):
        super().__init__()

        self.frozen_lm_head = frozen_lm_head
        self.trainable_lm_head = nn.Linear(
            frozen_lm_head.in_features, new_tokens, bias=frozen_lm_head.bias is not None
        )
        self.num_frozen = frozen_lm_head.out_features
        self.num_trainable = new_tokens
        self.in_features = frozen_lm_head.in_features
        self.out_features = self.num_frozen + self.num_trainable

        if do_init:
            self._init_trainable_lm_head(initializers)

        for param in self.frozen_lm_head.parameters():
            param.requires_grad = False

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return self.trainable_lm_head.state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )

    def load_state_dict(self, state_dict, strict=True):
        return self.trainable_lm_head.load_state_dict(state_dict, strict)

    def _init_trainable_lm_head(self, initializers: list[list[int]]):
        assert (
            len(initializers) == self.num_trainable
        ), "Initializers should be provided for all new tokens"

        old_lm_head_weight = self.frozen_lm_head.weight.data.to(torch.float32)
        new_lm_head_weight = _init_new_embeddings(
            old_lm_head_weight, initializers, add_1std_noise_for_empty_init=True
        )
        self.trainable_lm_head.weight.data = new_lm_head_weight

        if self.frozen_lm_head.bias is not None:
            old_lm_head_bias = self.frozen_lm_head.bias.data.to(torch.float32)
            new_lm_head_bias = _init_new_embeddings(
                old_lm_head_bias, initializers, add_1std_noise_for_empty_init=True
            )
            self.trainable_lm_head.bias.data = new_lm_head_bias

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        frozen_logits = self.frozen_lm_head(hidden_states)
        trainable_logits = self.trainable_lm_head(hidden_states)

        logits = torch.cat([frozen_logits, trainable_logits], dim=-1)
        return logits


def _init_new_embeddings(
    old_embeddings,
    initializers: list[list[int]],
    add_1std_noise_for_empty_init: bool = True,
):
    mean_embedding = old_embeddings.mean(dim=0)
    std_embedding = old_embeddings.std(dim=0)

    new_embd = []
    for init in initializers:
        if len(init) == 0:
            if add_1std_noise_for_empty_init:
                new_embd.append(torch.normal(mean_embedding, std_embedding))
            else:
                new_embd.append(mean_embedding)
        else:
            # init is a list of indices of old embeddings that should be averaged for new embedding
            new_embd.append(old_embeddings[init].mean(dim=0))

    return torch.stack(new_embd, dim=0)


if __name__ == "__main__":
    # Example of usage
    frozen_embedding = nn.Embedding(100, 128)
    new_tokens = 10
    initializers = [[], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
    do_init = True

    partial_frozen_embedding = PartiallyFrozenEmbedding(
        frozen_embedding, new_tokens, initializers, do_init
    )
    input_ids = torch.randint(0, 110, (32, 128))
    embeddings = partial_frozen_embedding(input_ids)
    print(embeddings.shape)

    frozen_lm_head = nn.Linear(128, 100)
    new_tokens = 10
    initializers = [[], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
    do_init = True

    partial_frozen_lm_head = PartiallyFrozenLMHead(
        frozen_lm_head, new_tokens, initializers, do_init
    )
    hidden_states = torch.randn(32, 128, 128)
    logits = partial_frozen_lm_head(hidden_states)
    print(logits.shape)
