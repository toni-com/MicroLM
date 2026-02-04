import torch
from torch import nn


class MicroModel(nn.Module):
    def __init__(
        self, vocab_size: int, embed_dims: int, block_size: int, hidden_dims: int
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dims = embed_dims
        self.block_size = block_size
        self.hidden_dims = hidden_dims

        # token embedding
        self.C = torch.nn.Embedding(embedding_dim=embed_dims, num_embeddings=vocab_size)

        # hidden layer
        dims = embed_dims * block_size
        self.layer1 = torch.nn.Linear(in_features=dims, out_features=hidden_dims)
        self.layer2 = torch.nn.Linear(in_features=hidden_dims, out_features=vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # embed and flatten
        x = self.C(x)
        x = x.flatten(start_dim=1)

        # layer1 + activation
        x = self.layer1(x)
        x = torch.nn.functional.relu(x)

        # output
        x = self.layer2(x)
        return x
