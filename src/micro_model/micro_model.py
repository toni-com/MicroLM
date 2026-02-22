import torch
from torch import nn



class FeedFoward(nn.Module):
    def __init__(self, n_embed, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed, n_head, block_size, dropout=0.2):
        super().__init__()
        self.sa = nn.MultiheadAttention(n_embed, n_head, dropout=dropout, batch_first=True)
        self.register_buffer('causal_mask', torch.ones(block_size, block_size, dtype=torch.bool).tril())
        self.ffwd = FeedFoward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        B, T, C = x.shape
        x_norm = self.ln1(x)
        # invert the causal mask
        attn_mask = ~self.causal_mask[:T, :T]
        attn_out, _ = self.sa(x_norm, x_norm, x_norm, attn_mask=attn_mask, is_causal=True)
        x = x + attn_out
        x = x + self.ffwd(self.ln2(x))
        return x

class MicroModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dims: int,
        block_size: int,
        hidden_dims: int,
        n_layer: int = 4,
        n_head: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dims = embed_dims
        self.block_size = block_size
        
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dims)
        self.position_embedding_table = nn.Embedding(block_size, embed_dims)
        self.blocks = nn.Sequential(*[Block(embed_dims, n_head=n_head, block_size=block_size, dropout=dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(embed_dims)
        self.lm_head = nn.Linear(embed_dims, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> torch.Tensor:
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits
