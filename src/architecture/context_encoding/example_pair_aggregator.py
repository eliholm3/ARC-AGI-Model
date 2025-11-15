import torch
import torch.nn as nn
import torch.nn.functional as F

class ExamplePairAggregator(nn.Module):
    """
    Aggregates the context vectors from k example pairs (I_i, O_i) into a single embedding
    """

    def __init__(
            self, 
            embed_dim: int,
            hidden_dim: int = 256
    ):
        super().__init__()

        # MLP to weigh context vectors
        self.score_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(), # [-1,1]
            nn.Linear(hidden_dim, 1)
        )

        # Normalize context vector
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
            self,
            h: torch.Tensor,  # all embeddings 
            mask: torch.Tensor | None = None  # where the tokens are valid
    ) -> torch.Tensor:
        ###############################
        #   B = batch size            #
        #   K = num example pairs     #
        #   D = embedding dimension   #
        ###############################

        B, K, D = h.shape

        # Infer scores
        scores = self.score_mlp(h)

        # Mask = where the tokens are
        if mask is not None:

            # Ensure boolean
            mask = mask.to(dtype=torch.bool)

            # Match to score shape
            mask_expanded = mask.unsqueeze(-1) # (B, K, 1)

            # Set where the mask is not to very negative
            scores = scores.masked_fill(~mask_expanded, float("-inf"))

        # Attention weights over k example pairs
        attn = F.softmax(scores, dim=1)

        # Weighted sum
        C = torch.sum(attn * h, dim=1)

        # Final normalization
        C = self.norm(C)

        return C