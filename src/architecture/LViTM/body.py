import torch
import torch.nn as nn
from attention import MultiHeadAttention


class LViTMBlock(nn.Module):
    def __init__(
            self, 
            embed_dim, 
            num_heads, 
            mlp_dim
    ):
        super().__init__()
        
        ###########################
        #   Attention Mechanism   #
        ###########################

        self.attn = MultiHeadAttention(embed_dim, num_heads)

        ###########
        #   MLP   #
        ###########
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim), # increase dimensionality
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim) # return to embedding dimension
        )

        #################
        #   Normalize   #
        #################

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
            self, 
            x,  # (B, L, D)
            key_padding_mask=None  # (B, L)
    ):
        # Apply attention to normalized input
        x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
        # Normalize and run through MLP
        x = x + self.mlp(self.norm2(x))
        return x
    

class LargeVisionTransformerModel(nn.Module):
    def __init__(
            self, 
            embed_dim,
            num_heads,
            mlp_dim,
            depth,
            num_proposals,
            z_dim
    ):
        super().__init__()

        #################
        #   Variables   #
        #################

        self.embed_dim = embed_dim
        self.num_proposals = num_proposals
        self.z_dim = z_dim

        #######################
        #   Proposal Tokens   #
        #######################

        self.proposal_tokens = nn.Parameter(
            torch.randn(1, num_proposals, embed_dim)
        )

        # No positional encoding
        self.pos_encoding = None

        ##########################
        #   Transformer Blocks   #
        ##########################

        self.blocks = nn.ModuleList(
            [LViTMBlock(embed_dim, num_heads, mlp_dim) for _ in range(depth)]
        )

        ##################################
        #   Project to Latent Proposal   #
        ##################################

        self.proposal_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, z_dim)
        )

    def forward(
            self, 
            C,
            test_tokens,
            key_padding_mask=None
    ):
        ###############################
        #   B = num batches           #
        #   S = sequence length       #
        #   D = embedding dimension   #
        #   T = num proposal tokens   #
        ###############################

        B, S, D = test_tokens.shape
        T = self.num_proposals

        ##############################
        #   Prepare Special Tokens   #
        ##############################
        
        # Expand context to a single token
        C_token = C.unsqueeze(1)

        # Place proposal tokens in each batch
        proposal_tok = self.proposal_tokens.expand(B, self.num_proposals, self.embed_dim)

        ###################
        #   Build Input   #
        ###################

        x = torch.cat([C_token, proposal_tok, test_tokens], dim=1)  # (B, 1+T+S, D)

        ##################
        #   Build Mask   #
        ##################

        if key_padding_mask is not None:
            c_pad = torch.zeros(B, 1, dtype=torch.bool, device=key_padding_mask.device)
            p_pad = torch.zeros(T, 1, dtype=torch.bool, device=key_padding_mask.device)
            full_mask = torch.cat([c_pad, p_pad, key_padding_mask], dim=1)  # (B, 1+T+S)
        else:
            full_mask = None
        
        # Optional positional encoding here

        ####################
        #   LViTM Blocks   #
        ####################

        for block in self.blocks:
            x = block(x, key_padding_mask=full_mask)
        
        ##########################
        #   Retrieve Proposals   #
        ##########################

        proposal_outs = x[:, 1:1+T, :]  # (B, 1+T+S, D) -> (B, T, D)

        ###############################
        #   Project to Latent Space   #
        ###############################

        Z = self.proposal_head(proposal_outs)

        return Z
    

"""
Inference-time sequential reasoning loop:

grid = I_test_clone  # current grid state

for step in range(T_steps):
    # 1. Re-encode current grid with conditional encoder
    test_tokens, key_padding_mask = cond_encoder(grid, C, mask_grid)

    # 2. Get proposals for this state
    Z = lvittm(C, test_tokens, key_padding_mask)  # (B, T, z_dim)

    # 3. Choose a z (e.g. best by critic, or z_0)
    z = select_proposal(Z, grid, critic, examples, C)  # shape (B, z_dim)

    # 4. Apply it with executor
    grid = executor(grid, z)  # new grid state

# final grid is your prediction
O_final = grid

"""