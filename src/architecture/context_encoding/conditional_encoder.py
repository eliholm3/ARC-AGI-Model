import torch
import torch.nn as nn

class ConditionalTestInputEncoder(nn.Module):

    def __init__(
            self, 
            vit_test: nn.Module
    ):
        ###############################
        #   B = batch size            #    
        #   D = token embedding dim   #
        #   S = num tokens            #
        #   H = height                #
        #   W = width                 #
        ###############################

        super().__init__()
        self.vit = vit_test
        self.embed_dim = self.vit.c_token.size(-1)

        # Project context vector to embedding dim
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
            self, 
            I_test,  # (B, 1, H, W)
            mask_test,  # (B, H, W) or None
            C  # (B, D)
    ):

        B, _, H, W = I_test.shape

        ######################
        #   Encode with ViT  #
        ######################

        tokens = self.vit.patch_embedding(I_test)  # (B, S, D)

        #######################
        #   Build Test Mask   #
        #######################

        key_padding_mask = None
        if mask_test is not None:
            # Ensure batch dimension exists
            if mask_test.dim() == 2:  # (H, W) -> (1, H, W)
                mask_test = mask_test.unsqueeze(0)

            # mask_test: (B, H, W), True = valid
            flat_mask = mask_test.reshape(B, -1)      # (B, S)
            key_padding_mask = ~flat_mask.to(torch.bool)  # True = pad

        ##########################
        #   Add Context Vector   #
        ##########################

        C_token = self.c_proj(C).unsqueeze(1)  # (B,1,D)
        tokens = torch.cat([C_token, tokens], dim=1)  # (B,1+S,D)

        if key_padding_mask is not None:
            # Add context to mask
            c_pad = torch.zeros(B, 1, dtype=torch.bool, device=key_padding_mask.device)
            key_padding_mask = torch.cat([c_pad, key_padding_mask], dim=1)  # (B,1+S)

        #####################################
        #   Positional Encoding + Dropout   #
        #####################################

        tokens = self.vit.pos_encoding(tokens)
        tokens = self.vit.dropout(tokens)

        # print("\n[CondEncoder] tokens shape:", tokens.shape)
        # if key_padding_mask is not None:
        #     print("[CondEncoder] key_padding_mask shape:", key_padding_mask.shape)


        return tokens, key_padding_mask