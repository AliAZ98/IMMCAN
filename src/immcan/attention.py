"""Original IMMCAN attention blocks, reorganized as a package module.

The idiom-to-multimodal stage attends to ONE pooled token, not to a
sequence of image patches; this is a documented methodological limitation.
"""

import torch
import torch.nn as nn

class _TextToImageBlock(nn.Module):
    """
    Single cross-attention block:
    - Text attends to image (Q = text, K/V = image)
    """

    def __init__(self, cross_dim=512, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cross_dim,
            num_heads=num_heads,
            batch_first=True,  # (B, L, C)
        )
        self.ffn = nn.Sequential(
            nn.Linear(cross_dim, cross_dim * 4),
            nn.ReLU(),
            nn.Linear(cross_dim * 4, cross_dim),
        )
        self.norm1 = nn.LayerNorm(cross_dim)
        self.norm2 = nn.LayerNorm(cross_dim)

    def forward(self, T, I, image_mask=None):
        """
        T: [B, T, C] (text)
        I: [B, P, C] (image)
        image_mask: [B, P] with 1=valid, 0=pad or None
        """
        key_padding_mask = None
        if image_mask is not None:
            key_padding_mask = (image_mask == 0)  # True = ignore

        # Cross-attention: text queries, image keys/values
        attn_out, _ = self.cross_attn(
            query=T,         # [B, T, C]
            key=I,           # [B, P, C]
            value=I,         # [B, P, C]
            key_padding_mask=key_padding_mask,
        )  # [B, T, C]

        x = self.norm1(T + attn_out)
        f = self.ffn(x)
        out = self.norm2(x + f)  # [B, T, C]

        return out
        

class TextImageCrossAttention(nn.Module):
    """
    Option A: text -> image cross-attention, stackable.

    - Project text/image once to cross_dim.
    - Pass through `num_layers` cross-attention blocks.
    - Only text stream is updated. Image stays as context.
    """

    def __init__(self, token_dim=1024, cross_dim=512, num_heads=8, num_layers=1):
        super().__init__()
        self.text_proj = nn.Linear(token_dim, cross_dim)
        self.image_proj = nn.Linear(token_dim, cross_dim)

        self.layers = nn.ModuleList([
            _TextToImageBlock(cross_dim=cross_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, text_tokens, image_tokens, text_mask=None, image_mask=None):
        """
        text_tokens:  [B, T, D]
        image_tokens: [B, P, D]
        text_mask:    [B, T] (currently unused, but kept for API symmetry)
        image_mask:   [B, P] with 1=valid, 0=pad or None
        """
        # Project to common dim (once)
        T = self.text_proj(text_tokens)     # [B, T, C]
        I = self.image_proj(image_tokens)   # [B, P, C]

        # Run through stacked cross-attn blocks
        for layer in self.layers:
            T = layer(T, I, image_mask=image_mask)   # [B, T, C]

        # Return fused text tokens
        return T   # [B, T, C]


class _CoAttentionBlock(nn.Module):
    """
    Single co-attention block:
    - Text attends to image (T -> I)
    - Image attends to updated text (I -> T_out)
    """

    def __init__(self, cross_dim=512, num_heads=8):
        super().__init__()

        self.attn_t2i = nn.MultiheadAttention(
            embed_dim=cross_dim,
            num_heads=num_heads,
            batch_first=True,   # (B, L, C)
        )
        self.attn_i2t = nn.MultiheadAttention(
            embed_dim=cross_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.ffn_text = nn.Sequential(
            nn.Linear(cross_dim, cross_dim * 4),
            nn.ReLU(),
            nn.Linear(cross_dim * 4, cross_dim),
        )
        self.ffn_image = nn.Sequential(
            nn.Linear(cross_dim, cross_dim * 4),
            nn.ReLU(),
            nn.Linear(cross_dim * 4, cross_dim),
        )

        self.norm_t1 = nn.LayerNorm(cross_dim)
        self.norm_t2 = nn.LayerNorm(cross_dim)
        self.norm_i1 = nn.LayerNorm(cross_dim)
        self.norm_i2 = nn.LayerNorm(cross_dim)

    def forward(self, T, I, text_mask=None, image_mask=None):
        """
        T: [B, T, C] (text)
        I: [B, P, C] (image)
        text_mask:  [B, T] with 1=valid, 0=pad or None
        image_mask: [B, P] with 1=valid, 0=pad or None
        """
        # Build key padding masks: True = ignore
        img_kpm = None
        txt_kpm = None
        if image_mask is not None:
            img_kpm = (image_mask == 0)  # [B, P]
        if text_mask is not None:
            txt_kpm = (text_mask == 0)   # [B, T]

        # ----- Text -> Image attention -----
        t2i_out, _ = self.attn_t2i(
            query=T,           # [B, T, C]
            key=I,             # [B, P, C]
            value=I,           # [B, P, C]
            key_padding_mask=img_kpm,
        )  # [B, T, C]

        T_res = self.norm_t1(T + t2i_out)
        T_ffn = self.ffn_text(T_res)
        T_out = self.norm_t2(T_res + T_ffn)   # [B, T, C]

        # ----- Image -> Text attention -----
        i2t_out, _ = self.attn_i2t(
            query=I,           # [B, P, C]
            key=T_out,         # [B, T, C]
            value=T_out,       # [B, T, C]
            key_padding_mask=txt_kpm,
        )  # [B, P, C]

        I_res = self.norm_i1(I + i2t_out)
        I_ffn = self.ffn_image(I_res)
        I_out = self.norm_i2(I_res + I_ffn)   # [B, P, C]

        return T_out, I_out


class TextImageCoAttention(nn.Module):
    """
    Option B: bidirectional co-attention, stackable.

    - Project text/image once to cross_dim.
    - Run `num_layers` co-attention blocks.
    """

    def __init__(self, token_dim=1024, cross_dim=512, num_heads=8, num_layers=1):
        super().__init__()
        self.text_proj = nn.Linear(token_dim, cross_dim)
        self.image_proj = nn.Linear(token_dim, cross_dim)

        self.layers = nn.ModuleList([
            _CoAttentionBlock(cross_dim=cross_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, text_tokens, image_tokens, text_mask=None, image_mask=None):
        """
        text_tokens:  [B, T, D]
        image_tokens: [B, P, D]
        text_mask:    [B, T] with 1=valid, 0=pad or None
        image_mask:   [B, P] with 1=valid, 0=pad or None
        """
        # Project once
        T = self.text_proj(text_tokens)     # [B, T, C]
        I = self.image_proj(image_tokens)   # [B, P, C]

        # Pass through stacked co-attention blocks
        for layer in self.layers:
            T, I = layer(T, I, text_mask=text_mask, image_mask=image_mask)

        return T, I   # fused_text, fused_image

class _IdiomToMMBlock(nn.Module):
    """
    Single block:
      - Idiom tokens attend to global multimodal token (Q=idiom, K/V=mm)
    """

    def __init__(self, cross_dim=512, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cross_dim,
            num_heads=num_heads,
            batch_first=True,  # [B, L, C]
        )
        self.ffn = nn.Sequential(
            nn.Linear(cross_dim, cross_dim * 4),
            nn.ReLU(),
            nn.Linear(cross_dim * 4, cross_dim),
        )
        self.norm1 = nn.LayerNorm(cross_dim)
        self.norm2 = nn.LayerNorm(cross_dim)

    def forward(self, I, MM, idiom_mask=None):
        """
        I : [B, T_idiom, C]  (idiom tokens, projected)
        MM: [B, 1, C]        (multimodal global token, projected)
        idiom_mask: [B, T_idiom] with 1=valid, 0=pad or None
        """
        # key_padding_mask on MM is usually None (single token per sample)

        attn_out, _ = self.cross_attn(
            query=I,          # [B, T_idiom, C]
            key=MM,           # [B, 1, C]
            value=MM,         # [B, 1, C]
            key_padding_mask=None,
        )  # [B, T_idiom, C]

        x = self.norm1(I + attn_out)
        f = self.ffn(x)
        out = self.norm2(x + f)
        return out


class IdiomMMCrossAttention(nn.Module):
    """
    Idiom tokens attend to concatenated multimodal global embedding.

    Inputs:
        idiom_tokens: [B, T_idiom, D_idiom]
        mm_global   : [B, 1024]   (concat of global_text[512] + global_image[512])

    Output:
        fused_idiom : [B, T_idiom, cross_dim]
    """

    def __init__(self, idiom_dim, mm_dim=1024, cross_dim=512, num_heads=8, num_layers=1):
        super().__init__()
        self.idiom_proj = nn.Linear(idiom_dim, cross_dim)
        self.mm_proj = nn.Linear(mm_dim, cross_dim)

        self.layers = nn.ModuleList([
            _IdiomToMMBlock(cross_dim=cross_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, idiom_tokens, mm_global, idiom_mask=None):
        """
        idiom_tokens: [B, T_idiom, D_idiom]
        mm_global   : [B, 1024]
        idiom_mask  : [B, T_idiom] or None
        """
        # project
        I = self.idiom_proj(idiom_tokens)      # [B, T_idiom, C]
        MM = self.mm_proj(mm_global)          # [B, C]
        MM = MM.unsqueeze(1)                  # [B, 1, C]

        # stacked layers
        for layer in self.layers:
            I = layer(I, MM, idiom_mask=idiom_mask)

        return I   # [B, T_idiom, C]
