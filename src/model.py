"""
model.py
--------
U-TAE (Spatio-Temporal U-Net with Lightweight Temporal Attention Encoder)
adapted for Next-Day Wildfire Spread Prediction with 17 Input Channels.

Architecture:
  - Input: (B, T=5, C=17, H=128, W=128)
  - Encoder: 4-stage Multi-scale Convolutional Pyramid ([64, 64, 64, 128])
  - Temporal Bottleneck: L-TAE with Multi-Head Self-Attention (16 heads, d_k=4)
  - Temporal Aggregator: Attention-weighted skip connection fusion
  - Decoder: 4-stage Convolutional Upsampling ([32, 32, 64, 128])
  - Output Head: [32, 1] convolutions -> Logits of shape (B, 1, 128, 128)
"""

import os
import sys
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

# Ensure UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Import UTAE from utae_backbone (or fallback to utae-paps repository if present)
try:
    from .utae_backbone.utae import UTAE
except ImportError:
    try:
        from src.utae_backbone.utae import UTAE
    except ImportError:
        # Fallback to utae-paps in parent directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        workspace_dir = os.path.dirname(parent_dir)
        utae_paps_path = os.path.join(workspace_dir, "utae-paps")
        if os.path.exists(utae_paps_path) and utae_paps_path not in sys.path:
            sys.path.insert(0, utae_paps_path)
        from src.backbones.utae import UTAE


class WildfireUTAE(nn.Module):
    """
    Wildfire Prediction Model based on U-TAE architecture.
    Takes 5-day sequences of 17 multi-modal raster channels and predicts next-day fire mask.
    """

    def __init__(
        self,
        input_dim: int = 17,
        encoder_widths: List[int] = [64, 64, 64, 128],
        decoder_widths: List[int] = [32, 32, 64, 128],
        out_conv: List[int] = [32, 1],
        str_conv_k: int = 4,
        str_conv_s: int = 2,
        str_conv_p: int = 1,
        agg_mode: str = "att_group",
        encoder_norm: str = "group",
        n_head: int = 16,
        d_model: int = 256,
        d_k: int = 4,
        pad_value: float = 0.0,
        padding_mode: str = "reflect",
    ):
        super(WildfireUTAE, self).__init__()
        self.input_dim = input_dim

        self.utae = UTAE(
            input_dim=input_dim,
            encoder_widths=encoder_widths,
            decoder_widths=decoder_widths,
            out_conv=out_conv,
            str_conv_k=str_conv_k,
            str_conv_s=str_conv_s,
            str_conv_p=str_conv_p,
            agg_mode=agg_mode,
            encoder_norm=encoder_norm,
            n_head=n_head,
            d_model=d_model,
            d_k=d_k,
            encoder=False,
            return_maps=False,
            pad_value=pad_value,
            padding_mode=padding_mode,
        )

    def forward(
        self,
        x: torch.Tensor,
        batch_positions: Optional[torch.Tensor] = None,
        return_att: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        Args:
            x: (B, T=5, C=17, H=128, W=128)
            batch_positions: Optional (B, T) step numbers. If None, uses [1, 2, 3, 4, 5].
            return_att: If True, returns (logits, attention_maps).
        Returns:
            logits: (B, 1, 128, 128)
            (optional) att: (n_head, B, T, h_bot, w_bot)
        """
        b, t = x.shape[0], x.shape[1]

        # Auto-generate temporal step positions if not provided
        if batch_positions is None:
            batch_positions = (
                torch.arange(1, t + 1, device=x.device, dtype=torch.long)
                .unsqueeze(0)
                .expand(b, -1)
            )

        if return_att:
            logits, att = self.utae(x, batch_positions=batch_positions, return_att=True)
            return logits, att
        else:
            logits = self.utae(x, batch_positions=batch_positions, return_att=False)
            return logits

    @torch.no_grad()
    def predict_proba(
        self,
        x: torch.Tensor,
        batch_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes sigmoid probability map in [0, 1]."""
        self.eval()
        logits = self.forward(x, batch_positions=batch_positions, return_att=False)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
        batch_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            probs: (B, 1, 128, 128) float in [0, 1]
            mask: (B, 1, 128, 128) float in {0.0, 1.0}
        """
        probs = self.predict_proba(x, batch_positions=batch_positions)
        mask = (probs >= threshold).float()
        return probs, mask

    @staticmethod
    def get_attention_weights(att: torch.Tensor) -> torch.Tensor:
        """
        Averages attention maps across heads and spatial dimensions.
        Args:
            att: (n_head, B, T, H, W)
        Returns:
            attn_weights: (B, T) normalized weights over the sequence days.
        """
        # Average over heads and spatial dimensions (dim 0, 3, 4)
        weights = att.mean(dim=(0, 3, 4))  # (B, T)
        # Normalize so each batch row sums to 1.0
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        return weights


def get_wildfire_utae(
    input_dim: int = 17,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> WildfireUTAE:
    """Helper factory function to create and optionally load WildfireUTAE model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WildfireUTAE(input_dim=input_dim)

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {checkpoint_path}")

    model = model.to(device)
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing WildfireUTAE model on device: {device}")

    model = WildfireUTAE(input_dim=17).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,} (Trainable: {trainable_params:,})")

    # Forward pass test
    dummy_x = torch.randn(2, 5, 17, 128, 128, device=device)
    logits, att = model(dummy_x, return_att=True)
    probs, mask = model.predict(dummy_x, threshold=0.5)

    print(f"Logits shape: {logits.shape}")
    print(f"Attention map shape: {att.shape}")
    print(f"Probabilities shape: {probs.shape}, range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")
    print(f"Prediction mask shape: {mask.shape}, unique values: {torch.unique(mask).tolist()}")

    attn_summary = WildfireUTAE.get_attention_weights(att)
    print(f"Temporal attention weights per sample (shape {attn_summary.shape}):")
    print(attn_summary.detach().cpu().numpy())
    print("WildfireUTAE model verified successfully!")
