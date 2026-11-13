"""CASITA UNet — a protocol-conditioned 3D UNet.

Builds on the FiLM-based label conditioning of CoNeMOS (Billot et al., MIDL
2024), but conditioned on the requested annotation protocol (vs. a single
label).

Each training sample is conditioned on a *protocol* one-hot vector (one per
image).  The network outputs a sigmoid
probability map for every foreground channel simultaneously; the loss is masked
to the channels annotated by the sample's protocol.

Architecture:
  condition (num_protocols,) -> 4-layer MLP -> z (64,)
  image (1, H, W, D)  ─┐
                        ├─ UNet encoder-decoder with FiLM(z) after every block
  z ─────────────────── ┘
  -> 1×1 conv + FiLM -> sigmoid -> (num_fg_channels, H, W, D)
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class FiLM(nn.Module):
    """Feature-wise Linear Modulation (Perez et al., 2018).

    Each FiLM instance has its own Linear projection from the shared MLP
    embedding z to per-channel gamma and beta.
    """

    def __init__(self, num_features: int, cond_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(cond_dim, 2 * num_features)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # x: (B, C, *spatial),  z: (B, cond_dim)
        params = self.fc(z)                    # (B, 2C)
        gamma, beta = params.chunk(2, dim=1)   # (B, C) each
        ndim = x.dim() - 2
        for _ in range(ndim):
            gamma = gamma.unsqueeze(-1)
            beta  = beta.unsqueeze(-1)
        return (1.0 + gamma) * x + beta


class CondMLP(nn.Module):
    """Maps the protocol one-hot to a shared latent embedding z."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, n_layers: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z)


class ConvBlock3D(nn.Module):
    """n_conv sequential Conv3d -> ReLU -> Norm [-> FiLM] layers.

    Conditioned convs have no bias because FiLM's beta plays that role.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_conv: int,
        conv_size: int,
        norm_type: Optional[str],
        use_film: list[bool],
        cond_dim: int,
    ) -> None:
        super().__init__()
        assert len(use_film) == n_conv

        self.convs    = nn.ModuleList()
        self.norms    = nn.ModuleList()
        self.films    = nn.ModuleList()
        self.use_film = use_film

        for i in range(n_conv):
            in_ch = in_channels if i == 0 else out_channels
            self.convs.append(
                nn.Conv3d(in_ch, out_channels, kernel_size=conv_size,
                          padding=conv_size // 2, bias=not use_film[i])
            )
            if norm_type == "batch":
                self.norms.append(nn.BatchNorm3d(out_channels))
            elif norm_type == "instance":
                self.norms.append(nn.InstanceNorm3d(out_channels, affine=False))
            else:
                self.norms.append(nn.Identity())

            self.films.append(FiLM(out_channels, cond_dim) if use_film[i] else nn.Identity())

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        for conv, norm, film, use_f in zip(
            self.convs, self.norms, self.films, self.use_film
        ):
            x = F.relu(conv(x), inplace=True)
            x = norm(x)
            if use_f:
                x = film(x, z)
        return x


# ---------------------------------------------------------------------------
# Full UNet
# ---------------------------------------------------------------------------

class CASITAUNet(nn.Module):
    """FiLM-conditioned 3-D UNet conditioned on protocol.

    Args:
        num_protocols: Size of the one-hot protocol conditioning vector.
        num_fg_channels: Number of foreground segmentation output channels
            (i.e. max channel index in label_map_csv, excluding background=0).
        n_levels: Number of resolution levels.
        n_features_init: Feature maps at the first encoder level.
        feat_mult: Feature multiplier per level.
        n_conv_per_level: Convolutions per encoder/decoder block.
        conv_size: Kernel size for all 3-D convolutions.
        norm_type: ``"batch"``, ``"instance"``, or ``None``.
        n_conditioned_layers: How many of the *last* conv layers receive FiLM.
            ``0`` conditions all layers (recommended).
        mlp_hidden_dim: Hidden dimension of the conditioning MLP.
        mlp_n_layers: Depth of the conditioning MLP.
    """

    def __init__(
        self,
        num_protocols: int,
        num_fg_channels: int,
        n_levels: int = 4,
        n_features_init: int = 16,
        feat_mult: int = 2,
        n_conv_per_level: int = 2,
        conv_size: int = 3,
        norm_type: Optional[str] = None,
        n_conditioned_layers: int = 0,
        mlp_hidden_dim: int = 64,
        mlp_n_layers: int = 4,
        conditioned: bool = True,
    ) -> None:
        super().__init__()

        self.conditioned = conditioned
        cond_dim = mlp_hidden_dim

        if conditioned:
            self.cond_mlp = CondMLP(num_protocols, mlp_hidden_dim, mlp_n_layers)

        # When conditioned=False, push idx_start beyond all layers so every
        # use_film list is all-False (FiLM layers become parameter-free Identity).
        n_enc_conv   = n_levels * n_conv_per_level
        n_dec_conv   = (n_levels - 1) * n_conv_per_level
        n_conv_total = n_enc_conv + n_dec_conv + 1   # +1 for final 1x1
        if not conditioned:
            idx_start = n_conv_total
        else:
            if n_conditioned_layers == 0:
                n_conditioned_layers = n_conv_total
            idx_start = n_conv_total - n_conditioned_layers

        feats = [int(n_features_init * (feat_mult ** i)) for i in range(n_levels)]

        # ---- Encoder ----
        self.encoder = nn.ModuleList()
        self.pool    = nn.MaxPool3d(kernel_size=2, stride=2)
        for lvl in range(n_levels):
            use_film = [
                (lvl * n_conv_per_level + k) >= idx_start
                for k in range(n_conv_per_level)
            ]
            in_ch = 1 if lvl == 0 else feats[lvl - 1]
            self.encoder.append(
                ConvBlock3D(in_ch, feats[lvl], n_conv_per_level,
                            conv_size, norm_type, use_film, cond_dim)
            )

        # ---- Decoder ----
        self.up      = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for lvl in range(n_levels - 1):
            rev      = n_levels - 2 - lvl   # skip index (deepest first)
            in_ch_up = feats[rev + 1]        # features arriving from below
            skip_ch  = feats[rev]            # features from the skip
            out_ch   = feats[rev]
            use_film = [
                (n_enc_conv + lvl * n_conv_per_level + k) >= idx_start
                for k in range(n_conv_per_level)
            ]
            self.up.append(
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
            )
            self.decoder.append(
                ConvBlock3D(in_ch_up + skip_ch, out_ch, n_conv_per_level,
                            conv_size, norm_type, use_film, cond_dim)
            )

        # ---- Final 1x1 conv (no bias) + FiLM ----
        # n_out = background (0) + num_fg_channels foreground structures.
        n_out = num_fg_channels + 1
        final_layer_idx   = n_enc_conv + n_dec_conv
        final_conditioned = final_layer_idx >= idx_start
        self.final_conv = nn.Conv3d(feats[0], n_out, kernel_size=1,
                                    bias=not final_conditioned)
        self.final_film: Optional[FiLM] = (
            FiLM(n_out, cond_dim) if final_conditioned else None
        )

    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:         Image tensor       (B, 1, H, W, D).
            condition: Protocol one-hot   (B, num_protocols).
        Returns:
            Raw logits (B, num_fg_channels + 1, H, W, D).
            Channel 0 = background, channels 1..num_fg_channels = structures.
            No activation applied, use F.cross_entropy for the loss (which
            applies softmax internally) and torch.softmax for prediction.
        """
        z = self.cond_mlp(condition) if self.conditioned else None  # (B, cond_dim) or None

        # Encoder
        skips: list[torch.Tensor] = []
        for lvl, block in enumerate(self.encoder):
            x = block(x, z)
            if lvl < len(self.encoder) - 1:
                skips.append(x)
                x = self.pool(x)

        # Decoder
        for up, block, skip in zip(self.up, self.decoder, reversed(skips)):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:   # odd spatial dims
                x = F.interpolate(x, size=skip.shape[2:],
                                  mode="trilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = block(x, z)

        x = self.final_conv(x)
        if self.final_film is not None:
            x = self.final_film(x, z)

        return x   # raw logits (B, num_fg_channels + 1, H, W, D)
