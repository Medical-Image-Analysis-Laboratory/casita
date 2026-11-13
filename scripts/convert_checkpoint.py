#!/usr/bin/env python3
"""Convert a training checkpoint into a portable, publishable weights file.

    python scripts/convert_checkpoint.py <run.ckpt> -o casita.pt

A Lightning ``.ckpt`` cannot be published as-is. ``save_hyperparameters()``
captured the live ``net`` object, so the pickle carries GLOBAL opcodes naming
``casita.models.casita_unet`` -- the file only opens where that exact
module path is importable, which breaks the moment the package is renamed, and
it forces ``weights_only=False`` on anyone who downloads it.

This writes instead::

    {"state_dict": <the UNet's tensors>,
     "meta": {"net_kwargs", "protocol_names", "annotation_mask",
              "num_fg_channels", "source_checkpoint"}}

which contains no pickled classes, loads under ``weights_only=True``, drops the
optimizer/loop/callback state an inference user has no use for (~19 MB -> ~6 MB),
and survives any future rename.

The architecture kwargs are derived from the loaded module tree, then checked by
rebuilding the network from them and requiring an exact state_dict match and
bit-identical forward output. Conversion fails loudly rather than writing a file
that silently reconstructs a different network.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from casita.models.casita_unet import CASITAUNet


def derive_net_kwargs(net: CASITAUNet, num_protocols_fallback: int) -> dict:
    """Recover the constructor arguments from a built network.

    CASITAUNet does not retain its init arguments, so every value is read back
    off the module tree. Verified by reconstruction in ``convert()``.
    """
    conditioned = bool(net.conditioned)

    if conditioned:
        linears = [m for m in net.cond_mlp.mlp if isinstance(m, nn.Linear)]
        num_protocols = linears[0].in_features
        mlp_hidden_dim = linears[0].out_features
        mlp_n_layers = len(linears)
    else:
        # Unconditioned runs build no MLP, so the one-hot width is unrecoverable
        # from the weights; take it from the segmentor's annotation_mask.
        num_protocols = num_protocols_fallback
        mlp_hidden_dim, mlp_n_layers = 64, 4

    n_levels = len(net.encoder)
    n_conv_per_level = len(net.encoder[0].convs)
    n_features_init = net.encoder[0].convs[0].out_channels
    feat_mult = (
        net.encoder[1].convs[0].out_channels // n_features_init if n_levels > 1 else 2
    )

    norm0 = net.encoder[0].norms[0]
    norm_type = (
        "batch" if isinstance(norm0, nn.BatchNorm3d)
        else "instance" if isinstance(norm0, nn.InstanceNorm3d)
        else None
    )

    # n_conditioned_layers counts the *last* N conv layers carrying FiLM. 0 is
    # the canonical "all of them", so normalise to 0 when every layer is FiLM'd.
    n_film = sum(sum(b.use_film) for b in net.encoder)
    n_film += sum(sum(b.use_film) for b in net.decoder)
    n_film += int(net.final_film is not None)
    n_total = n_levels * n_conv_per_level + (n_levels - 1) * n_conv_per_level + 1

    return {
        "num_protocols": int(num_protocols),
        "num_fg_channels": int(net.final_conv.out_channels - 1),
        "n_levels": int(n_levels),
        "n_features_init": int(n_features_init),
        "feat_mult": int(feat_mult),
        "n_conv_per_level": int(n_conv_per_level),
        "conv_size": int(net.encoder[0].convs[0].kernel_size[0]),
        "norm_type": norm_type,
        "n_conditioned_layers": 0 if n_film == n_total else int(n_film),
        "mlp_hidden_dim": int(mlp_hidden_dim),
        "mlp_n_layers": int(mlp_n_layers),
        "conditioned": conditioned,
    }


def convert(ckpt_path: Path, out_path: Path, force: bool = False) -> dict:
    if out_path.exists() and not force:
        raise FileExistsError(
            f"{out_path} already exists. Converted weights are a published "
            "artifact -- refusing to overwrite. Pass --force if you mean it."
        )
    for forbidden in ("None/logs", "trainignweights"):
        if forbidden in str(out_path.resolve()):
            raise ValueError(
                f"refusing to write into {forbidden} -- that is training state, "
                "not a release artifact."
            )

    print(f"Reading {ckpt_path}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "hyper_parameters" not in ck or "state_dict" not in ck:
        raise ValueError("not a Lightning checkpoint (no hyper_parameters/state_dict)")

    hp, sd = ck["hyper_parameters"], ck["state_dict"]
    protocol_names = list(hp["protocol_names"])
    annotation_mask = sd["annotation_mask"].bool()

    # The authoritative weights are the ones in state_dict, not whatever the
    # pickled net happens to carry: the net object is a snapshot from __init__
    # time, while state_dict is what the trainer actually saved.
    net: CASITAUNet = hp["net"]
    net_sd = {k[len("net."):]: v for k, v in sd.items() if k.startswith("net.")}
    net.load_state_dict(net_sd)
    net.eval()

    net_kwargs = derive_net_kwargs(net, num_protocols_fallback=annotation_mask.shape[0])
    print("  derived architecture:")
    for k, v in net_kwargs.items():
        print(f"    {k:<22} {v}")

    # ---- verify the derivation reconstructs the identical network ----
    rebuilt = CASITAUNet(**net_kwargs).eval()
    src_sd = net.state_dict()
    reb_sd = rebuilt.state_dict()
    if src_sd.keys() != reb_sd.keys():
        only_src = sorted(set(src_sd) - set(reb_sd))[:5]
        only_reb = sorted(set(reb_sd) - set(src_sd))[:5]
        raise ValueError(
            f"derived kwargs rebuild a different architecture.\n"
            f"  missing from rebuild: {only_src}\n  unexpected in rebuild: {only_reb}"
        )
    for k in src_sd:
        if src_sd[k].shape != reb_sd[k].shape:
            raise ValueError(
                f"shape mismatch for {k}: {tuple(src_sd[k].shape)} vs {tuple(reb_sd[k].shape)}"
            )
    rebuilt.load_state_dict(src_sd)

    # Strongest check: same input, same output, bit for bit.
    torch.manual_seed(0)
    x = torch.randn(1, 1, 32, 32, 32)
    cond = torch.zeros(1, net_kwargs["num_protocols"])
    cond[0, 0] = 1.0
    with torch.no_grad():
        delta = (net(x, cond) - rebuilt(x, cond)).abs().max().item()
    if delta != 0.0:
        raise ValueError(f"rebuilt network output differs (max |diff| = {delta})")
    print(f"  reconstruction verified: identical state_dict and forward output")

    payload = {
        "state_dict": src_sd,
        "meta": {
            "net_kwargs": net_kwargs,
            "protocol_names": protocol_names,
            "annotation_mask": annotation_mask,
            "num_fg_channels": net_kwargs["num_fg_channels"],
            # basename only -- the full path is a local detail, not provenance
            # anyone else can use, and it should not travel with a public file.
            "source_checkpoint": ckpt_path.name,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)

    # ---- verify what landed on disk, loaded the way users will load it ----
    back = torch.load(out_path, map_location="cpu", weights_only=True)
    for k, v in src_sd.items():
        if not torch.equal(back["state_dict"][k], v):
            raise ValueError(f"written weights differ from source for {k}")
    if not torch.equal(back["meta"]["annotation_mask"], annotation_mask):
        raise ValueError("written annotation_mask differs from source")
    if list(back["meta"]["protocol_names"]) != protocol_names:
        raise ValueError("written protocol_names differ from source")

    src_mb = ckpt_path.stat().st_size / 1e6
    out_mb = out_path.stat().st_size / 1e6
    print(f"  verified after write: loads with weights_only=True, tensors identical")
    print(f"\nWrote {out_path}  ({src_mb:.1f} MB -> {out_mb:.1f} MB)")
    print(f"  protocols: {', '.join(protocol_names)}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert a Lightning checkpoint into publishable CASITA weights."
    )
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing output file")
    args = ap.parse_args()
    convert(args.checkpoint, args.output, force=args.force)


if __name__ == "__main__":
    main()
