"""CASITA inference: a T2w brain volume + a requested annotation protocol -> segmentation.

    python casita/predict.py scan.nii.gz --protocol feta --weights casita.pt

Takes any NIfTI at any resolution: the preprocessing chain below (orientation ->
0.75 mm isotropic -> 256^3) is applied internally, and the labels are resampled
back afterwards so the output matches the input's shape, voxel spacing and
affine.

Output values are the shared 9-channel space -- 0 background, 1 CSF, 2 cortical
grey matter, 3 white matter, 4 ventricles, 5 cerebellum, 6 deep grey matter,
7 brainstem/spinal cord, 8 hippocampi+amygdalae -- the same space every CASITA
metric was computed in. The requested protocol decides which channels compete in
the argmax and where boundaries fall.

Made this light on dependencies: drives ``CASITAUNet`` directly rather than the
LightningModule.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.ndimage
import torch
from monai.data import MetaTensor
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    CropForegroundd,
    Orientationd,
    ScaleIntensityd,
    SignalFillEmptyd,
    Spacingd,
    SpatialPadd,
)

try:
    from casita.models.casita_unet import CASITAUNet
except ModuleNotFoundError:  # invoked as a plain script rather than a module
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from casita.models.casita_unet import CASITAUNet


# Protocols are registered under these names at training time. `drawem9` is the
# name the paper uses for what the configs call `drawem9_albert`; accept both so
# nobody has to know the internal spelling.
PROTOCOL_ALIASES = {"drawem9": "drawem9_albert"}

CHANNEL_NAMES = {
    0: "background",       1: "CSF",              2: "cortical grey matter",
    3: "white matter",     4: "ventricles",       5: "cerebellum",
    6: "deep grey matter", 7: "brainstem and spinal cord",
    8: "hippocampi and amygdalae",
}


def build_transforms() -> Compose:
    """Preprocessing applied before the network, mirroring the chain every
    CASITA evaluation used (configs/data/transforms/inference.yaml).

    CropForegroundd recentres the 256^3 window on the head rather than the full
    field of view, which matters for off-centre or tilted acquisitions; it does
    not move the prediction, because the affine travels with the MetaTensor and
    the output is resampled back through it.
    """
    return Compose([
        Orientationd(keys=["image"], axcodes="RAS"),
        SignalFillEmptyd(keys=["image"], replacement=0),
        CropForegroundd(keys=["image"], source_key="image", allow_smaller=True, margin=0),
        Spacingd(keys=["image"], pixdim=[0.75, 0.75, 0.75], mode=["bilinear"]),
        SpatialPadd(keys=["image"], spatial_size=[256, 256, 256], mode="constant"),
        CenterSpatialCropd(keys=["image"], roi_size=[256, 256, 256]),
        ScaleIntensityd(keys=["image"], minv=0, maxv=1),
    ])


def load_model(
    weights: str | Path, device: torch.device
) -> tuple[CASITAUNet, list[str], torch.Tensor]:
    """Load the released ``.pt``, or a Lightning ``.ckpt`` written by train.py.

    Returns ``(net, protocol_names, annotation_mask)``, the mask being
    ``(num_protocols, num_fg_channels)`` bool -- True where that protocol
    annotates that channel.
    """
    weights = Path(weights)

    try:
        ck = torch.load(weights, map_location="cpu", weights_only=True)
        meta = ck["meta"]
        net = CASITAUNet(**meta["net_kwargs"])
        net.load_state_dict(ck["state_dict"])
        ann = torch.as_tensor(meta["annotation_mask"], dtype=torch.bool)
        protocol_names = list(meta["protocol_names"])
    except Exception:
        ck = torch.load(weights, map_location="cpu", weights_only=False)
        if "hyper_parameters" not in ck:
            raise ValueError(
                f"{weights} is neither a released CASITA .pt nor a Lightning "
                "checkpoint with hyper_parameters."
            )
        hp = ck["hyper_parameters"]
        sd = ck["state_dict"]
        net = hp["net"]
        # Strip the LightningModule prefix: `net.encoder...` -> `encoder...`
        net.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("net.")})
        ann = sd["annotation_mask"].bool()
        protocol_names = list(hp["protocol_names"])

    net.eval().to(device)
    return net, protocol_names, ann.to(device)


def resample_to_reference(
    vol: np.ndarray, src_affine: np.ndarray, reference: str | Path
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a label volume from the working grid back onto ``reference``'s grid.

    Nearest-neighbour, via the exact voxel-to-voxel map derived from the two
    affines, so no resampling library is needed and labels are never blended.
    """
    ref = nib.load(str(reference))
    ref_affine = ref.affine

    # reference voxel -> world -> source voxel
    m = np.linalg.inv(src_affine) @ ref_affine
    out = scipy.ndimage.affine_transform(
        vol.astype(np.float64), m[:3, :3], offset=m[:3, 3],
        output_shape=ref.shape[:3], order=0, cval=0, prefilter=False,
    )
    return out, ref_affine


def predict_one(
    image_path: Path,
    protocol: str,
    net: CASITAUNet,
    protocol_names: list[str],
    annotation_mask: torch.Tensor,
    transforms: Compose,
    device: torch.device,
    mask: bool = True,
    space: str = "input",
) -> tuple[np.ndarray, np.ndarray]:
    """Segment one volume. Returns ``(labels, affine)``."""
    nii = nib.load(str(image_path))
    image = MetaTensor(
        x=torch.from_numpy(np.asanyarray(nii.dataobj).astype(np.float32)).unsqueeze(0),
        affine=torch.as_tensor(nii.affine, dtype=torch.float64),
    )

    data = transforms({"image": image})
    x = data["image"].as_tensor().unsqueeze(0).to(device)
    work_affine = np.asarray(data["image"].affine, dtype=np.float64)

    p_idx = protocol_names.index(protocol)
    cond = torch.zeros(1, len(protocol_names), device=device)
    cond[0, p_idx] = 1.0

    with torch.inference_mode():
        logits = net(x, cond)  # (1, num_fg + 1, 256, 256, 256)

        if mask:
            # Channels the requested protocol does not annotate are removed from
            # the argmax, so the output is valid under the requested convention.
            # This mirrors the training-time loss mask.
            ann = annotation_mask[p_idx]                       # (num_fg,) bool
            logits[:, 1:][:, ~ann] = float("-inf")

        pred = logits.argmax(dim=1).squeeze(0)                 # (256, 256, 256)

    labels = pred.cpu().numpy().astype(np.uint8)

    if space == "input":
        labels, affine = resample_to_reference(labels, work_affine, image_path)
        return labels.astype(np.uint8), affine
    return labels, work_affine


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Segment a perinatal brain MRI under a requested annotation protocol.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("input", type=Path, help="NIfTI file, or a directory of them")
    ap.add_argument("--protocol", required=True,
                    help="requested annotation protocol: feta, drawem9, or bobs9")
    ap.add_argument("--weights", type=Path, default=None,
                    help="CASITA weights: the released .pt, or a .ckpt from train.py. "
                         "Defaults to $CASITA_WEIGHTS, which the Docker image sets when "
                         "weights are baked into it")
    ap.add_argument("--output", type=Path, default=None,
                    help="output file, or directory when input is a directory "
                         "[default: alongside the input]")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--space", default="input", choices=["input", "model"],
                    help="'input' aligns the output with the input volume; 'model' "
                         "returns it on the internal 0.75 mm 256^3 grid instead")
    ap.add_argument("--no-mask", action="store_true",
                    help="let every channel compete in the argmax, including ones the "
                         "requested protocol does not annotate. Produces output that is "
                         "NOT valid under that convention; for inspection only")
    ap.add_argument("--suffix", default=None,
                    help="output filename suffix [default: _<protocol>_dseg]")
    args = ap.parse_args()

    # $CASITA_WEIGHTS is how the Docker image points at baked-in weights. It is
    # ignored when it points nowhere, so an image built without them still gives
    # the useful "pass --weights" message rather than a confusing missing-file one
    # but we build with the weights baked in.
    weights = args.weights
    if weights is None:
        env = os.environ.get("CASITA_WEIGHTS")
        if env and Path(env).exists():
            weights = Path(env)
        else:
            ap.error("no weights given: pass --weights /path/to/casita.pt")
    if not weights.exists():
        ap.error(f"weights not found: {weights}")

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto"
        else args.device
    )

    net, protocol_names, annotation_mask = load_model(weights, device)

    protocol = PROTOCOL_ALIASES.get(args.protocol, args.protocol)
    if protocol not in protocol_names:
        ap.error(
            f"unknown protocol {args.protocol!r}. This model was trained on: "
            f"{', '.join(protocol_names)} "
            f"(aliases: {', '.join(f'{k} -> {v}' for k, v in PROTOCOL_ALIASES.items())})"
        )

    inputs = (sorted(args.input.glob("*.nii*")) if args.input.is_dir() else [args.input])
    if not inputs:
        ap.error(f"no NIfTI files found in {args.input}")

    if args.input.is_dir():
        out_dir = args.output or args.input
        out_dir.mkdir(parents=True, exist_ok=True)

    transforms = build_transforms()
    suffix = args.suffix or f"_{protocol}_dseg"
    annotated = [c + 1 for c in range(annotation_mask.shape[1]) if annotation_mask[protocol_names.index(protocol), c]]

    print(f"CASITA  |  protocol: {protocol}  |  device: {device}")
    print(f"  channels under this protocol: "
          f"{', '.join(f'{c}={CHANNEL_NAMES[c]}' for c in annotated)}")
    if args.no_mask:
        print("  masking DISABLED — output is not valid under the requested convention")

    for path in inputs:
        labels, affine = predict_one(
            path, protocol, net, protocol_names, annotation_mask, transforms,
            device, mask=not args.no_mask, space=args.space,
        )

        if args.input.is_dir():
            stem = path.name.split(".nii")[0]
            out_path = out_dir / f"{stem}{suffix}.nii.gz"
        else:
            out_path = args.output or path.with_name(
                f"{path.name.split('.nii')[0]}{suffix}.nii.gz"
            )

        nib.save(nib.Nifti1Image(labels, affine), str(out_path))
        present = sorted(int(v) for v in np.unique(labels))
        print(f"  {path.name} -> {out_path}   labels present: {present}")


if __name__ == "__main__":
    main()
