#!/usr/bin/env python3
"""Resample a BIDS-style cohort to the resolution CASITA trains at.

    python scripts/prepare_training_data.py /raw/cohort /prepared/cohort

Only training data needs this. The training pipeline only crops and
pads -- it has no resampling step -- and the augmentation generator is
configured for 0.75 mm isotropic data, so volumes must already be at that
resolution on disk. (casita/predict.py resamples internally and accepts any
input resolution.)

Expects, and reproduces, a layout like::

    cohort/sub-001/anat/sub-001_T2w.nii.gz
    cohort/sub-002/ses-01/anat/sub-002_ses-01_T2w.nii.gz

Images are resampled with trilinear interpolation, segmentations with nearest
neighbour.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.data import MetaTensor
from monai.transforms import Orientation, Spacing


def find_pairs(root: Path, img_suffix: str, seg_suffix: str | None) -> list[tuple[Path, Path | None]]:
    """Find every *_<img_suffix>.nii.gz under root, with its segmentation if present."""
    pairs: list[tuple[Path, Path | None]] = []
    for img in sorted(root.rglob(f"*_{img_suffix}.nii.gz")):
        seg = None
        if seg_suffix:
            cand = img.parent / img.name.replace(f"_{img_suffix}.nii.gz", f"_{seg_suffix}.nii.gz")
            seg = cand if cand.exists() else None
        pairs.append((img, seg))
    return pairs


def resample(path: Path, resolution: float, is_seg: bool, size: int | None) -> nib.Nifti1Image:
    nii = nib.load(str(path))
    arr = np.asanyarray(nii.dataobj)
    t = MetaTensor(
        x=torch.from_numpy(arr.astype(np.float32)).unsqueeze(0),
        affine=torch.as_tensor(nii.affine, dtype=torch.float64),
    )

    t = Orientation(axcodes="RAS")(t)
    t = Spacing(pixdim=(resolution,) * 3, mode="nearest" if is_seg else "bilinear")(t)

    if size:
        from monai.transforms import CenterSpatialCrop, SpatialPad
        t = SpatialPad(spatial_size=[size] * 3, mode="constant")(t)
        t = CenterSpatialCrop(roi_size=[size] * 3)(t)

    out = t.as_tensor().squeeze(0).numpy()
    # Segmentations keep an integer dtype; images stay float.
    out = out.astype(np.uint8 if is_seg else np.float32)
    return nib.Nifti1Image(out, np.asarray(t.affine, dtype=np.float64))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Resample a cohort to CASITA's training resolution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("input", type=Path, help="BIDS-style cohort root")
    ap.add_argument("output", type=Path, help="where the resampled copy is written")
    ap.add_argument("--resolution", type=float, default=0.75, help="isotropic mm")
    ap.add_argument("--img-suffix", default="T2w")
    ap.add_argument("--seg-suffix", default="dseg",
                    help="pass '' to resample images only")
    ap.add_argument("--size", type=int, default=256,
                    help="also pad/crop to this cube, mirroring the preprocessing used "
                         "for the published model. Pass 0 to skip it -- the training "
                         "datamodule crops to 256 at load time regardless")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pairs = find_pairs(args.input, args.img_suffix, args.seg_suffix or None)
    if not pairs:
        ap.error(f"no *_{args.img_suffix}.nii.gz found under {args.input}")

    n_seg = sum(1 for _, s in pairs if s is not None)
    print(f"{len(pairs)} images ({n_seg} with segmentations) -> {args.resolution} mm isotropic")
    if args.size:
        print(f"  also pad/crop to {args.size}^3")

    for img, seg in pairs:
        for path, is_seg in ((img, False), (seg, True)):
            if path is None:
                continue
            out_path = args.output / path.relative_to(args.input)
            if out_path.exists() and not args.overwrite:
                print(f"  skip (exists) {out_path.relative_to(args.output)}")
                continue
            print(f"  {path.relative_to(args.input)}")
            if args.dry_run:
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(resample(path, args.resolution, is_seg, args.size), str(out_path))

    if args.dry_run:
        print("\ndry run: nothing written")
    else:
        print(f"\nwritten to {args.output}")
        print("Point CASITA_DATA_ROOT at the parent of this folder, and give each "
              "cohort a split CSV with participant_id,splits columns.")


if __name__ == "__main__":
    main()
