# CASITA

**C**onditioning **A**cross protocols for **S**egmentation of **I**nfant and fe**T**al br**A**ins.

A 3D U-Net that segments perinatal brain MRI under a *requested annotation
protocol*. It is trained jointly on cohorts whose annotation conventions
disagree, and conditions every layer on the protocol you ask for via FiLM, so
one model reproduces each convention instead of averaging them into a
compromise.

Asking for a different protocol on the same image changes the output in the way
the protocol definitions predict — most visibly for the ventricles, where the
three conventions include different structures.

---

## Install

```bash
git clone https://github.com/Medical-Image-Analysis-Laboratory/casita
cd casita
pip install .            # inference
pip install -e .[train]  # inference + training
```

Conda users: `conda env create -f environment.yml`.

## Inference

Download `casita.pt` from the [releases page](https://github.com/Medical-Image-Analysis-Laboratory/casita/releases), then:

```bash
casita-predict scan.nii.gz --protocol feta --weights casita.pt
```

**Input**: any T2w brain volume as NIfTI, at any resolution or orientation. No
preprocessing required — orientation, resampling to 0.75 mm and cropping happen
internally. **Output**: a label map aligned with your input, matching its shape,
voxel spacing and affine, so the two overlay directly in any viewer.

Point `--protocol` at the convention you want (`feta`, `drawem9`, `bobs9`),
independently of where the image came from. Other options: `--output`,
`--device`, `--space model` to keep the 0.75 mm working grid, and a directory
as input to batch over its NIfTIs.

### Container

The published image has the weights baked in, so nothing else is needed:

```bash
docker run --rm -v "$PWD":/data \
    ghcr.io/medical-image-analysis-laboratory/casita:cpu \
    scan.nii.gz --protocol feta --output seg.nii.gz
```

On HPC, convert it to Apptainer/Singularity first:

```bash
apptainer build casita.sif docker://ghcr.io/medical-image-analysis-laboratory/casita:cpu
apptainer run --bind "$PWD":/data casita.sif \
    scan.nii.gz --protocol feta --output seg.nii.gz
```

The published image is CPU-only — around 1–3 minutes per scan, versus ~7 s on a
GPU. To build a CUDA image yourself:

```bash
docker build -t casita:gpu \
    --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 \
    --build-arg WEIGHTS_URL=<url of casita.pt from the releases page> .
docker run --rm --gpus all -v "$PWD":/data casita:gpu \
    scan.nii.gz --protocol feta --output seg.nii.gz
```

Omit `WEIGHTS_URL` to build without weights, then mount a `.pt` and pass
`--weights` at run time.

## Output labels

Predictions always use this shared 9-channel space, whichever protocol you
request. The protocol determines which channels compete and where boundaries
fall, not which integers are written.

| | structure | feta | drawem9 | bobs9 |
|---|---|:-:|:-:|:-:|
| 0 | background | ● | ● | ● |
| 1 | cerebrospinal fluid | ● | ● | — |
| 2 | cortical grey matter | ● | ● | ● |
| 3 | white matter | ● | ● | ● |
| 4 | ventricles | ● | ● | ● |
| 5 | cerebellum | ● | ● | ● |
| 6 | deep grey matter | ● | ● | ● |
| 7 | brainstem and spinal cord | ● | ● | ● |
| 8 | hippocampi and amygdalae | — | ● | ● |

Channels a protocol does not annotate are excluded before the argmax, so they
never appear in its output.

## The three protocols

They disagree about what a label *means*. The
clearest case is the ventricles:

| protocol | ventricle label includes | source cohorts |
|---|---|---|
| `drawem9` | lateral ventricles only | dHCP fetal and neonatal |
| `bobs9` | + third and fourth ventricles | Baby Open Brains |
| `feta` | + cavum septi pellucidi | FeTA 2.4, CHUV |

`bobs9` also omits CSF, and `feta` omits hippocampi/amygdalae. Choose the
protocol whose definitions match the analysis you are doing.

## Training

Training runs from a repo checkout.

**1. Prepare the data.** Volumes must be 0.75 mm isotropic on disk; the
training pipeline does not resample.

```bash
python scripts/prepare_training_data.py /raw/cohort /prepared/cohort
```

Expects a BIDS-style layout, `sub-XXX/[ses-YYY/]anat/sub-XXX_T2w.nii.gz` with a
matching `_dseg.nii.gz`.

**2. Give each cohort a split CSV** with `participant_id` and `splits` columns,
whose values match `train_split` / `val_split` / `test_split` in the config.

**3. Map your labels** in `configs/label_map_multiprotocol.csv`, which maps each
cohort's raw segmentation integers onto the shared channels. It ships filled in
for the three protocols above; add rows for your own. Adding a *new* protocol
name changes the conditioning width, so the published weights will no longer
load and you must train from scratch.

**4. Train.**

```bash
export CASITA_DATA_ROOT=/path/to/cohorts
python casita/train.py experiment=casita
```

Edit `configs/experiment/casita.yaml` to point at your cohorts. `data.num_workers`
defaults to 0; raising it speeds up training, but augmentation runs on the GPU
inside the data pipeline, so each worker holds its own CUDA context (increase
it only if you have GPU memory to spare!).


## Scope

This repository contains training and inference only. The evaluation harness,
statistical analysis, and no subject-level data of any kind ships here.

## Attribution

`casita/synthgen/` is vendored from
[FetalSynGen](https://github.com/Medical-Image-Analysis-Laboratory/fetalsyngen)
under Apache-2.0 — see [NOTICE](NOTICE). The rest is MIT; see [LICENSE](LICENSE).

## Citation

> D. Correa Orozco, V. Zalevskyi, T. Sanchez, M. Bach Cuadra.
> *CASITA: Conditioning across protocols for segmentation of infant and fetal brains.*
