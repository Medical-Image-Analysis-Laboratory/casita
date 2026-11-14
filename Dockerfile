# CASITA inference image: a T2w volume + a requested protocol -> segmentation.
#
# Training is not containerised: it runs from a repo checkout (Hydra resolves
# configs relative to the source tree). This image ships predict.py only, which
# needs no Hydra, no Lightning and no augmentation stack.
#
# Build the CPU image (~800 MB, runs anywhere):
#   docker build -t casita:cpu .
#
# Build the GPU image (~5 GB, needs nvidia-container-toolkit on the host):
#   docker build -t casita:gpu \
#     --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 .
#
# Bake the weights in (6 MB) so the image is self-contained and users need
# nothing but `docker run`:
#   docker build -t casita:cpu \
#     --build-arg WEIGHTS_URL=https://github.com/Medical-Image-Analysis-Laboratory/casita/releases/download/v1.0.0/casita.pt .
#
# Run. With baked weights --weights is optional:
#   docker run --rm -v "$PWD":/data casita:cpu \
#     scan.nii.gz --protocol feta --output seg.nii.gz
#
#   docker run --rm --gpus all -v "$PWD":/data casita:gpu \
#     scan.nii.gz --protocol feta --output seg.nii.gz
#
# Without them, mount a .pt and point at it:
#   docker run --rm -v "$PWD":/data -v /path/to/weights:/w casita:cpu \
#     scan.nii.gz --protocol feta --weights /w/casita.pt --output seg.nii.gz

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

# The PyTorch CUDA wheels bundle their own CUDA runtime, so a plain slim base
# works for both variants -- only the host driver and container toolkit differ.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
# Pinned: an unpinned install picks up whatever CUDA version is newest on build day.
ARG TORCH_VERSION=2.5.1

WORKDIR /opt/casita

COPY pyproject.toml README.md LICENSE NOTICE ./
COPY licenses/ licenses/
COPY casita/ casita/

# torch first, from the selected index; the second install then sees the
# requirement already satisfied and pulls only monai/nibabel/numpy/scipy.
RUN pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url ${TORCH_INDEX} \
    && pip install --no-cache-dir .

# Optionally bake the released weights (~6 MB) into the image. Downloaded with
# the interpreter that is already here, so the slim base needs no curl/wget.
# Left empty, the image still works -- the user mounts a .pt and passes --weights.
ARG WEIGHTS_URL=""
RUN if [ -n "$WEIGHTS_URL" ]; then \
        python -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], '/opt/casita/casita.pt')" "$WEIGHTS_URL" \
        && ls -lh /opt/casita/casita.pt; \
    fi

# predict.py falls back to this, and ignores it when nothing was baked in.
ENV CASITA_WEIGHTS=/opt/casita/casita.pt

# Everything the user passes is relative to their mounted volume.
WORKDIR /data

ENTRYPOINT ["casita-predict"]
CMD ["--help"]
