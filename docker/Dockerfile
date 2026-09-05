# LEMC on AMD Radeon RX 7600 (ROCm) — matches prior PIE GPU docker workflow.
#
# Base image: ROCm 7.14 + PyTorch 2.12 (same stack as YOLO density-boost run).
# PyTorch still uses torch.cuda.* on ROCm/HIP — that is expected.
FROM rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.12.0

WORKDIR /workspace/lemc

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget unzip ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Torch/torchvision already in base — do not pip install torch (would pull CUDA wheels).
RUN pip install --no-cache-dir \
    numpy pandas pyyaml scikit-learn scipy matplotlib \
    opencv-python-headless tqdm pytest gdown

# Code is bind-mounted at runtime; copy is only for standalone image use.
COPY lemc ./lemc
COPY scripts ./scripts
COPY configs ./configs
COPY tests ./tests
COPY paper ./paper
COPY README_REPRO.md DECISIONS.md PAPER_DRAFT.md SUBMISSION_CHECKLIST.md ./

ENV PYTHONPATH=/workspace/lemc \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    HSA_OVERRIDE_GFX_VERSION=11.0.0 \
    PYTORCH_ROCM_ARCH=gfx1102

CMD ["bash"]
