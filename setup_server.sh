#!/usr/bin/env bash
# One-time environment for G0 on the lab server (4x RTX A6000, driver 535, Ubuntu 20.04).
# Usage: bash setup_server.sh [EXT_DIR] [ENV_NAME]   (defaults: ~/Desktop/sj/peerview_ext, ysj)
# Installs into conda env "ysj" (2nd arg). torch goes FIRST and pinned: accelerate would otherwise pull the
# newest PyPI torch, whose CUDA build needs a newer driver than 535. torch 2.2.0 + cu121 = OpenVLA's tested pin.
set -euo pipefail
EXT=${1:-$HOME/Desktop/sj/peerview_ext}
ENV_NAME=${2:-ysj}
mkdir -p "$EXT" && cd "$EXT"

[ -d LIBERO ] || git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git
[ -d Libero-Occ ] || git clone --depth 1 https://github.com/litsh/Libero-Occ.git

source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -q "^$ENV_NAME " || conda create -y -n "$ENV_NAME" python=3.10
conda activate "$ENV_NAME"

pip install --no-cache-dir "numpy<2" "mujoco==3.2.7" "robosuite==1.4.1" "bddl==1.0.1" future easydict "gym==0.25.2" \
    cloudpickle pyyaml pillow matplotlib imageio
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 "torch==2.2.0" "torchvision==0.17.0"
pip install --no-cache-dir "transformers==4.40.1" "tokenizers==0.19.1" "timm==0.9.10" accelerate sentencepiece protobuf \
    "tensorflow-cpu==2.15.1" "numpy<2"

# LIBERO has no libero/__init__.py, so `pip install -e` does not expose it: use PYTHONPATH (run.md).
# Write LIBERO's config non-interactively (importing libero.libero otherwise prompts on stdin).
# Kept inside EXT, not ~/.libero: run.md exports LIBERO_CONFIG_PATH to point here.
mkdir -p "$EXT/.libero"
R="$EXT/LIBERO/libero/libero"
cat > "$EXT/.libero/config.yaml" <<YAML
benchmark_root: $R
bddl_files: $R/bddl_files
init_states: $R/init_files
datasets: $R/../datasets
assets: $R/assets
YAML

# NVIDIA's EGL library is installed but not registered with glvnd (/usr/share/glvnd/egl_vendor.d has only
# 50_mesa.json) -> EGL sees 0 devices. Register it without sudo; run.md exports __EGL_VENDOR_LIBRARY_FILENAMES.
mkdir -p "$EXT/egl"
printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}\n' > "$EXT/egl/10_nvidia.json"

python - <<'PY'
import torch, mujoco, robosuite, transformers, numpy
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "archs", torch.cuda.get_arch_list())
print("mujoco", mujoco.__version__, "robosuite", robosuite.__version__, "transformers", transformers.__version__, "numpy", numpy.__version__)
PY
