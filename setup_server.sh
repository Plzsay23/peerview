#!/usr/bin/env bash
# One-time environment for G0 on the lab server (A6000 sm_86 / RTX PRO 5000 sm_120).
# Usage: bash setup_server.sh [EXT_DIR]     (default EXT_DIR=~/Desktop/sj/peerview_ext)
# Installs into conda env "peerview". torch goes LAST: earlier pins would otherwise drag it back.
set -euo pipefail
EXT=${1:-$HOME/Desktop/sj/peerview_ext}
mkdir -p "$EXT" && cd "$EXT"

[ -d LIBERO ] || git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git
[ -d Libero-Occ ] || git clone --depth 1 https://github.com/litsh/Libero-Occ.git

source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -q '^peerview ' || conda create -y -n peerview python=3.10
conda activate peerview

pip install "numpy<2" "mujoco==3.2.7" "robosuite==1.4.1" "bddl==1.0.1" future easydict "gym==0.25.2" \
    cloudpickle pyyaml pillow matplotlib imageio
pip install "transformers==4.40.1" "tokenizers==0.19.1" "timm==0.9.10" accelerate sentencepiece protobuf \
    "tensorflow-cpu==2.15.1"
pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu128 torch torchvision
pip install "numpy<2"  # torch's reinstall may pull numpy 2

# LIBERO has no libero/__init__.py, so `pip install -e` does not expose it: use PYTHONPATH (run.md).
# Write LIBERO's config non-interactively (importing libero.libero otherwise prompts on stdin).
mkdir -p ~/.libero
R="$EXT/LIBERO/libero/libero"
cat > ~/.libero/config.yaml <<YAML
benchmark_root: $R
bddl_files: $R/bddl_files
init_states: $R/init_files
datasets: $R/../datasets
assets: $R/assets
YAML

python - <<'PY'
import torch, mujoco, robosuite, transformers, numpy
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "archs", torch.cuda.get_arch_list())
print("mujoco", mujoco.__version__, "robosuite", robosuite.__version__, "transformers", transformers.__version__, "numpy", numpy.__version__)
PY
