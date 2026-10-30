#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/genesis3d"
source ../.venv/bin/activate
export PYOPENGL_PLATFORM=egl
export PATH="$HOME/.local/bin:$PATH"

echo "=== nvidia-smi ==="
nvidia-smi

echo "=== torch CUDA ==="
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
assert torch.cuda.is_available(), "CUDA not available"
PY

# ensure classifier exists
if [ ! -f shape_mlp_3d.joblib ]; then
  python shape_ai.py
fi

EPISODES="${1:-2}"
python genesis_sort.py --episodes "$EPISODES"
ffmpeg -y -i out/genesis_raw.mp4 -c:v libx264 -pix_fmt yuv420p out/PAI_final_demo_repro_02_genesis.mp4
cp -f out/PAI_final_demo_repro_02_genesis.mp4 "/home/ai4s/ダウンロード/2026年10月30日AIRoA/02_01Gensis版/PAI_final_demo_repro_02_genesis.mp4"
python -c "import json; r=json.load(open('out/results.json')); print({k:v for k,v in r.items() if k!='log'})"
ffprobe -v error -show_entries format=duration,size -of default=noprint_wrappers=1 out/PAI_final_demo_repro_02_genesis.mp4
echo "DONE: out/PAI_final_demo_repro_02_genesis.mp4"
