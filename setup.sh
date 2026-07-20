#!/bin/bash
# TRELLIS.2 3D Generator - One-click setup
set -e

echo "========================================"
echo "  TRELLIS.2 3D Generator Setup"
echo "========================================"

# Check CUDA
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "WARNING: CUDA not detected. GPU required for inference."
fi

# 1. Clone TRELLIS.2
if [ ! -d "TRELLIS.2" ]; then
    echo ""
    echo "[1/3] Cloning TRELLIS.2..."
    git clone https://github.com/microsoft/TRELLIS.2.git
    cd TRELLIS.2
    # Use mirror if GitHub blocked
    # git clone https://kkgithub.com/microsoft/TRELLIS.2.git
else
    echo ""
    echo "[1/3] TRELLIS.2 already cloned"
    cd TRELLIS.2
fi

# 2. Install TRELLIS.2 dependencies
echo ""
echo "[2/3] Installing TRELLIS.2 dependencies..."
pip install -e . --quiet || {
    echo "Trying with mirror..."
    pip install -e . -i https://mirrors.aliyun.com/pypi/simple/ --quiet
}

# Install flash-attn separately (sometimes fails in requirements)
pip install flash-attn --no-build-isolation --quiet 2>/dev/null || {
    echo "NOTE: flash-attn install failed (may need CUDA dev tools)"
    echo "  Inference will use slower attention backend"
}

cd ..

# 3. Install our additional deps
echo ""
echo "[3/3] Installing additional dependencies..."
pip install -r requirements.txt --quiet 2>/dev/null || {
    pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --quiet
}

# 4. Download model weights
echo ""
echo "========================================"
echo "  Setup complete! Now download weights:"
echo "  python download_weights.py"
echo ""
echo "  Then generate:"
echo "  python generate.py <image> -o output.glb"
echo "========================================"
