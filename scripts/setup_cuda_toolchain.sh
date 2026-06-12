#!/bin/bash
# Local CUDA 12.8 toolchain from NVIDIA redist tarballs (no sudo), then JIT-build gsplat.
set -ex
CH=$HOME/.local/cuda-12.8
rm -rf "$CH"; mkdir -p "$CH" /tmp/cuda-redist
cd /tmp/cuda-redist
BASE=https://developer.download.nvidia.com/compute/cuda/redist
for pkg in cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.8.93-archive.tar.xz \
           cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.8.90-archive.tar.xz \
           cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.8.90-archive.tar.xz \
           cuda_profiler_api/linux-x86_64/cuda_profiler_api-linux-x86_64-12.8.90-archive.tar.xz; do
  f=$(basename "$pkg")
  [ -f "$f" ] || curl -fsSLO "$BASE/$pkg"
  tar -xJf "$f"
done
# merge all extracted archives into CUDA_HOME
for d in *-archive; do
  cp -rn "$d"/. "$CH"/ || true
done
ln -sfn "$CH/lib" "$CH/lib64" || true
"$CH/bin/nvcc" --version

export CUDA_HOME="$CH"
export PATH="$CH/bin:$PATH"
export MAX_JOBS=8
cd /tmp
python3 -c "
import gsplat
from gsplat.cuda._backend import _C
print('gsplat backend:', _C)
assert _C is not None, 'gsplat CUDA backend failed to build'
print('GSPLAT BUILD OK')
"
