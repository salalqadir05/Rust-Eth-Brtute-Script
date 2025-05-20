#!/usr/bin/env bash
set -euo pipefail

# run_cuda.sh: Comprehensive CUDA environment & device initialization checker
# Usage: chmod +x run_cuda.sh && ./run_cuda.sh

echo "\n=== 1. NVIDIA driver & GPU visibility (nvidia-smi) ==="
if ! command -v nvidia-smi &>/dev/null; then
  echo "Error: nvidia-smi not found. Install NVIDIA drivers."
else
  nvidia-smi
fi

echo "\n=== 2. CUDA compiler (nvcc) version ==="
if ! command -v nvcc &>/dev/null; then
  echo "Error: nvcc not found. Install CUDA Toolkit."
else
  nvcc --version
fi

echo "\n=== 3. Loaded NVIDIA kernel modules ==="
lsmod | grep -E 'nvidia(_uvm|_drm)?' || echo "Warning: NVIDIA kernel modules not loaded"

echo "\n=== 4. /dev/nvidia* device nodes & permissions ==="
if ls /dev/nvidia* &>/dev/null; then
  ls -l /dev/nvidia* || true
else
  echo "Warning: No /dev/nvidia* device files found"
fi

echo "\n=== 5. User groups for video/render access ==="
if groups "$USER" | grep -Eq '\b(video|render)\b'; then
  echo "OK: User '$USER' is in video/render group"
else
  echo "Warning: User '$USER' not in video or render group"
  echo "  -> add with: sudo usermod -aG video $USER"
fi

echo "\n=== 6. LD_LIBRARY_PATH & cuda .so visibility ==="
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "ldconfig cache for libcudart.so:"
ldconfig -p | grep libcudart || echo "Warning: libcudart not in ldconfig"

echo "\n=== 7. Scan for conflicting CUDA lib versions ==="
find /usr/local/cuda* /usr/lib/x86_64-linux-gnu -maxdepth 2 -type f -name "libcudart.so*" 2>/dev/null | sed 's/^/  /'

echo "\n=== 8. Test CUDA device query sample ==="
# Try multiple sample paths
for SAMPLE_DIR in "/usr/local/cuda/samples/1_Utilities/deviceQuery" \
                  "/usr/local/cuda-*/samples/1_Utilities/deviceQuery" \
                  "\$HOME/NVIDIA_CUDA-*/Samples/1_Utilities/deviceQuery"; do
  SAMPLE_DIR_EXPANDED=$(echo $SAMPLE_DIR)
  if [ -d "$SAMPLE_DIR_EXPANDED" ]; then
    echo "Found sample at: $SAMPLE_DIR_EXPANDED"
    (cd "$SAMPLE_DIR_EXPANDED" && make -j$(nproc) >/dev/null 2>&1 && echo "Running deviceQuery:" && "$SAMPLE_DIR_EXPANDED"/deviceQuery)
    break
  fi
done || echo "Info: CUDA sample deviceQuery not found or failed to run"

echo "\n=== 9. Minimal cuInit() test via C program ==="
# Create & compile a minimal CUDA Driver API test in C
cat << 'EOF' > /tmp/try_cuInit.c
#include <stdio.h>
#include <cuda.h>
int main() {
    CUresult r = cuInit(0);
    if (r == CUDA_SUCCESS) printf("cuInit() succeeded\n");
    else printf("cuInit() failed: %d\n", r);
    return 0;
}
EOF
if command -v gcc &>/dev/null; then
  gcc /tmp/try_cuInit.c -o /tmp/try_cuInit -lcuda && echo "Running cuInit() C test:" && /tmp/try_cuInit || echo "Error: cuInit() C test failed to compile or run"
else
  echo "Warning: gcc not found; skipping C cuInit() test"
fi

echo "\n=== 10. Summary ==="
echo "If any warnings or errors occurred above, address them by installing drivers, loading modules, fixing permissions, or resolving library conflicts."
