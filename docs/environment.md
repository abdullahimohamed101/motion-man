# Environment: the NVIDIA execution host (spec C§4, plan native-g0-g1 step 3)
Every value here was captured on the PC itself (owner's screenshots / pasted output), never inferred on the Mac.
Missing values are listed as unknown, not guessed.

## Inventory (2026-11-04, `nvidia-smi` in Windows PowerShell)
| Item | Value | Source |
|------|-------|--------|
| GPU | NVIDIA GeForce RTX 3060 Ti (Ampere; compute capability 8.6 per NVIDIA's published table) | `nvidia-smi` |
| VRAM | 8192 MiB total; 4405 MiB in use at capture (desktop apps and a game running) | `nvidia-smi` |
| Driver | 610.88 (KMD), Windows display driver model WDDM | `nvidia-smi` |
| Highest CUDA the driver supports | 13.3 (`CUDA UMD Version`); not proof a toolkit is installed | `nvidia-smi` |
| Power | 200 W cap; 31 W, P5, 51 °C at idle-ish capture | `nvidia-smi` |
| OS | Windows (edition and build unknown) | prompt `PS C:\Users\…` |
| WSL | `wsl` not recognised: not installed or unavailable | owner |
| `nvcc`, CMake, OpenCV, TensorRT | not installed in any Linux environment yet | owner |
| Disk free | unknown | |

## Consequences
- Windows host → the execution environment will be WSL2 + Ubuntu with the Windows NVIDIA driver (C§4; NVIDIA CUDA
  on WSL guide). Never install a Linux GPU driver inside WSL.
- 8 GB VRAM bounds engine size, batch and stream count; G4 admission must measure real free memory (C§7).
- Benchmarks must record competing GPU workloads; games and overlays were using ~4.4 GB at capture (C§10).
- Toolkit versions (CUDA, TensorRT, compiler) are chosen against NVIDIA's support matrix for Ampere + this driver
  once WSL is in place, and recorded in ADR-005.

## Still needed
WSL2 + Ubuntu installed; then inside Ubuntu: `nvidia-smi`, `nvcc --version`, `cmake --version`,
`uname -a; cat /etc/os-release`, `df -h ~`; Windows build number (`winver`).
