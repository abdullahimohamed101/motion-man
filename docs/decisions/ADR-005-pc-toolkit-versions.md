# ADR-005: PC toolkit: WSL2 Ubuntu 24.04, CUDA 13.3.1, TensorRT 11.2.1
Status: Accepted (G0 smoke test passed on the PC, 2026-11-07)
Date: 2026-11-07

## Context
Spec C§4 asks for one pinned, supported stack on the owner's PC. Inventory (`docs/environment.md`, captured over
SSH): RTX 3060 Ti (Ampere, SM 8.6, 8 GB), Windows 11 driver 610.88 whose highest CUDA is 13.3, WSL2 with Ubuntu
24.04.4 already installed and the GPU visible inside it, gcc 13.3. Package availability was read from NVIDIA's apt
indexes on 2026-11-07: the `wsl-ubuntu` CUDA repo tops out at `cuda-toolkit-13-3` 13.3.1; the `ubuntu2404` repo
has TensorRT up to 11.4.0 but built for CUDA 13.4, and 11.2.1.2 is the newest built for CUDA 13.3.

## Decision
| Component | Version | Source |
|-----------|---------|--------|
| Host | Windows 11 + WSL2, Ubuntu 24.04 (D3: WSL2, since the PC runs Windows) | existing |
| GPU driver | Windows 610.88; no Linux driver inside WSL | existing |
| CUDA toolkit | `cuda-toolkit-13-3` 13.3.1-1 | NVIDIA `wsl-ubuntu/x86_64` apt repo |
| TensorRT | `tensorrt` 11.2.1.2-1+cuda13.3 (`libnvinfer11`, `libnvinfer-dev`, `trtexec`) | NVIDIA `ubuntu2404/x86_64` apt repo |
| Compiler | gcc/g++ 13.3 | Ubuntu |
| CMake | Ubuntu 3.28.x (≥ 3.24 required by the root `CMakeLists.txt`) | Ubuntu |
| OpenCV | Ubuntu `libopencv-dev` 4.6 (CPU decode is enough through G2, C§7) | Ubuntu |
| CUDA arch | `CMAKE_CUDA_ARCHITECTURES=86` | — |

All packages are held (`apt-mark hold`) so an upgrade cannot change the stack under a measured result.

## Alternatives
- CUDA 13.4 + TensorRT 11.4: newer, but above the driver's 13.3; it would rely on minor-version compatibility or
  a driver update. Rejected for G0; revisit with a driver update.
- CUDA 12.9 + TensorRT 11.4 (`+cuda12.9`): supported, but an older major with no gain for SM 8.6. Rejected.
- TensorRT 10.x: spec C§5 assumes TensorRT 11's strongly typed precision workflow. Rejected.
- Native Ubuntu dual-boot: no Windows install to keep, more disruption to the owner. Rejected.

## Consequences
- Step 4 adds the two NVIDIA apt repos and installs exactly the packages above (plan D5); nothing else is upgraded.
- Every engine manifest and benchmark records these versions (C§5, C§10); an engine built here is valid only for
  this GPU + TensorRT pair.
- If CMake 3.28 cannot configure CUDA 13.3, install a newer CMake from Kitware's apt repo and amend this ADR.

## Revisit when
The driver is updated past 13.3, the smoke test fails, TensorRT 11.2 cannot parse the pinned RT-DETR ONNX (G1),
or the host changes.
