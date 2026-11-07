# Environment: the NVIDIA execution host (spec C§4, plan native-g0-g1 step 3)
Every value here was captured on the PC itself (owner's screenshots / pasted output, or over SSH from the Mac),
never inferred on the Mac. Missing values are listed as unknown, not guessed.

## Access
`ssh motionman-pc` from the Mac (key `~/.ssh/motionman_pc`, Windows OpenSSH Server, LAN 192.168.1.68, user
`ahmed`). Linux commands: `ssh motionman-pc "wsl -d Ubuntu-24.04 -- bash -s" < script` (cmd.exe mangles pipes and
quotes, so send the script on stdin).

## Inventory (2026-11-07, over SSH)
| Item | Value | Source |
|------|-------|--------|
| GPU | NVIDIA GeForce RTX 3060 Ti (Ampere; compute capability 8.6 per NVIDIA's published table) | `nvidia-smi` |
| VRAM | 8192 MiB total; 4405–5217 MiB in use by desktop apps / games at capture | `nvidia-smi` |
| Driver | 610.88 (Windows, WDDM); same driver seen inside WSL via `/usr/lib/wsl/lib/libcuda.so` | `nvidia-smi` |
| Highest CUDA the driver supports | 13.3 (`CUDA UMD Version`); not proof a toolkit is installed | `nvidia-smi` |
| Power | 200 W cap | `nvidia-smi` |
| CPU / RAM | Intel i7-12700F, 20 threads; 15.8 GB RAM (WSL sees 7 GB by default) | CIM, `nproc`, `free` |
| Windows | Windows 11 Home, build 10.0.26200.9711 | `ver`, CIM |
| WSL | WSL2, kernel 6.18.33.2-microsoft-standard-WSL2; distro Ubuntu-24.04 (24.04.4 LTS) | `wsl -l -v`, `uname` |
| WSL default user | root (no Linux user created yet) | `whoami` |
| Compiler | gcc 13.3.0 | `gcc --version` |
| `nvcc`, CMake, OpenCV, TensorRT | not installed | `command not found` |
| Disk free | C: 86.9 GB (after the owner removed a game; was 0.1 GB); D: 123.2 GB; WSL root VHD 941 GB free | `Get-PSDrive`, `df` |

## Consequences
- Windows host → the execution environment is WSL2 + Ubuntu 24.04 with the Windows NVIDIA driver (C§4; NVIDIA CUDA
  on WSL guide). Never install a Linux GPU driver inside WSL; install the WSL-Ubuntu CUDA toolkit only.
- 8 GB VRAM bounds engine size, batch and stream count; G4 admission must measure real free memory (C§7).
- Benchmarks must record competing GPU workloads; desktop apps and games used 4–5 GB at capture (C§10).
- WSL's 7 GB RAM default may be tight for TensorRT engine builds; raise it in `.wslconfig` only if a build fails.
- Toolkit versions: ADR-005 (CUDA 13.3.1, TensorRT 11.2.1.2+cuda13.3, gcc 13.3, CMake 3.28, OpenCV 4.6), pending
  the step 4 install and the step 5 smoke test.
