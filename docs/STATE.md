# State
<!-- Rewritten (not appended) at the end of every session. Cap: 60 lines. History lives in git log and done plans. -->

Updated: 2026-11-07

## Now
- Releases A and B part 1 (P3, P4) merged (PRs #6–#13). Release C, the native C++/CUDA/TensorRT/RT-DETR engine,
  is in progress (`docs/spec-c.md`, plan `docs/plans/native-g0-g1.md`).
- **G0 complete, at the G0 gate (stop for the owner).** Branch `feat/native-g0` (off `feat/native-plan`; neither
  pushed). Steps 1–5 done; NG-1..NG-5 filled.
- PC: Windows 11 + WSL2 Ubuntu 24.04, RTX 3060 Ti 8 GB, driver 610.88. Stack per ADR-005 (Accepted): CUDA 13.3.1,
  TensorRT 11.2.1.2, CMake 3.28, OpenCV 4.6; 85 packages held. Details: `docs/environment.md`.
- Access: `ssh motionman-pc` from the Mac (key auth, Windows OpenSSH); Linux via
  `ssh motionman-pc "wsl -d Ubuntu-24.04 -u root -- bash -s" < script` (WSL default user is root).
- Code on the PC: `~/motion-man` in WSL, cloned from a git bundle of `feat/native-g0` (not GitHub).
- The owner games on this PC: check `tasklist` for Valorant before GPU work; stop at once if asked
  (`wsl --shutdown`).

## Next step (exact)
1. Owner reviews G0 and says go for G1.
2. G1 step 6: clone the pinned RT-DETR repo and checkpoint on the PC (download needs the owner's yes), ONNX export,
   manifest, R0 reference (needs Python + PyTorch in WSL; not installed yet).

## Blocked / needs owner
- G0 gate go-ahead; RT-DETR repo + checkpoint download approval.
- Optional: fix the frame 60 box in `crosswak-eval` and commit `r3`.
- Side request: portfolio website update; waiting for its folder/repo and what to change.

## Last check
- PC `ctest --preset linux-cuda-release`: 3/3 passed (cuda_smoke on the RTX 3060 Ti).
- Mac `scripts/check.sh`: ruff ok, 525 passed; `ctest --preset mac-cpu` 2/2.
- Benchmarks (crosswalk, M5 Pro CPU): P2 `benchmark` on battery 45.86 fps (imgsz 640), 54.21 (imgsz 480).
