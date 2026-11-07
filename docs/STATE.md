# State
<!-- Rewritten (not appended) at the end of every session. Cap: 60 lines. History lives in git log and done plans. -->

Updated: 2026-11-07

## Now
- Releases A and B part 1 (P3, P4) merged (PRs #6–#13). Release C, the native C++/CUDA/TensorRT/RT-DETR engine,
  is in progress (`docs/spec-c.md`, plan `docs/plans/native-g0-g1.md`).
- G0 complete (steps 1–5). **G1 in progress: steps 6–7 done** (RT-DETRv2-S ONNX + manifest, R0, bit-exact C++
  preprocessing, ONNX Runtime parity PASS). Branch `feat/native-g0` (off `feat/native-plan`; neither pushed).
  NG-1..NG-8 filled.
- PC: Windows 11 + WSL2 Ubuntu 24.04, RTX 3060 Ti 8 GB, driver 610.88. Stack per ADR-005 (Accepted): CUDA 13.3.1,
  TensorRT 11.2.1.2, CMake 3.28, OpenCV 4.6; 85 packages held. Details: `docs/environment.md`.
- Access: `ssh motionman-pc` from the Mac (key auth, Windows OpenSSH); Linux via
  `ssh motionman-pc "wsl -d Ubuntu-24.04 -u root -- bash -s" < script` (WSL default user is root).
- Code on the PC: `~/motion-man` in WSL, cloned from a git bundle of `feat/native-g0` (not GitHub).
- The owner games on this PC: check `tasklist` for Valorant before GPU work; stop at once if asked
  (`wsl --shutdown`).

## Next step (exact)
1. G1 step 8: `tools/model_export/build_engine.py` (D6, trtexec) → TensorRT FP32 engine + `engine_manifest.json`
   with full identity (C§5); a mismatched manifest refuses to load. Then step 9 (C++ TensorRT runner + parity).
   Preprocessing is done: C++ `resize_bilinear_pillow` is bit-exact with the Python reference.
- PC caches: venv `~/.cache/motionman/venv`, upstream `~/.cache/motionman/src/RT-DETR`, golden PNGs
  `~/.cache/motionman/golden`, model dir `~/.cache/motionman/models/rtdetr_r18_640` (checkpoint, ONNX, manifest, r0).

## Blocked / needs owner
- Optional: fix the frame 60 box in `crosswak-eval` and commit `r3`.
- Side request: portfolio website update; waiting for its folder/repo and what to change.

## Last check
- PC `ctest --preset linux-cuda-release`: 5/5 (incl. preprocess_golden, cuda_smoke); manifest ok; ORT parity PASS.
- Mac `scripts/check.sh`: ruff ok, 533 passed; `ctest --preset mac-cpu` 3/3.
- Benchmarks (crosswalk, M5 Pro CPU): P2 `benchmark` on battery 45.86 fps (imgsz 640), 54.21 (imgsz 480).
