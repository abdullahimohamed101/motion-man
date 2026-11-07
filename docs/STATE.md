# State
<!-- Rewritten (not appended) at the end of every session. Cap: 60 lines. History lives in git log and done plans. -->

Updated: 2026-11-07

## Now
- Releases A and B part 1 (P3, P4) merged (PRs #6–#13). Release C, the native C++/CUDA/TensorRT/RT-DETR engine,
  is in progress (`docs/spec-c.md`, plan `docs/plans/native-g0-g1.md`).
- **G0 and G1 complete; the owner approved planning G2.** RT-DETRv2-S runs in TensorRT FP32 from C++
  (`motionman-native infer`) with bit-exact preprocessing and parity PASS vs PyTorch (852/852, max |Δscore|
  1.5e-4); mismatched engines refused. NG-1..NG-11 all recorded. ADR-003 Accepted.
- Re-estimate (plan § G1 result): G2–G6 16–28 focused days, 20–35 with contingency. Raised: G2 (ByteTrack port),
  G3 (TensorRT 11 is strongly typed, so FP16 likely needs a mixed-precision ONNX export).
- One PR per gate, stacked; the owner merges in order: #14 spec + plan (`feat/native-plan` → main), #15 G0
  (`feat/native-g0`), #16 G1 (`feat/native-g1`). Each PR's base is the one before it.
- **Active plan: `docs/plans/native-g2.md` (Active, steps 1–5 done)** on `feat/native-g2` (off `feat/native-g1`, not pushed).
  `motionman-native run` works on the PC GPU (crosswalk 60 frames, deterministic; refusal, no-overwrite, SIGINT).
  ADR-006 Proposed (native run contract, backend metadata, tracker semantics).
- PC: Windows 11 + WSL2 Ubuntu 24.04, RTX 3060 Ti 8 GB, driver 610.88. Stack per ADR-005 (Accepted): CUDA 13.3.1,
  TensorRT 11.2.1.2, CMake 3.28, OpenCV 4.6; 85 packages held. Details: `docs/environment.md`.
- Access: `ssh motionman-pc` from the Mac on the home LAN only (key auth, Windows OpenSSH); Linux via
  `ssh motionman-pc "wsl -d Ubuntu-24.04 -u root -- bash -s" < script`. Use `</dev/null` for child scripts
  inside that heredoc; copy files with `COPYFILE_DISABLE=1 tar --no-xattrs` (no `._*` files).
- Code on the PC: `~/motion-man` in WSL on `feat/native-g2`, synced from a git bundle sent over ssh stdin (not
  GitHub). WSL clears `/tmp` when it restarts between calls.
- PC caches: venv `~/.cache/motionman/venv`, upstream `~/.cache/motionman/src/RT-DETR`, golden PNGs
  `~/.cache/motionman/golden`, model dir `~/.cache/motionman/models/rtdetr_r18_640` (checkpoint, ONNX,
  manifests, r0, FP32 engine, trt_fp32), clip `~/.cache/motionman/clips/crosswalk.mp4` (same sha256 as the Mac's).
  No ffmpeg and no Python OpenCV on the PC yet.
- The owner games on this PC: check `tasklist` for Valorant before GPU work; stop at once if asked
  (`wsl --shutdown`).

## Next step (exact)
1. G2 step 6 [Mac]: `motionman/native.py` controller and `--backend native` (plan step 6), pytest with a fake
   native executable for the six failure cases. Contract to honour is in the plan's step 5 log line (runspec
   `tracking.class_policy`, `backend.{profile, engine_sha256}`, codec `mp4v`; receipt fields; exit codes).
   Step 7 needs the owner's yes to install the Python app env on the PC (download).

## Blocked / needs owner
- Merging PRs #14 → #15 → #16.
- Off-network PC access (Tailscale or not): deferred by the owner, needed by G5.
- Optional: fix the frame 60 box in `crosswak-eval` and commit `r3`.
- Side request: portfolio website update; waiting for its folder/repo and what to change.

## Last check
- PC `ctest --preset linux-cuda-release`: 13/13 (7 unit, golden, cuda_smoke, trt_infer, trt_parity,
  trt_refuse_tampered, native_run_smoke) (G2 step 5).
- Mac `scripts/check.sh`: ruff ok, 568 passed; `ctest --preset mac-cpu` 7/7 (G2 step 5).
- Benchmarks (crosswalk, M5 Pro CPU): P2 `benchmark` on battery 45.86 fps (imgsz 640), 54.21 (imgsz 480).
