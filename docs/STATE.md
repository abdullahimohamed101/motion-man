# State
<!-- Rewritten (not appended) at the end of every session. Cap: 60 lines. History lives in git log and done plans. -->

Updated: 2026-11-07

## Now
- Releases A and B part 1 (P3, P4) merged (PRs #6–#13). Release C, the native C++/CUDA/TensorRT/RT-DETR engine,
  is in progress (`docs/spec-c.md`, plan `docs/plans/native-g0-g1.md`).
- **G0, G1 and G2 complete; G2 gate reached 2026-11-07, waiting for the owner.** RT-DETRv2-S runs in TensorRT FP32 from C++
  (`motionman-native infer`) with bit-exact preprocessing and parity PASS vs PyTorch (852/852, max |Δscore|
  1.5e-4); mismatched engines refused. NG-1..NG-11 all recorded. ADR-003 Accepted.
- Re-estimate (plan `native-g2.md` § G2 result): G3–G6 13–23 focused days, 16–29 with contingency. G4 raised
  (CPU preprocessing and drawing are as slow as inference, so they need overlapping).
- PRs: #14–#16 merged 2026-11-07, but #16 (G1) merged into `feat/native-g0` after #15 had already merged, so
  **G1 never reached `main`**. Re-landed as its own PR `feat/native-g1` → `main`; the G2 PR also targets `main`
  (its diff includes G1 until the G1 PR merges). One PR per gate; from now on every gate PR targets `main`.
- Plan `docs/plans/native-g2.md`: all 9 steps done, N2-1..N2-12 recorded, ADR-006 Accepted with amendments.
  Branch `feat/native-g2` (off `feat/native-g1`), pushed.
  `motionman-native run` works on the PC GPU (crosswalk 60 frames, deterministic; refusal, no-overwrite, SIGINT).
  `python -m motionman --backend native` publishes on the PC (crosswalk, 1356 frames, H.264; analyze, evaluate
  and the Mac UI read it; Ctrl-C leaves it unpublished). Real replay parity on the full crosswalk clip: exact.
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
  App env `~/motion-man/.venv` (Python 3.12.3, CPU torch, pinned requirements-dev). No ffmpeg. Replay dump and
  fixture in `~/.cache/motionman/replay` (not in git).
- The owner games on this PC: check `tasklist` for Valorant before GPU work; stop at once if asked
  (`wsl --shutdown`).

## Next step (exact)
1. **G2 gate: wait for the owner.** They review and merge the G1 re-land PR and the G2 PR (both → `main`) and
   decide the two gate notes: low-score clutter in native video (admission > 0.1, D6) and the UI not showing
   the backend. Then, with their go-ahead, plan G3 (FP16; spec C§11 and `docs/spec-c.md`).

## Blocked / needs owner
- Merging the G1 re-land PR and the G2 PR into `main`.
- Off-network PC access (Tailscale or not): deferred by the owner, needed by G5.
- Optional: fix the frame 60 box in `crosswak-eval` and commit `r3`.
- Side request: portfolio website update; waiting for its folder/repo and what to change.

## Last check
- PC `ctest --preset linux-cuda-release`: 13/13 (7 unit, golden, cuda_smoke, trt_infer, trt_parity,
  trt_refuse_tampered, native_run_smoke) (G2 step 8).
- Mac `scripts/check.sh`: ruff ok, 583 passed; `ctest --preset mac-cpu` 7/7 (G2 step 8).
- Benchmarks (crosswalk, M5 Pro CPU): P2 `benchmark` on battery 45.86 fps (imgsz 640), 54.21 (imgsz 480).
