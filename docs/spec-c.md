# MotionMan Release C spec: native NVIDIA engine (G0–G6)
Condensed from `docs/reference/native-architecture.md` (cited as NA§N). The reference wins on any disagreement.
Extends `docs/spec-a.md` and `docs/spec-b.md`; every A§ and B§ contract still holds for saved runs.
Grep for the `C§N` you need; do not read whole.

## C§1 Goal and boundary
Replace the per-frame runtime with a C++ executable, `motionman-native`, that genuinely runs OpenCV, CUDA and
TensorRT (RT-DETR) on the owner's NVIDIA PC. Keep the Python app (UI, inspection, zones, analytics, review and
evaluation) as the consumer of its saved runs (NA§1, NA§2). Selective rewrite of the execution layer, not of the
product. One NVIDIA workstation; whole-run remote execution; bounded concurrency; verified artifacts; measured
claims.
**Not in this release (NA§3.4):** multi-GPU, distributed inference, Kubernetes/Kafka/Redis, cloud autoscaling,
public endpoints, multi-tenant access, training, custom TensorRT plugins (unless the model needs one),
hand-written kernels without a measured purpose, RTSP, VFR, frame dropping, cross-camera re-ID, tracker-state
recovery after a crash, automatic driver installs or network exposure without the owner's approval.

## C§2 Phases and gates (NA§16)
| Gate | Exit evidence |
|------|---------------|
| G0 Hardware and baseline | PC inventory, one pinned supported stack, real CUDA smoke test, frozen reference fixtures |
| G1 Model feasibility | Pinned RT-DETR export, TensorRT FP32 engine loaded from C++, parity with the FP32 reference |
| G2 Native serial slice | One video, batch 1, one slot: native artifacts pass the existing Python readers and validators |
| G3 FP16 and GPU safety | Mixed-precision engine within quality budget; buffer-lifetime and error-propagation tests |
| G4 Async multistream | Ordered, bounded, fair processing at 2, 4, 8, 16, 25 sources; session manifest |
| G5 Remote integration | Mac UI submits, reconnects, cancels and imports verified results without manual paths |
| G6 Evidence and release | Paired benchmarks, labelled evaluation, claim table (met / unmet / inconclusive) |
One gate at a time; each has its own plan and stops for the owner's go-ahead. Re-estimate after G1.

## C§3 Shared rules (all gates)
- No silent CPU fallback in native acceptance or benchmark runs: a missing device or runtime fails visibly.
- A GPU gate is never marked passed from a mock, a skipped test or the Mac. GPU checks run on the PC and record
  the actual host; a skipped GPU test is not a pass (INV-7).
- `pytest` and the Mac `ctest` preset need no GPU, network or weights (INV-8). GPU tests are labelled `gpu`
  and run only on the PC.
- Configuration is resolved once, before launch, into canonical JSON (A§2 rules); the worker receives those
  exact bytes and never re-hashes its own serialization. A running worker never changes model, size, precision,
  batching, thresholds or output policy (NA§4.4, NA§10.3).
- Model export and engine builds are offline steps, never inside a timed run (NA§5.1).
- Version control holds code, configs and manifests only. Footage, weights, ONNX, engines and build trees stay
  out of git.
- Every native result carries backend, model, precision, engine hash, host and evidence status, so a TensorRT run
  is never indistinguishable from the Python reference (NA§8.3).

## C§4 Development environment (NA§9)
- Mac: editor, browser, Python UI, portable C++ (CMake preset `mac-cpu`, no CUDA discovery).
- PC: native Linux or WSL2, chosen at G0 after the inventory. Under WSL2 the Windows NVIDIA driver is used; never
  install a Linux GPU driver inside WSL. CMake preset `linux-cuda-release` fails if CUDA or TensorRT is missing.
- Access: authenticated SSH with a verified host key. No unauthenticated server and no public exposure.
- Synchronisation: git commits are the unit. No bidirectional folder sync. A run records the executed commit and,
  for a deliberate dirty experiment, a dirty-tree digest.
- Toolkit choice: the supported intersection of GPU compute capability, driver, CUDA, TensorRT, compiler and OS,
  frozen in `docs/environment.md`. `nvidia-smi`'s CUDA line does not prove `nvcc` exists.

## C§5 Model contract (NA§5)
- Detector: one original RT-DETR implementation (lyuwenyu/RT-DETR), one checkpoint (R18-class first), one pinned
  upstream revision. No mixing with an unrelated exporter of the same name; no YOLO behind a renamed backend.
- `configs/native/<profile>.json` states intended settings. A generated `manifest.json` next to the cached model
  records actual artifacts: checkpoint SHA-256, upstream commit, config, class vocabulary, exporter revision,
  ONNX opset, tensor names, dtypes and shapes, precision policy, preprocessing revision, licence.
- Export keeps the upstream deploy postprocessor: inputs `images` and `orig_target_sizes` (width, height); outputs
  `labels`, `boxes` (xyxy, source pixels), `scores`. Verify against the pinned export; never add NMS, sigmoid,
  top-k or scaling a second time. The application owns the label vocabulary.
- Preprocessing starts at fixed 640×640, batch 1, direct resize (no letterbox), RGB, NCHW. Interpolation,
  scaling, normalisation and rounding are frozen in a machine-readable contract and checked by golden tensors.
- Engine identity: checkpoint and ONNX hashes, exporter and preprocessing revisions, GPU, TensorRT and CUDA
  versions, profiles, precision, plugins, build flags. Any mismatch requires an explicit rebuild.
- Precision follows the pinned TensorRT version's documented workflow (strongly typed in TensorRT 11; no copied
  `--fp16` recipes). The FP32 reference's TF32 policy is recorded and held constant.

## C§6 Model acceptance gates (NA§5.4, project policy, not vendor guarantees)
Order: pinned PyTorch reference (R0) → ONNX Runtime → TensorRT FP32 → TensorRT mixed precision, same images and
transform contract, detections compared by class and geometric matching, not array position.
| Comparison | Gate |
|------------|------|
| Layout conversion, no resize | max normalised error 1e-6 (FP32), 5e-4 (FP16) |
| Matching resize implementations | MAE ≤ 1/255, p99 ≤ 3/255 |
| TensorRT FP32 detections | ≥ 99% same-class matches at IoU 0.99; score error ≤ 0.005 |
| Mixed-precision detections | ≥ 99% same-class matches at IoU 0.95; score error ≤ 0.02 |
Zero non-finite accepted outputs. Near-threshold and tiny-object exceptions are reported, not dropped. A failure
is investigated before any tolerance changes; tolerances are frozen before a candidate is evaluated.

## C§7 Native runtime (NA§6, NA§7)
- One headless process owns decode, scheduling, preprocessing, inference, tracking, rendering and record writing.
  No per-frame Python callback, RPC or tensor shuttling.
- Input: a resolved session config file. Output: the existing staged artifact layout plus a completion receipt.
  Structured status records on one stream, human logs on another. Python finalises and publishes; native exit
  alone is never success (NA§6.1).
- G2 starts with one context, one non-default CUDA stream, batch 1, fixed shape, CPU OpenCV decode.
- Slot states `FREE → FILLING → SUBMITTED → GPU_COMPLETE → CPU_CONSUMING → FREE`, with a generation number;
  stale completions are rejected. Output is read only after its completion event (NA§7.2).
- Every queue is bounded by count and bytes; offline mode blocks, never drops. At most one admitted inference
  frame per source until tracking acknowledges it (no reordering in G2–G4).
- Owned, move-only frame and buffer handles; no borrowed pointer crosses an async boundary. Device and pinned
  pools are allocated at startup; no per-frame allocation or global device sync.
- A fatal CUDA error stops submissions, records partial status and ends the worker; recovery is a new process.

## C§8 Tracking and records (NA§8, NA§10)
- Native ByteTrack adaptation, one tracker per stream with its own ID allocator and class-aware association.
  Upstream C++ is reference material (licence preserved), not a drop-in: no shared static ID counter.
- Tracker parity is checked by replaying identical cached detections into the Python reference and the native
  adapter, compared after canonical ID renaming.
- Each stream is one child run whose `tracks.jsonl` keeps record schema 1.0 exactly: contiguous frame indices,
  `source_t_us` from index and FPS (golden fixtures fix the rounding), empty frames recorded, null IDs kept.
- `report.json` gains an explicit schema version before any native report reaches current views. Overlapping
  stage times are never forced into the old sequential "unattributed" arithmetic.
- New identity: `(session_id, stream_id, local_track_id)`. A `session.json` groups child runs.

## C§9 Remote execution (NA§11, NA§13; G5)
Whole-job operations only: `submit`, `status`, `cancel`, `fetch_result`. A durable agent on the PC owns jobs,
not SSH or Streamlit. Submission is idempotent by random `request_id` plus request digest; a timeout never
triggers an automatic resubmit. Imports are lossless (original bytes, not the debug bundle), staged, validated
(paths, hashes, schema, IDs, nonce) and published atomically. A cancelled child keeps legacy `INTERRUPTED`.

## C§10 Evidence and resume claims (NA§3.3, NA§15)
Targets, not results: 40+ FPS across 25+ streams, 45% lower latency, 1.8× throughput, 35% fewer identity
switches under occlusion, RT-DETR FP16. Each is published as **met, unmet or inconclusive** with raw trials.
- Aggregate FPS = completed frames across sources ÷ one shared processing interval; per-stream FPS always shown.
  40 FPS across 25 streams is 1.6 FPS per stream and is never written as per-camera speed.
- Latency reduction and speedup compare the same statistic, boundary, checkpoint, GPU and input. Mac CPU YOLO11n
  (H0) is never the denominator of a TensorRT acceleration claim.
- Baseline ladder: H0 Python YOLO11n (Mac), R0 RT-DETR FP32 reference (PC), N0 native FP32, N1 native FP16,
  N2 N1 plus one optimisation at a time, N3 async multistream.
- At least 5 paired, alternating trials; median, range, p50/p95; every trial kept, failures included.
- Identity-switch claims need TrackEval on labelled, contiguous, held-out sequences with absolute IDSW, IDF1,
  HOTA, FN and fragmentation. P4's isolated reviewed frames do not establish identity continuity.

## C§11 ADRs for this release (NA§4.3)
Approved by the owner 2026-11-04. Recorded so far: ADR-001 native backend boundary, ADR-002 evidence policy (both Accepted), ADR-003 RT-DETR model contract
(Accepted at G1), ADR-004 tracker ownership and INV-1 change (Proposed until G4), ADR-005 PC toolkit (Accepted
at G0). Still to record at their gate: child-run schema compatibility (G2), slot and scheduling
lifetime (G4), durable remote job protocol (G5).
