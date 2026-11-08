# MotionMan Release A spec (P0–P2)
Condensed from `docs/reference/master-design.txt` (v1.1, cited as MD§N) and `docs/reference/ui-reference-v2.txt`
(UI2). The reference files win on any disagreement. `docs/spec.md` remains the MVP contract; this file extends it.
Grep for the `A§N` you need; do not read whole.

## A§1 Goal and boundary
A local workbench: run a clip, inspect evidence, compare a controlled change, keep a reproducible case (MD§1).
Release A = P0 verify and formalize, P1 inspect, P2 measure. **Not in Release A:** zones, events, heatmaps
(P3), review and evaluation (P4), experiments (P5), job queue, cancel and derived media exports (P6), masks and
webcam (P7), TensorRT (P8), new models, a custom player, accounts, any network service (MD§19).
After each phase gate, stop for the owner's go-ahead (MD§25).

## A§2 Identities and storage (MD§4)
- `input_id` = SHA-256 of the managed source bytes; the filename is only a display label.
- `run_id` = one execution attempt; benchmark repeats are separate runs. Never cache inference implicitly.
- Data root layout: `inputs/<input_id>/` (managed source plus metadata), `staging/<run_id>/` (never listed as
  results), `runs/<run_id>/` (published, immutable). JSON/JSONL is the source of truth; no database.
- Analysis/derived keys hash canonical JSON (sorted keys, fixed separators, no NaN/Inf, explicit units).
- Keep the MVP `--output-dir` meaning. Legacy outputs stay readable through adapters; never fabricate hashes.

## A§3 RunSpec and manifest (MD§5)
- Validate the full request before expensive work; resolve defaults once and persist the resolved form:
  input (id, path, dims, timebase, frame range), model (weights id and hash, class-map hash), tracker (type,
  resolved config, hash), inference (device requested/actual, imgsz, conf/IoU, class filter, precision),
  rendering (codec, size/FPS, trails, audio), reproducibility (code revision, dependency lock hash, env), limits.
  RunSpec and report 1.1 add a `backend` block (name, precision; for native also profile and hashes, ADR-006); a
  1.0 RunSpec or report has none and means the legacy Ultralytics FP32 backend.
- Terminal manifest: schema_version, run_id, status, input_id, config_sha256, records_sha256, UTC start/finish,
  counts (decoded, inferred, written), artifacts (path, bytes, sha256 each), validation (records, video_decode),
  capabilities. Unknown schema versions fail clearly. Missing measurements are null with a reason, never 0.

## A§4 Frame records (MD§6)
Per decoded frame: schema_version, run_id, source_frame_index (0-based), source_t_us
(= round(i × 1e6 / fps) for validated CFR), sample_index, frame_status (`INFERRED`), source_width/height,
objects[] with observation_id (`"<frame>:<n>"`), track_id (int|null), class_id, class_name, confidence,
bbox_xyxy (source pixels, finite, positive extent, in bounds), mask_ref null. Empty frames are `objects: []`;
a decode failure is a run error, never a manufactured empty frame.

## A§5 Lifecycle and publication (MD§7)
States RUNNING → FINALIZING → SUCCEEDED | FAILED | INTERRUPTED. One exclusive execution lease per data root.
Publication: new staging dir → stream outputs → close and validate (order, counts, schema, decoded video) →
hash artifacts, write manifest → atomic rename into `runs/`. Readers list only published runs. Session state
holds references only; completed runs are discoverable from disk.

## A§6 Media envelope and limits (MD§8)
Short CFR MP4, positive FPS, even dims. Positive FPS is not proof of CFR: validate frame timestamps or reject
with an actionable message. Defaults: upload 512 MiB; ≤10 min, ≤1920×1080, ≤36,000 frames; one heavy job;
inspector cache 8 frames (LRU); trails 30 points; output budget 2 GiB; UI progress ≤2 Hz. Record effective
limits per run; hitting a limit is a visible failure, not silent truncation.

## A§7 UI (MD§9, UI2)
- Journey: Choose video / Try demo → Run tracking → Inspect → Export. Advanced settings (device, thresholds)
  collapsed with plain-language help. One lime primary action per stage (Run tracking; Export when a result is
  open; New video stays neutral).
- Header: wordmark, source name, saved-run selector, written status. Run IDs, device, versions in Run details.
- Navigation in Release A: Inspect, Performance only. Unsupported sections hidden.
- Settings contract: view options re-render only; tracking settings say "These settings apply to your next
  run". Opening a run, moving a slider or changing tabs never runs the model. Selections restore per run;
  stale IDs from another run are cleared.
- Copy and states per UI2 "Plain-language copy" and "Component and state rules". No cancel button before P6.

## A§8 P1 Inspect (MD§11, MD§20)
Run shelf (validated published runs, plus imported legacy runs). Sparse frame-record index every 128 records,
bound to the records hash. Exact-frame slider plus annotated still, separate from playback ("never pretend they
are synchronized"). Object selector: class, ID, frame/time, observed intervals; on demand: confidence summary,
gaps, trajectory, selected-frame crops, bounded contact sheet. Spotlight, trail length/opacity, display filters,
evidence PNG. Explicit presets that keep rendering, inference and analysis settings separate.
Gate: a fake backend that raises if called proves inspection never infers; still, record and PNG agree.

## A§9 P2 Measure (MD§12, MD§17, MD§21)
Stage timings: setup_s, decode/detect_track/render/write ms, finalize_s, job_wall_s; per-frame rows kept;
warmup policy persisted. Three-repeat benchmark orchestration with median and range; a comparison validator
that names intended changes and flags confounders. `python -m motionman doctor`: read-only checks (model load,
device, decode, encode, writable storage, capacity, timing support) plus labelled input-quality hints.
Optional telemetry about 1 Hz (process RAM; GPU only where supported, else Unavailable). Sanitized debug
bundle ZIP from an allowlist with a hashed manifest; footage and full paths opt-in.
Gate: repeats never reuse inference; unsupported metrics show Unavailable; bundles reopen and leak nothing.

## A§10 Security and tests (MD§17, MD§18)
Loopback only. Filenames are display data; managed paths come from validated IDs; reject traversal and
symlink escape; allowlisted config; argument arrays for subprocesses. Offline CI: unit, schema, fake-backend
integration, lint; no weights or GPU. Real-model and hardware suites are separate and report SKIPPED, not pass.
