# P2 Measure: stage timings, benchmarks, doctor and debug bundles
Status: Active · Branch: `feat/p2-measure` · Spec: A§9, A§10 (MD§12, MD§17, MD§21)

## Outcome
Every run records where its time went (setup, decode, detect+track, render, write, finalize) with per-frame
rows and a persisted warmup policy. `benchmark` repeats a configuration with fresh inference and reports
median and range. `compare` names intended changes and flags confounders. `doctor` diagnoses the environment
read-only, with labelled input-quality hints. `bundle` exports a sanitized, hash-verified ZIP that reopens.
Release A is then complete.

## Non-goals
- No sampling or frame skipping (every-frame baseline only), no rendering-disabled mode, no overlap/pipelining.
- No MPS device option, no CUDA work on this host (CUDA timing stays UNVERIFIED), no auto-install or tuning.
- No UI for benchmark, compare, doctor or bundle (CLI only); the UI gains only the stage breakdown.
- No accuracy claims: comparisons speak about speed and counts, never "better detections".
- No zones, events, heatmaps or review layers (P3+).

## Decisions (confirmed 2026-11-01)
- D1 Stage boundaries, all `perf_counter` wall time per frame: **decode** = time inside `next()` on the
  frame reader (includes the timestamp check); **detect_track** = `tracker.update` as today; **render** =
  `trails.update` + `annotate`; **write** = video `writer.write` + JSONL line write. Stage sums may be less than
  wall time (loop overhead, progress callback); the report shows the remainder as `unattributed_ms`, never
  forced to match.
- D2 Run-level durations: **setup_s** = from entering `_run` to just before the first decode (validation,
  import/hash, model load, writer open); `model_load_s` stays as a part of it. **finalize_s** = from the writer
  release to the report write (output decode check, records validation, hashing). The final atomic rename
  happens after the report and is excluded; the scope string says so. **job_wall_s** = setup + processing +
  finalize.
- D3 Per-frame rows go to `timings.jsonl` in the run folder (one row per processed frame: source_frame_index
  and the four stage ms), streamed like the records, listed in the manifest. Percentiles are computed from
  bounded in-memory float lists (≤36,000 frames × 4 by the input limits; INV-4 is about frames, not floats).
- D4 Warmup policy persisted in `report.timing.warmup`: `{policy: "exclude_first_inferred", frames: 10,
  samples: n}`; too few samples → null with a reason (existing rule, now for every stage).
- D5 `python -m motionman benchmark --source <clip> [--repeats 3] [--device] [--imgsz] [--max-frames]
  [--label]`: N sequential canonical runs, each a fresh `pipeline.run` (new tracker, new run_id; never reads
  a previous run's records). Writes `data/benchmarks/<id>/benchmark.json`: settings, per-trial run_id, status,
  fps, stage p50/p95; median and min–max fps; failed trials stay listed and are excluded from the median with a
  count. Records power state (`psutil.sensors_battery()`: AC / battery / unknown) per trial.
- D6 `python -m motionman compare <benchA> <benchB> [--intended imgsz,device]`: checks input hash, frame
  policy, sampling, output policy, warmup, model weights hash, imgsz, conf, iou, precision, device, package
  versions, CPU/GPU, platform and power state. Differences not listed in `--intended` are **confounders**.
  Prints the median fps change and stage deltas, labelled as speed only; writes `comparison.json`. Exit 0
  always: a confounded comparison is information, not a crash.
- D7 `python -m motionman doctor [--source <clip>] [--device] [--data-root]`: read-only checks with
  PASS / WARN / FAIL / UNAVAILABLE and a next step each: Python and pinned packages, weights present and model
  load (timed), device (cpu always; cuda/mps reported as available or Unavailable), decode (the given source,
  or a 1 s clip generated in an app-owned temp folder), encode (writes, decodes, removes only its own probe),
  data root writable, free space vs the output budget, timer resolution. Exit 0 if no FAIL, else 1.
- D8 Input-quality hints (doctor `--source` only): sample up to 60 evenly spaced frames; mean luma < 40 →
  "possibly dark"; median Laplacian variance < 50 → "possibly blurry"; > 20% identical consecutive samples →
  "possible repeated frames". Always worded as hints, never as faults; thresholds documented in README.
- D9 Telemetry: opt-in `--telemetry` on the run and benchmark commands. A daemon thread samples process RSS
  (psutil) about once per second into `telemetry.jsonl` (bounded by the 10-minute input limit); the report gets
  peak and median RSS. GPU memory/utilization only via NVML when importable and CUDA is used, else
  `"Unavailable"` with a reason. The thread never touches the tracker (INV-1).
- D10 `python -m motionman bundle --run <run_id> [--include-footage] [--include-paths]` →
  `data/exports/bundle-<id>/bundle.zip`. Allowlist: manifest, runspec, report, timings.jsonl, telemetry.jsonl,
  tracks.jsonl, a fresh doctor result (without model load), and README.txt with reproduce commands; plus
  `bundle_manifest.json` with each entry's sha256 and bytes. Absolute paths and the home folder are replaced
  with `<redacted>/<basename>` unless `--include-paths`. Footage (source, annotated.mp4) only with
  `--include-footage`. No environment variables ever. `bundle --verify <zip>` re-hashes every entry and
  refuses unknown or traversing names.
- D11 CLI: the existing `python -m motionman --source …` stays unchanged; a first argument of `doctor`,
  `benchmark`, `compare` or `bundle` dispatches to that subcommand. Exit codes reuse 0/1/2/75/130.
- D12 UI: the Performance tab adds a stage breakdown (p50 per stage and share of wall time), setup, finalize and
  job wall. Runs from before P2 show "Not recorded in this run" for these, never zeros.

## Steps
1. [x] Stage timings (D1–D4): pipeline instrumentation, `timings.jsonl` artifact in the manifest, report
   timing fields + warmup policy. Tests: rows = frames, stage sums ≤ wall, short-run nulls with reasons,
   failed runs still report what was measured.
2. [x] UI Performance breakdown (D12) and README timing definitions; one real crosswalk run to name the
   measured bottleneck.
3. [x] Telemetry (D9) with a fake sampler/clock in tests; Unavailable paths tested.
4. [x] `benchmark` (D5). Tests with the fake tracker: N tracker constructions, N distinct run_ids, a failing
   trial kept and excluded from the median.
5. [x] `compare` (D6). Tests: identical configs → no confounders; an intended change is not flagged; an
   undeclared change (power state, package version) is.
6. [x] `doctor` + input-quality hints (D7, D8). Tests with injected model loader and synthetic dark / blurry /
   repeated clips; probe files removed; nothing outside its temp folder touched.
7. [x] `bundle` + `--verify` (D10). Tests: allowlist only, footage opt-in, no absolute paths, home folder or
   env vars in any entry, tamper and traversal refused, bundle reopens.
8. [x] Real verification and docs: 3-repeat benchmark, one controlled comparison, doctor on this host (and
   a failing check shown), bundle exported and verified; README, `docs/interview.md`, `docs/demo-script.md`.

## Acceptance
| # | Criterion | Command | Result |
|---|-----------|---------|--------|
| P2-1 | Offline gate passes; new suites need no weights, GPU or network | `scripts/check.sh`; CI || Local pass, 355 tests (doctor's torch probe and model loader injected; no weights or network). CI: pending on the P2 PR |
| P2-2 | Every run reports setup, decode, detect_track, render, write, finalize and job wall; rows = frames | test + real run || Pass: test_stages (rows = frames, totals ≤ wall, job = setup + wall + finalize); real runs report all fields (e.g. setup 3.45 s, wall 29.36 s) |
| P2-3 | Warmup policy and sample counts persisted; too few samples → null with a reason | test || Pass: test_stages::test_warmup_policy_and_short_runs_are_explicit; report.timing.warmup = exclude_first_inferred / 10 / samples |
| P2-4 | Benchmark repeats never reuse inference: N trackers, N run_ids, N fresh records hashes on real repeats | test + real || Pass: test_benchmark (3 trackers, 3 run_ids, 3 records hashes); real: 6 trials, 6 distinct run_ids and records hashes |
| P2-5 | Benchmark shows median and range; failed trials visible and excluded from the median | test || Pass: test_benchmark median/range and failed-trial tests; real: 45.86 (43.99–46.19) and 54.21 (50.61–55.16) |
| P2-6 | Compare flags undeclared differences as confounders and never labels accuracy | test + real || Pass: test_compare; real: smoke pair flagged `code`; final pair "Controlled", +18.2%, scope line says speed only |
| P2-7 | Unsupported metrics show Unavailable with a reason (GPU here; old runs in the UI) | test + browser || Pass: GPU "Unavailable" with reason (telemetry, doctor); pre-P2 runs show "—" / Not recorded in the browser (step 2) |
| P2-8 | Doctor is read-only, removes only its own probes, reports a failing check with a next step | test + real || Pass: test_doctor (tree unchanged, probe folder removed, data root not created); real `--device cuda` → FAIL + next step, exit 1 |
| P2-9 | Bundles reopen and verify; tampering refused; footage and full paths only when opted in; no env dump | test + real || Pass: test_bundle (allowlist, redaction, footage opt-in, tamper/traversal refused); real bundle verified, 0 home-path hits, tampered copy exit 1 |
| P2-10 | A measured bottleneck is explained from stage timings in README (real numbers, power state stated) | real run || Pass: README "Where the time goes": detect+track 89.5% of wall; imgsz 480 cuts its p50 by 3.25 ms for +18.2% FPS (battery stated) |

## Risks
- Instrumentation overhead: four extra `perf_counter` pairs per frame (~µs) against ~22 ms frames; measured
  before/after on the same power state, reported.
- Power state dominates CPU numbers (38% swing measured in the MVP): recorded per trial and flagged by
  compare; unknown on hosts without a battery API.
- `psutil.sensors_battery()` and NVML vary by host: both degrade to "unknown"/"Unavailable", never guessed.
- Real benchmarks take ~2 min per 3 repeats on mains; the full step 8 is about 10 minutes of runs.

## Log
<!-- One line per step: date, step, commit, result. No pasted output. -->
- 2026-11-01 step 1: pipeline times decode (inside next()), detect_track, render, write per frame → timings.jsonl (streamed, in the manifest); Metrics gains begin_job/add_frame/finish; report timing adds setup_s, finalize_s, job_wall_s, per-stage p50/p95/total, unattributed remainder, warmup policy + samples, instrumentation mode; old keys kept. 9 tests, 303 total. Real crosswalk (battery 97%, discharging): 44.4 fps; setup 3.33 s (model load 2.99), wall 30.54 s, finalize 0.79 s, job 34.66 s; detect_track 89.5% of wall (p50 19.35 ms), render 3.7%, write 3.7%, decode 2.1%, unattributed 0.9%. Caveat for README: decode/write time the calls only; OpenCV's codec may work on background threads.
- 2026-11-01 step 2: Performance tab gains "Where the time went": Setup / Processing / Finalize / Job total cards and a stage table (p50, p95, share of processing as a bar, unattributed row) with the warmup and codec-thread caveats. view.stage_breakdown / phase_metrics are pure; runs from before P2 show "—" with "Not recorded in this run" as help and a caption (the full phrase was cut off in the cards in the browser). README "Where the time goes" rewritten: field table, warmup rule, lower-bound note for decode/write, measured bottleneck (detect+track 89.5% of wall). 3 tests, 306 total. Browser: new run shows the table with real numbers; baseline 2ad4d6 shows not recorded, no zeros.
- 2026-11-01 step 3: telemetry.py (daemon thread, RSS via psutil about 1 Hz, sample at start and stop, ≤7,200 rows then truncated flag, sampler errors recorded never raised; GPU via NVML only on CUDA, else Unavailable with reason). RunConfig.telemetry (not in config hash), --telemetry flag, telemetry.jsonl in the manifest, report.telemetry ({"enabled": false} when off). psutil 7.2.2 pinned. 8 tests, 314 total. Real crosswalk --telemetry (battery 95%): 33 samples, RSS peak 1,441 MB, median 1,403 MB, GPU Unavailable (cpu); 43.54 fps vs 44.4 untelemetered, within run-to-run noise on battery (not a measured overhead; step 8 compares properly).
- 2026-11-01 step 4: benchmark.py (N sequential canonical runs, fresh tracker each; failed trials recorded and the series continues; Ctrl-C writes the file with status interrupted; summary: median/min/max fps, failed count, config consistency, median stage p50s, power sources; power via psutil.sensors_battery, unknown with reason). CLI subcommand dispatch (`benchmark`; the run command unchanged), exit 1 if any trial failed. 7 tests, 321 total. Real smoke (3 × 240 frames, battery 93%): 42.42 / 43.11 / 37.76 fps → median 42.42, range 37.76–43.11; 3 distinct records hashes; config consistent; detect_track p50 20.3–21.0 ms in all three, so trial 3's drop came from outside the median call time.
- 2026-11-01 step 5: compare.py: 21 factors from each benchmark's first succeeded run (runspec + report + environment + power sources); intended vs confounders; notes for failed trials, inconsistent settings, mixed power, < 3 trials; median fps change %, range overlap, stage p50 deltas; verdict says "Controlled" only with no confounders and no notes; speed-only scope line. CLI `compare A B --intended …` (IDs or paths), writes data/comparisons/<id>/comparison.json, exit 0 (2 on unreadable input). 8 tests, 329 total. Real (3 × 240 frames, battery): imgsz 640 median 42.42 vs imgsz 480 49.09 fps, +15.7%, detect_track p50 −4.06 ms; ranges overlap; code commit flagged as confounder (different commits, dirty tree).
- 2026-11-01 step 6: doctor.py: python, packages vs requirements pins, weights, device (resolve_device + CUDA/MPS availability), model load with warmup (skipped on device FAIL or --no-model), encode/decode probe in a fresh mkdtemp removed in finally, --source preflight via open_reader, storage (no data root created), capacity vs budget, timer; each PASS/WARN/FAIL/UNAVAILABLE + next step; exit 1 on FAIL; --json. Input-quality measurements + hints from 60 sequentially decoded samples (dark < 40 luma, blurry < 50 Laplacian var at 640 px, repeated > 20%). 13 tests, 342 total (device probe and model loader injected: no torch in tests). Real: all PASS (model load 2.28 s, encode probe 4 ms, crosswalk preflight 228 ms; luma 121.4, Laplacian 3,604, 0% repeats → no hints); --device cuda → FAIL with next step, model load skipped, exit 1.
- 2026-11-01 step 7: bundle.py: verified run only (open_run re-hashes); allowlist (manifest, runspec, report, tracks, timings, telemetry, doctor.json without model load, README.txt with reproduce commands) + bundle_manifest.json (bytes + sha256 per entry); redaction: home folder → <redacted> first (a test caught the basename rule keeping the username), other absolute paths → <redacted>/<name>; --include-paths, --include-footage (streamed); verify: flat allowlisted names only, no duplicates, size cap, manifest/entry agreement, streamed re-hash. store.sha256_bytes. CLI `bundle --run` / `--verify` (exit 1 on a bad bundle, 2 on unknown or damaged run). 13 tests, 355 total. Real: bundle of r_20261101T073439Z_625362 = 610 KiB, 8 entries + manifest, verify ok, zero matches for $HOME or the username.
- 2026-11-01 step 8: real verification on clean commit 2d4ca5e, all on battery (86% → 80%). Benchmark imgsz 640: 46.19 / 45.86 / 43.99 → median 45.86; imgsz 480: 55.16 / 54.21 / 50.61 → median 54.21; compare --intended imgsz: +18.2%, ranges disjoint, Controlled; detect_track p50 19.48 → 16.23 ms, other stages ≤ 0.11 ms. First trial setup 3.45 s vs ~0.39 s later (one-time process startup). Overhead: pre-P2 code (6a419d7, temp worktree) 45.98 / 44.91 / 45.14 → median 45.14, overlapping the instrumented range: no measurable cost. Doctor all PASS; bundle of r_20261101T081305Z_d8239d verified (608 KiB, 0 home hits), a tampered copy refused (exit 1). Docs: README results + instrumentation cost, interview (Release A section, stale line refs fixed, test count), demo script (new UI flow, measurement tools), ARCHITECTURE layout.
