# Five-minute architecture explanation
For the interview handoff (spec §10). Every number here comes from a run recorded in `docs/plans/done/`.
Describe MotionMan as a **Python object-tracking tool**, not a C++/TensorRT or silicon-validation platform.

## The walkthrough (about 5 minutes)

**0:00 – What it is (30 s).**
One command reads a recorded video, detects and tracks objects with a pretrained YOLO11n model plus ByteTrack,
and writes three artifacts: an annotated MP4, one JSONL record per frame, and a measured `report.json`.
It runs on CPU. The point is not the model; it's a small tool with reproducible inputs, structured results,
clear failures and honest measurement, which is what validation tooling needs.

**0:30 – The pipeline (60 s).** `pipeline.run(config)` is one synchronous loop
(`motionman/pipeline.py:88`): decode a frame → `tracker.update` → update trails → draw on a copy → write the
video frame and the JSONL line → record timings. The modules follow that loop:
- `video_io` validates the input (constant frame rate, even dimensions) and writes H.264.
- `tracking` is the only module that imports Ultralytics. It converts results into plain `Track` records and
  **never invents an ID**: if the tracker gives none, the ID stays null (`tracking.py:98`).
- `render` keeps at most 30 points per ID and forgets an ID after 60 unseen frames (`render.py:27`, `:41`).
- `metrics` measures, and `cli` maps failures to exit codes.

Dependencies point one way. Leaf modules never import each other, which is why the tests can swap in a fake
tracker.

**1:30 – Contracts and failure (60 s).**
- **Fail early, before writing anything.** Bad input, a missing device or bad weights fail before the output
  folder exists, with exit code 2 and one readable line. A non-empty output folder is refused, so results are
  never overwritten.
- **Cleanup survives every exit.** It lives in one `finally` (`pipeline.py:250`): the reader and writer are
  released, the JSONL is closed, and `report.json` is still written, marked `interrupted` or `error`.
- **Success is verified, not assumed.** A run only says `success` if the output video, reopened and decoded,
  has exactly the number of frames we processed (`pipeline.py:228`).
- **The tests run offline.** 355 of them use synthetic clips and a fake tracker. The suite blocks the network
  and fails if torch is ever imported, so it needs no GPU and no weights. CI runs the same gate.

**2:30 – Measurement (60 s).**
- **Wall time** runs from before the first decode to after the writer is released (`pipeline.py:167`–`:208`).
  End-to-end FPS is frames divided by that time, so it includes decode, inference, drawing and encoding.
- **Stage timings** split every frame into decode, detect+track, render and write, with setup and finalize
  around the loop and the remainder shown as unattributed. On the crosswalk clip detect+track is 89.5% of the
  wall time, render and write 3.7% each, decode 2.1%: the model is the bottleneck.
- **Latency:** detect+track p50/p95 time the whole `model.track` call and skip the first 10 frames as warmup
  (`metrics.py:19`, `:99`). I never call that number "inference-only".
- **A warmup fix the data forced.** `YOLO()` is lazy: the first `track` call took 2.5 s and landed in the wall
  time. One predict on a blank frame inside model load (`tracking.py:76`) dropped the first call to 70 ms with
  identical IDs, because predict mode never touches tracker state.
- **Results:** on an Apple M5 Pro CPU, three runs of a 1,356-frame 1080p clip measured a median of
  **44.29 fps on mains power** (p50 ≈ 20 ms) and **32.14 fps on battery** (p50 ≈ 27 ms) in the MVP. With the P2
  `benchmark` command, on battery at 86–80%: **45.86 fps** (43.99–46.19) at imgsz 640 and **54.21 fps**
  (50.61–55.16) at imgsz 480, a controlled +18.2% that `compare` confirmed had no other differences.
- **Honest variance.** An earlier battery set measured 25.95, which the warmup only partly explained. I
  reported it as unexplained until a later run on mains power showed power state was the missing variable.
  I then benchmarked both states properly, and P2 made power state a recorded, compared factor.

**3:30 – The UI (60 s).** A Streamlit page drives the same `pipeline.run`.
- **Why inference runs on a thread.** I tested Streamlit before designing the page. Any widget click stops the
  page's script thread with a `BaseException`, which would have killed a run mid-video. So `ui/runner.py`
  runs it on a worker thread (`runner.py:125`), one job per process (`runner.py:34`).
- **How the page stays current.** It polls read-only state snapshots: every 0.5 s during a run, every 2 s
  otherwise (`app.py:108`).
- **What I verified in a browser:**
  - metrics equal `report.json`;
  - clicks mid-run didn't stop it;
  - downloads were hash-identical to disk;
  - a second tab followed a run on its own.

**4:30 – Limits and next steps (30 s).**
- **Track IDs aren't counts.** The demo clip produced 701 distinct IDs with a median life of 11 frames.
- **There are no accuracy claims** without labelled data.
- **CUDA code exists but is unverified:** there's no NVIDIA GPU here.
- **The optional decoder thread wasn't built.** That's next, along with a GPU run that holds model, input
  size, precision and output constant.

## Release A additions (if asked, about 1 minute)
- **Contracts (P0).** Runs are staged, validated and atomically published with a manifest of content hashes;
  inputs are imported by hash; a resolved `runspec.json` records every setting, with a config hash.
- **Inspection (P1).** The workbench reopens any verified run without re-running the model: the exact source
  frame with its own records, per-object observed intervals, and an evidence PNG whose recipe is hashed.
- **Measurement (P2).** Stage timings in every run; `benchmark` (fresh trials, median and range, power state);
  `compare` (declared changes vs confounders, speed only); `doctor` (read-only checks with next steps and
  input-quality hints); `bundle` (allowlisted, redacted, hash-verified debug ZIP). The stage instrumentation
  costs nothing measurable: pre-P2 code ran at 45.14 fps median (44.91–45.98) under the same conditions.

## Release B, P3 (if asked, about 1 minute)
- **Events & zones.** Lines and rectangles in normalized coordinates per video; a pure rules engine over saved
  observations (3 px deadband, 100 ms / 2-observation confirmation, any missed frame resets a track, crossings need
  the motion to cross the finite line); results published atomically under an ID that hashes every input, so
  reruns are byte-identical and speed-independent. On the crosswalk clip: 22 crossings in under half a second, 4 of 6 sampled
  checked as genuine by eye, 2 at the frame edge; one false pair from a box changing shape, kept and explained
  rather than tuned away (a bigger deadband drops genuine crossings too).

## Release B, P4 (if asked, about 1 minute)
- **Review and evaluation.** A person reviews sampled frames into a draft (classes fully checked, accepted or added
  boxes, ignored areas, excluded frames), then commits a read-only, hashed revision. The evaluator is pure: fixed
  confidence and IoU thresholds, greedy highest-confidence matching with documented tie-breaks, only fully checked
  classes scored, ignored areas count neither way, null rates with a reason. Every score names revision, hash,
  thresholds and evaluator version.
- **Real result, stated carefully.** On 46 crosswalk frames: the 480 run scores precision 0.962, recall 0.785
  against reviewed boxes taken from the 640 run, so it is an agreement number, not true accuracy. Tracing examples
  showed 4 genuine small-pedestrian misses (frame 480), one merged box (frame 870), and one mistake in my own answer
  key (frame 60). A regression check (remove one prediction or one reviewed box) moves exactly one count.
- **Bug found on real data.** Ignored areas first only dropped predictions, so reviewed boxes inside them became
  false misses (149 of them here). Fixed, tested, and the evaluator version bumped so old scores can't mix in.

## The five questions the brief says to be ready for

**Why are detection and tracking separate?**
Detection answers "what is in this frame"; it is stateless, so each frame stands alone. Tracking answers
"which detection is the same object as last frame". It is stateful: ByteTrack matches new boxes to existing
tracks using motion prediction and IoU, keeps low-confidence boxes to rescue tracks through occlusion, and
assigns IDs. Keeping them separate means you can swap the detector without touching identity logic. It also
explains the failure modes: a missed detection is a detector problem, and an ID switch is an association problem.

**Why does a tracker need ordered frames and one owner?**
The tracker's state is a prediction of where each object will be in the next frame. Feed it frames out of
order, or from two videos, and those predictions are wrong. Tracks then die and IDs fragment or swap. So
there is one model instance per run (`pipeline.py:144`), frames go in strictly in order with `persist=True`
(`tracking.py:63`), and runs never share a tracker. In the UI that's enforced by a process-wide lock: a
second run is refused rather than run alongside the first.

**Why does a bounded queue need backpressure?** *(Designed, not built: spec §7.)*
A decoder thread would feed frames to the main loop through `Queue(maxsize=2)`. If decode is faster than
tracking, an unbounded queue grows until memory runs out, holding about 6 MB per 1080p frame. With a bound,
`put()` blocks, so the decoder waits for the consumer. That's backpressure: the slow stage sets the pace.
Dropping frames instead would corrupt the tracker's motion assumptions and break "one JSONL record per frame".
To stay safe, errors on the decoder thread must reach the main thread, and both sides need a stop event and
bounded waits so cancelling can't deadlock. The UI's job lock is the same idea at a coarser level: capacity
one, and the overflow is refused rather than queued.

**How does cleanup survive exceptions?**
Everything that must happen goes in one `finally` (`pipeline.py:250`): release the capture, release the writer
(which finalises the MP4), close the JSONL, stop the clock and write the report. The status depends on what
was caught: `KeyboardInterrupt` gives `interrupted`, anything else gives `error`, and a non-`Exception` exit
like Streamlit's stop gives `error` with its name (`pipeline.py:241`, `:247`). The report is written to a
temporary file and then renamed (`metrics.py:219`), so a crash never leaves half a report. Tested: Ctrl-C at
frame 126 left 126 processed frames = 126 JSONL lines = 126 decoded frames, and the partial video played.

**Why does GPU timing need synchronization?**
CUDA calls are asynchronous: they queue work and return before the GPU finishes. Without synchronising, a
timer around the call measures how long it took to enqueue the work, not to do it, and the real cost shows up
in some later call. So `Tracker.update` calls `torch.cuda.synchronize()` before returning (`tracking.py:84`),
which makes the caller's `perf_counter` measure finished work. Honest caveat: that path is UNVERIFIED here,
because this Mac has no NVIDIA GPU.

## Likely follow-ups, with real answers

- **"Why 701 IDs on a 57-second clip?"** It's a crowded scramble crosswalk at 1080p, downscaled to 640 for
  inference. Small pedestrians drop below confidence, and occlusion fragments their tracks. ByteTrack's
  counter also skips numbers, so labels reach the thousands. Static traffic lights keep one ID for the whole
  clip. A re-ID model or a larger input size would help, but both are out of scope.
- **"Why is the UI slower?"** The browser decodes video on the same CPU: UI runs measured 20–24 fps against
  31–32 on the CLI, all on battery. Benchmarks use the CLI.
- **"Is it deterministic?"** On CPU, yes. Two UI runs of the same clip produced byte-identical annotated
  videos.
- **"What would you do with a GPU?"** Hold the model, input size, FP32 and output settings constant, then
  compare CPU and CUDA with `benchmark` and `compare --intended device`. Stage timings say what comes next:
  detect+track is about 89% of the frame today; decode (2.1%) and write (3.7%) are small but measured as lower
  bounds, because the codec may work on its own threads, so they would be re-measured once inference is faster.
- **"What was hardest?"** Measuring honestly. The lazy-loading warmup, tracking down the run-to-run variance (power
  state), and making a
  partial run's report say exactly what each file holds after a Ctrl-C.
