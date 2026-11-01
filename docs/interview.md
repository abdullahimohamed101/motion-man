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
(`motionman/pipeline.py:86`): decode a frame → `tracker.update` → update trails → draw on a copy → write the
video frame and the JSONL line → record timings. The modules follow that loop:
- `video_io` validates the input (constant frame rate, even dimensions) and writes H.264.
- `tracking` is the only module that imports Ultralytics. It converts results into plain `Track` records and
  **never invents an ID**: if the tracker gives none, the ID stays null (`tracking.py:98`).
- `render` keeps at most 30 points per ID and forgets an ID after 60 unseen frames (`render.py:37`, `:40`).
- `metrics` measures, and `cli` maps failures to exit codes.

Dependencies point one way. Leaf modules never import each other, which is why the tests can swap in a fake
tracker.

**1:30 – Contracts and failure (60 s).**
- **Fail early, before writing anything.** Bad input, a missing device or bad weights fail before the output
  folder exists, with exit code 2 and one readable line. A non-empty output folder is refused, so results are
  never overwritten.
- **Cleanup survives every exit.** It lives in one `finally` (`pipeline.py:211`): the reader and writer are
  released, the JSONL is closed, and `report.json` is still written, marked `interrupted` or `error`.
- **Success is verified, not assumed.** A run only says `success` if the output video, reopened and decoded,
  has exactly the number of frames we processed (`pipeline.py:187`).
- **The tests run offline.** 200 of them use synthetic clips and a fake tracker. The suite blocks the network
  and fails if torch is ever imported, so it needs no GPU and no weights. CI runs the same gate.

**2:30 – Measurement (60 s).**
- **Wall time** runs from before the first decode to after the writer is released (`pipeline.py:160`–`:183`).
  End-to-end FPS is frames divided by that time, so it includes decode, inference, drawing and encoding.
- **Latency:** detect+track p50/p95 time the whole `model.track` call and skip the first 10 frames as warmup
  (`metrics.py:19`, `:51`). I never call that number "inference-only".
- **A warmup fix the data forced.** `YOLO()` is lazy: the first `track` call took 2.5 s and landed in the wall
  time. One predict on a blank frame inside model load (`tracking.py:76`) dropped the first call to 70 ms with
  identical IDs, because predict mode never touches tracker state.
- **Results:** on an Apple M5 Pro CPU, three runs of a 1,356-frame 1080p clip measured a median of
  **44.29 fps on mains power** (p50 ≈ 20 ms) and **32.14 fps on battery** (p50 ≈ 27 ms).
- **Honest variance.** An earlier battery set measured 25.95, which the warmup only partly explained. I
  reported it as unexplained until a later run on mains power showed power state was the missing variable.
  I then benchmarked both states properly.

**3:30 – The UI (60 s).** A Streamlit page drives the same `pipeline.run`.
- **Why inference runs on a thread.** I tested Streamlit before designing the page. Any widget click stops the
  page's script thread with a `BaseException`, which would have killed a run mid-video. So `ui/runner.py`
  runs it on a worker thread (`runner.py:120`), one job per process (`runner.py:34`).
- **How the page stays current.** It polls read-only state snapshots: every 0.5 s during a run, every 2 s
  otherwise (`app.py:90`).
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
there is one model instance per run (`pipeline.py:141`), frames go in strictly in order with `persist=True`
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
Everything that must happen goes in one `finally` (`pipeline.py:211`): release the capture, release the writer
(which finalises the MP4), close the JSONL, stop the clock and write the report. The status depends on what
was caught: `KeyboardInterrupt` gives `interrupted`, anything else gives `error`, and a non-`Exception` exit
like Streamlit's stop gives `error` with its name (`pipeline.py:202`, `:208`). The report is written to a
temporary file and then renamed (`metrics.py:138`), so a crash never leaves half a report. Tested: Ctrl-C at
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
  compare CPU and CUDA across three runs each. Then try the decoder thread, because at about 20 ms of
  detect+track out of about 22.6 ms per frame (on mains power), decode and encode are the next bottleneck once
  inference gets faster.
- **"What was hardest?"** Measuring honestly. The lazy-loading warmup, tracking down the run-to-run variance (power
  state), and making a
  partial run's report say exactly what each file holds after a Ctrl-C.
