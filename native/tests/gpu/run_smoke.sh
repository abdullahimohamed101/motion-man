#!/usr/bin/env bash
# `motionman-native run` on the PC GPU (plan native-g2 step 5, N2-7). Uses the committed runspec that Python's own
# resolve_runspec built for the first 60 frames of the crosswalk clip. Checks:
#   1. a run succeeds: receipt success, 60 records and timing rows, contiguous record indices, a written video;
#   2. a second run into the same folder is refused and changes nothing (INV-5);
#   3. a runspec naming another engine is refused (exit 3) before any output is written;
#   4. SIGINT mid-run stops after a whole frame: exit 130, receipt interrupted, records == timing rows == processed.
# usage: run_smoke.sh <motionman-native> <runspec.json> <source.mp4> <profile.json> <model dir>
set -euo pipefail
bin=$1 runspec=$2 source=$3 profile=$4 model=$5
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fail() { echo "FAIL: $*"; exit 1; }
run() {  # run <staging dir>: stdout to <dir>.out, stderr to <dir>.err; returns the exit code (call as `run d || rc=$?`)
  local rc=0
  "$bin" run --runspec "$1/runspec.json" --run-id r_smoke --source "$source" --out "$1" --profile "$profile" \
    --model-dir "$model" >"$1.out" 2>"$1.err" || rc=$?
  return $rc
}
receipt() { python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(eval(sys.argv[2], {}, {"r": r}))' "$1/native_receipt.json" "$2"; }
patched() {  # patched <dest dir> <python expression on s (settings)>
  mkdir -p "$1"
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); s=d["settings"]; exec(sys.argv[2]); json.dump(d, open(sys.argv[3], "w"))' \
    "$runspec" "$2" "$1/runspec.json"
}

# 1. success
ok=$tmp/ok
mkdir "$ok" && cp "$runspec" "$ok/runspec.json"
run "$ok" || fail "run exited $? ($(tail -3 "$ok.err"))"
[ "$(receipt "$ok" 'r["status"]')" = success ] || fail "receipt status $(receipt "$ok" 'r["status"]')"
[ "$(receipt "$ok" 'r["frames"]["processed"], r["frames"]["records"], r["frames"]["timing_rows"]')" = "(60, 60, 60)" ] ||
  fail "counts $(receipt "$ok" 'r["frames"]')"
python3 - "$ok" "$runspec" <<'PY' || fail "outputs do not match the contract"
import json, sys
from pathlib import Path
out, spec = Path(sys.argv[1]), json.load(open(sys.argv[2]))
rows = [json.loads(line) for line in (out / "tracks.jsonl").read_text().splitlines()]
assert [r["source_frame_index"] for r in rows] == list(range(60)), "record indices are not 0..59"
assert all(r["schema_version"] == "1.0" and r["run_id"] == "r_smoke" for r in rows)
assert all(r["source_width"] == 1920 and r["source_height"] == 1080 for r in rows)
objs = [o for r in rows for o in r["objects"]]
ids = {o["track_id"] for o in objs if o["track_id"] is not None}
assert objs and ids, "no tracked objects on the crosswalk clip"
assert all(o["class_name"] in spec["class_names"].values() for o in objs)
t = [json.loads(line) for line in (out / "timings.jsonl").read_text().splitlines()]
assert [list(r) for r in t[:1]] == [["source_frame_index", "decode_ms", "preprocess_ms", "infer_ms", "track_ms",
                                     "render_ms", "write_ms"]]
r = json.load(open(out / "native_receipt.json"))
assert r["engine"]["engine_sha256"] == spec["settings"]["backend"]["engine_sha256"]
assert r["source_sha256"] == spec["settings"]["input"]["input_id"]
assert (out / "annotated.mp4").stat().st_size > 0
tot = r["timing"]["stage_total_ms"]
print(f"run: 60 frames, {len(objs)} objects, {len(ids)} track IDs (max {r['objects']['max_track_id']}), "
      f"{r['objects']['admitted_detections']} admitted, {r['objects']['dropped_no_extent']} dropped, codec "
      f"{r['video']['codec']}, wall {r['timing']['wall_s']} s incl. {r['timing']['model_load_s']} s load; "
      "stage ms/frame (observation) " + ", ".join(f"{k[:-3]} {v / 60:.2f}" for k, v in tot.items()))
PY
grep -q '"event": "finished", "status": "success"' "$ok.out" || fail "no success status line"

# 2. never overwrite
before=$(sha256sum "$ok/tracks.jsonl" "$ok/native_receipt.json")
if run "$ok"; then fail "a second run into a finished folder succeeded"; fi
[ "$(sha256sum "$ok/tracks.jsonl" "$ok/native_receipt.json")" = "$before" ] || fail "the second run changed outputs"
echo "rerun into the same folder: refused, outputs unchanged"

# 3. wrong engine
bad=$tmp/bad
patched "$bad" 's["backend"]["engine_sha256"] = "sha256:" + "0" * 64'
rc=0
run "$bad" || rc=$?
[ "$rc" -eq 3 ] || fail "wrong engine exited $rc, expected 3"
[ "$(receipt "$bad" 'r["status"]')" = refused ] || fail "wrong engine receipt $(receipt "$bad" 'r["status"]')"
[ ! -e "$bad/tracks.jsonl" ] && [ ! -e "$bad/annotated.mp4" ] || fail "a refused run wrote outputs"
echo "wrong engine: refused (exit 3), no outputs"

# 4. SIGINT
int=$tmp/int
patched "$int" 's["input"]["frame_range"] = [0, None]'
"$bin" run --runspec "$int/runspec.json" --run-id r_smoke --source "$source" --out "$int" --profile "$profile" \
  --model-dir "$model" >"$int.out" 2>"$int.err" &
pid=$!
for _ in $(seq 100); do grep -q '"event": "progress"' "$int.out" 2>/dev/null && break; sleep 0.2; done
kill -INT "$pid"
rc=0
wait "$pid" || rc=$?
[ "$rc" -eq 130 ] || fail "SIGINT exited $rc, expected 130 ($(tail -3 "$int.err"))"
[ "$(receipt "$int" 'r["status"]')" = interrupted ] || fail "SIGINT receipt $(receipt "$int" 'r["status"]')"
n=$(receipt "$int" 'r["frames"]["processed"]')
[ "$(receipt "$int" 'r["frames"]["records"], r["frames"]["timing_rows"]')" = "($n, $n)" ] || fail "partial counts differ"
[ "$(wc -l <"$int/tracks.jsonl")" -eq "$n" ] && [ "$(wc -l <"$int/timings.jsonl")" -eq "$n" ] ||
  fail "partial files do not hold $n whole lines"
[ "$n" -gt 0 ] && [ "$n" -lt 1356 ] || fail "interrupted after $n frames"
echo "SIGINT: exit 130, interrupted after $n whole frames"
echo "native_run_smoke passed"
