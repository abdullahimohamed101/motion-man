#!/usr/bin/env bash
# The C++ runner must refuse an engine whose manifest disagrees with this host or binary (NG-9, spec C§5).
# Builds a scratch model dir that links the real files but claims another TensorRT version, then expects exit 3.
# usage: refuse_tampered.sh <motionman-native> <python> <profile.json> <model dir> <images dir>
set -euo pipefail
bin=$1 py=$2 profile=$3 model=$4 images=$5
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
ln -s "$model/manifest.json" "$tmp/manifest.json"
ln -s "$model/model.fp32.engine" "$tmp/model.fp32.engine"
"$py" -c 'import json,sys; m=json.load(open(sys.argv[1])); m["identity"]["tensorrt"]="11.4.0.106"; json.dump(m, open(sys.argv[2], "w"))' \
  "$model/engine_manifest.fp32.json" "$tmp/engine_manifest.fp32.json"
set +e
out=$("$bin" infer --profile "$profile" --model-dir "$tmp" --images "$images" --out "$tmp/out.json" 2>&1)
rc=$?
set -e
echo "$out"
[ "$rc" -eq 3 ] || { echo "expected exit 3 (refused), got $rc"; exit 1; }
grep -q "tensorrt: engine built with '11.4.0.106'" <<<"$out" || { echo "refusal did not name the TensorRT mismatch"; exit 1; }
[ ! -e "$tmp/out.json" ] || { echo "a refused engine still wrote detections"; exit 1; }
echo "refused as expected"
