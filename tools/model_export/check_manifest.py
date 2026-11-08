"""Check a generated manifest against its profile and the files on disk (NG-6). Exit 1 on any mismatch.

python tools/model_export/check_manifest.py rtdetr_r18_640
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cache_dir, compare_manifest, load_profile, sha256_file  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    args = ap.parse_args()
    profile = load_profile(args.profile)
    d = cache_dir(profile)
    manifest = json.loads((d / "manifest.json").read_text())
    problems = compare_manifest(profile, manifest)
    for what, path, want in [
        ("checkpoint", d / "checkpoint.pth", manifest["checkpoint"]["sha256"]),
        ("onnx", d / manifest["onnx"]["file"], manifest["onnx"]["sha256"]),
    ]:
        got = sha256_file(path)
        if got != want:
            problems.append(f"{what} file {path} is {got}, manifest says {want}")
    for p in problems:
        print("MISMATCH", p)
    print("manifest ok" if not problems else f"{len(problems)} mismatches")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
