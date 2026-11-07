"""Shared pieces of the G1 model tools (spec C§5, ADR-003). Pure helpers here need no torch, so pytest can import
them (INV-8); `build_model` imports torch and the pinned upstream source only when called, on the PC."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROFILES = ROOT / "configs" / "native"
GOLDEN = ROOT / "configs" / "reference" / "golden_images.json"
UPSTREAM_SRC = Path("~/.cache/motionman/src/RT-DETR").expanduser()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def load_profile(name: str) -> dict:
    return json.loads((PROFILES / f"{name}.json").read_text())


def cache_dir(profile: dict) -> Path:
    return Path(profile["cache_dir"]).expanduser()


def repo_commit() -> dict:
    """This repository's commit and whether the tree is dirty, for the exporter revision."""

    def git(*a):
        return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, check=True).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def check_upstream(profile: dict) -> Path:
    """The pinned upstream checkout, refusing any other commit or local edits."""
    up = profile["upstream"]
    head = subprocess.run(
        ["git", "-C", str(UPSTREAM_SRC), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(UPSTREAM_SRC), "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if head != up["commit"] or dirty:
        raise SystemExit(f"upstream checkout is {head} (dirty={bool(dirty)}), profile pins {up['commit']}")
    return UPSTREAM_SRC / up["subdir"]


def check_checkpoint(profile: dict) -> Path:
    path = cache_dir(profile) / "checkpoint.pth"
    got = sha256_file(path)
    if got != profile["checkpoint"]["sha256"]:
        raise SystemExit(f"checkpoint {path} is {got}, profile pins {profile['checkpoint']['sha256']}")
    return path


def build_model(profile: dict):
    """Upstream deploy model + deploy postprocessor, exactly as tools/export_onnx.py wraps them. Returns
    (module, class names in label order, cfg)."""
    import torch
    import torch.nn as nn

    src = check_upstream(profile)
    ckpt = check_checkpoint(profile)
    sys.path.insert(0, str(src))
    from src.core import YAMLConfig, yaml_utils  # noqa: E402  (pinned upstream source)
    from src.data.dataset.coco_dataset import mscoco_category2name, mscoco_label2category  # noqa: E402

    up = profile["upstream"]
    cfg = YAMLConfig(str(src / up["config"]), **yaml_utils.parse_cli(up["config_overrides"]))
    state = torch.load(ckpt, map_location="cpu")
    cfg.model.load_state_dict(state["ema"]["module"] if "ema" in state else state["model"])

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            return self.postprocessor(self.model(images), orig_target_sizes)

    names = [mscoco_category2name[mscoco_label2category[i]] for i in range(len(mscoco_label2category))]
    return Model().eval(), names, cfg


def compare_manifest(profile: dict, manifest: dict) -> list[str]:
    """Every way the generated manifest disagrees with the intended profile (empty means consistent)."""
    problems = []
    exp = profile["export"]
    pairs = [
        ("profile", profile["profile"], manifest.get("profile")),
        ("upstream commit", profile["upstream"]["commit"], manifest.get("upstream", {}).get("commit")),
        ("upstream config", profile["upstream"]["config"], manifest.get("upstream", {}).get("config")),
        ("checkpoint sha256", profile["checkpoint"]["sha256"], manifest.get("checkpoint", {}).get("sha256")),
        ("opset", exp["opset"], manifest.get("onnx", {}).get("opset")),
    ]
    problems += [f"{what}: profile {want!r}, manifest {got!r}" for what, want, got in pairs if want != got]
    tensors = manifest.get("onnx", {}).get("tensors", {})
    for kind in ("inputs", "outputs"):
        want = exp[kind]
        got = tensors.get(kind, {})
        if list(want) != list(got):
            problems.append(f"{kind} names: profile {list(want)}, manifest {list(got)}")
            continue
        for name, spec in want.items():
            for key in ("dtype", "shape"):
                if spec[key] != got[name].get(key):
                    problems.append(f"{name} {key}: profile {spec[key]}, manifest {got[name].get(key)}")
    if len(manifest.get("classes", {}).get("names", [])) != 80:
        problems.append("class vocabulary is not the 80 COCO names")
    return problems
