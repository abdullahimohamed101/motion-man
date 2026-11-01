"""The run shelf (spec A§8, plan P1 D4/D5): discover published runs and open them only after verification.

Listing is cheap (manifests only). Opening a canonical run re-hashes every artifact against its manifest and
the managed source against its input_id, and refuses the run on any mismatch. Legacy MVP outputs are listed
read-only and labelled unverified: they carry no hashes, so nothing is claimed about them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from motionman import pipeline, store
from motionman.types import MotionManError

Kind = Literal["canonical", "legacy"]
KNOWN_MANIFEST_SCHEMAS = {"1.0"}


class RunError(MotionManError):
    """A run cannot be opened as trustworthy evidence; the message says why."""


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    kind: Kind
    path: Path
    display_name: str | None  # the source's original filename, when known
    started_at: str | None
    frames: int | None

    @property
    def label(self) -> str:
        name = self.display_name or "unknown source"
        when = (self.started_at or "")[:16].replace("T", " ")
        # Selectboxes match choices by label, so two runs of one clip in the same minute need a distinct tail.
        tag = self.run_id.removeprefix("legacy:") if self.kind == "legacy" else self.run_id.rsplit("_", 1)[-1]
        suffix = " · Legacy (unverified)" if self.kind == "legacy" else ""
        return " · ".join(x for x in (name, when, tag) if x) + suffix


@dataclass(frozen=True)
class OpenRun:
    summary: RunSummary
    report: dict
    manifest: dict | None  # None for legacy runs
    runspec: dict | None
    records_path: Path
    video_path: Path
    source_path: Path | None  # None when the source can't be found (legacy runs only)
    source_verified: bool  # True only when the source bytes match the run's input_id
    fps: float
    width: int
    height: int
    records_sha256: str | None

    @property
    def run_id(self) -> str:
        return self.summary.run_id


def list_runs(root: store.DataRoot, legacy_dir: Path | None = None) -> list[RunSummary]:
    """Published canonical runs (newest first), then legacy MVP outputs. Staging is never listed."""
    canonical = []
    if root.runs.is_dir():
        for folder in root.runs.iterdir():
            manifest = _read_json(folder / store.MANIFEST)
            if not manifest or manifest.get("status") != "SUCCEEDED" or manifest.get("run_id") != folder.name:
                continue
            canonical.append(
                RunSummary(
                    run_id=folder.name,
                    kind="canonical",
                    path=folder,
                    display_name=_display_name(root, manifest.get("input_id")),
                    started_at=manifest.get("started_at_utc"),
                    frames=(manifest.get("counts") or {}).get("decoded_frames"),
                )
            )
    canonical.sort(key=lambda r: (r.started_at or "", r.run_id), reverse=True)  # ms timestamps; ID breaks exact ties
    legacy = []
    if legacy_dir is not None and legacy_dir.is_dir():
        for folder in sorted(legacy_dir.iterdir()):
            report = _read_json(folder / pipeline.REPORT)
            if not report or report.get("status") != "success" or not (folder / pipeline.TRACKS).is_file():
                continue
            legacy.append(
                RunSummary(
                    run_id=f"legacy:{folder.name}",
                    kind="legacy",
                    path=folder,
                    display_name=Path((report.get("source") or {}).get("path") or folder.name).name,
                    started_at=report.get("created_at"),
                    frames=report.get("processed_frames"),
                )
            )
    return canonical + legacy


def open_run(summary: RunSummary, root: store.DataRoot) -> OpenRun:
    return _open_canonical(summary, root) if summary.kind == "canonical" else _open_legacy(summary)


def _open_canonical(summary: RunSummary, root: store.DataRoot) -> OpenRun:
    folder = summary.path
    manifest = _read_json(folder / store.MANIFEST)
    if not manifest:
        raise RunError(f"{summary.run_id}: manifest missing or unreadable")
    if manifest.get("schema_version") not in KNOWN_MANIFEST_SCHEMAS:
        raise RunError(f"{summary.run_id}: unsupported manifest schema {manifest.get('schema_version')!r}")
    if manifest.get("status") != "SUCCEEDED":
        raise RunError(f"{summary.run_id}: run status is {manifest.get('status')}, not SUCCEEDED")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not all(isinstance(a, dict) for a in artifacts):
        raise RunError(f"{summary.run_id}: manifest artifacts are malformed")
    listed = {a.get("path") for a in artifacts}
    missing = {pipeline.TRACKS, pipeline.ANNOTATED, pipeline.REPORT} - listed
    if missing or not manifest.get("records_sha256"):
        raise RunError(f"{summary.run_id}: manifest doesn't cover {', '.join(sorted(missing)) or 'the records hash'}")
    for artifact in artifacts:
        name, digest = artifact.get("path"), artifact.get("sha256")
        path = folder / str(name)
        if not isinstance(name, str) or Path(name).is_absolute() or folder.resolve() not in path.resolve().parents:
            raise RunError(f"{summary.run_id}: manifest names a file outside the run folder")
        if not path.is_file() or store.sha256_file(path) != digest:
            raise RunError(f"{summary.run_id}: {name} does not match its manifest; the run may be damaged")
    runspec = _read_json(folder / store.RUNSPEC)
    if not runspec or runspec.get("config_sha256") != manifest.get("config_sha256"):
        raise RunError(f"{summary.run_id}: runspec.json does not match the manifest")
    if store.sha256_json(runspec["settings"]) != runspec["config_sha256"]:
        raise RunError(f"{summary.run_id}: runspec.json settings do not match their own hash")
    input_id = manifest.get("input_id")
    source = root.inputs / str(input_id).removeprefix("sha256:") / store.MANAGED_SOURCE
    if not source.is_file() or store.sha256_file(source) != input_id:
        raise RunError(f"{summary.run_id}: its source video is missing or changed in {source.parent}")
    try:
        inp = runspec["settings"]["input"]
        fps, width, height = float(inp["fps"]), int(inp["width"]), int(inp["height"])
    except (KeyError, TypeError, ValueError):
        raise RunError(f"{summary.run_id}: runspec.json has no usable input timing") from None
    return OpenRun(
        summary=summary,
        report=_read_json(folder / pipeline.REPORT) or {},
        manifest=manifest,
        runspec=runspec,
        records_path=folder / pipeline.TRACKS,
        video_path=folder / pipeline.ANNOTATED,
        source_path=source,
        source_verified=True,
        fps=fps,
        width=width,
        height=height,
        records_sha256=manifest["records_sha256"],
    )


def _open_legacy(summary: RunSummary) -> OpenRun:
    folder = summary.path
    report = _read_json(folder / pipeline.REPORT)
    if not report:
        raise RunError(f"{summary.run_id}: report.json missing or unreadable")
    src = report.get("source") or {}
    try:
        fps, width, height = float(src["fps"]), int(src["width"]), int(src["height"])
    except (KeyError, TypeError, ValueError):
        raise RunError(f"{summary.run_id}: report.json has no usable source timing") from None
    recorded = Path(src["path"]) if src.get("path") else None
    return OpenRun(
        summary=summary,
        report=report,
        manifest=None,
        runspec=None,
        records_path=folder / pipeline.TRACKS,
        video_path=folder / pipeline.ANNOTATED,
        source_path=recorded if recorded is not None and recorded.is_file() else None,
        source_verified=False,  # MVP runs recorded no hashes; the file at that path may not be the one used
        fps=fps,
        width=width,
        height=height,
        records_sha256=None,
    )


def _display_name(root: store.DataRoot, input_id: str | None) -> str | None:
    if not input_id:
        return None
    meta = _read_json(root.inputs / input_id.removeprefix("sha256:") / "meta.json")
    return meta.get("display_name") if meta else None


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None
