# ADR-003: RT-DETR from the original implementation, frozen by manifest
Status: Proposed (confirmed or replaced at the end of G1)
Date: 2026-11-04

## Context
The native engine needs a detector that TensorRT can run without custom plugins and that the owner's direction
names: RT-DETR. Several packages expose a model called RT-DETR with different exporters, preprocessing and
output conventions. Mixing them silently changes results.

## Decision
- Use the original implementation (lyuwenyu/RT-DETR, `rtdetrv2_pytorch`) at one pinned commit, with one R18-class
  COCO checkpoint as the first candidate: RT-DETRv2-S (`rtdetrv2_r18vd_120e_coco`), profile
  `configs/native/rtdetr_r18_640.json`. Larger variants are a G3/G4 hardware decision.
- Amended 2026-11-07 (G1 step 6): v2, not `rtdetr_pytorch` v1, because v1 pins torch 2.0.1, which has no wheels for
  the PC's Python 3.12; v2 is the same author's PyTorch-native successor with the same deploy exporter contract.
- Export with the upstream ONNX exporter and its deploy postprocessor: inputs `images`, `orig_target_sizes`
  (width, height); outputs `labels`, `boxes` (xyxy in source pixels), `scores`. The real exported names, dtypes
  and shapes are checked and recorded; nothing is assumed from YOLO.
- Preprocessing: fixed 640×640, batch 1, direct resize (no letterbox), RGB, NCHW, frozen in a machine-readable
  contract with golden tensors.
- Precision: FP32 engine first (TF32 policy recorded), mixed precision only after G1 parity, following the pinned
  TensorRT version's own precision workflow.
- `configs/native/` holds the intended profile; a generated `manifest.json` holds the actual hashes and revisions.
  Weights, ONNX and engines are never committed.

## Alternatives
- Ultralytics `RTDETR`: convenient, but a different exporter and preprocessing from the original. Rejected for the
  native contract; usable only as a cross-check.
- Keep YOLO11n in TensorRT: does not meet the direction. Rejected.

## Consequences
- The checkpoint and the RT-DETR repository are downloaded on the PC at G1, with the owner's go-ahead.
- Detection numbers from YOLO11n runs and RT-DETR runs are different models; comparisons say so.

## Revisit when
The pinned export needs a TensorRT plugin, fails parity (spec C§6), or does not fit the PC's GPU.
