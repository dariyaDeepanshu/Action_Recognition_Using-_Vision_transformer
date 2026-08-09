# =============================================================================
# TUBULAR UTILITIES FOR ACTION TUBES
# =============================================================================
# This module provides helper functions for converting DETR's per‑temporal‑slice
# detections into per‑frame detections and for linking detections across frames
# into action tubes. These utilities are used in the localisation evaluation
# pipeline (frame‑mAP, video‑mAP) and in qualitative visualisations.
# =============================================================================

from typing import List, Dict, Any
import numpy as np


def iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute the Intersection over Union (IoU) of two axis-aligned bounding boxes.

    Args:
        box1, box2 : list of four floats [x1, y1, x2, y2] where (x1,y1) is the
                     top-left corner and (x2,y2) is the bottom-right corner.

    Returns:
        IoU value in [0, 1]. Returns 0 if there is no overlap.
    """
    # Intersection rectangle coordinates
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])

    # Intersection area (zero if boxes do not overlap)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Individual box areas
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Union area
    union = area1 + area2 - inter

    # IoU (add epsilon to avoid division by zero)
    return inter / (union + 1e-8)


def expand_teff_to_frames(
    per_slice_dets: List[Dict[str, Any]],
    T: int,
    tube_t: int,
) -> List[Dict[str, Any]]:
    """
    Expand per-temporal-slice detections to per-frame detections.

    VideoMAE processes clips by grouping consecutive frames into temporal tubes
    of size `tube_t`. Each such tube produces one set of detections for a
    temporal slice. This function maps those slice-wise detections back to
    individual frames by repeating each slice's detections for all frames
    that belong to that slice.

    Example (tube_t = 2, T = 8 frames):
        The model produces T_eff = T // tube_t = 4 slices.
        Slice 0 corresponds to frames 0 and 1,
        Slice 1 to frames 2 and 3,
        Slice 2 to frames 4 and 5,
        Slice 3 to frames 6 and 7.
        This function repeats the detections of slice 0 for frame 0 and frame 1,
        slice 1 for frames 2 and 3, etc.

    Args:
        per_slice_dets: List of T_eff dictionaries, each with keys:
            'boxes'  : list of [x1, y1, x2, y2] boxes (pixel coordinates).
            'scores' : list of confidence scores (float).
            'labels' : list of class indices (int).
        T             : Total number of frames in the original video clip.
        tube_t        : Temporal tube size (2 for VideoMAE-base).

    Returns:
        per_frame_dets: List of T dictionaries, each with the same structure
                        as the input per-slice dicts. Frame `t` receives the
                        detections of slice `t // tube_t` (clamped).
    """
    T_eff = len(per_slice_dets)
    per_frame_dets = []
    for t in range(T):
        # Determine which temporal slice this frame belongs to
        slice_idx = min(t // tube_t, T_eff - 1)
        per_frame_dets.append(per_slice_dets[slice_idx])
    return per_frame_dets


def link_detections_to_tubes(
    detections: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Link per-frame detections into action tubes using a greedy forward pass.

    The algorithm processes frames in temporal order (t = 0 … T-1). For each
    detection in the current frame, it tries to extend an existing tube if:
        - The tube's last frame is exactly the previous frame (no temporal gap).
        - The detection's label matches the tube's label.
        - The IoU between the detection and the tube's last box exceeds the
          threshold.
    If multiple detections could extend the same tube, the one with the highest
    IoU is chosen (greedy). Unmatched detections start new tubes.

    This method does **not** use the Hungarian algorithm; it is a fast greedy
    heuristic that works well when the number of actors per frame is low
    (typical for JHMDB). The resulting tubes are used to compute video-level
    mAP via spatio-temporal IoU.

    Args:
        detections   : List of T dictionaries, each with keys:
                           'boxes'  : list of [x1, y1, x2, y2] boxes (pixels).
                           'scores' : list of confidence scores (float).
                           'labels' : list of class indices (int).
        iou_threshold: Minimum IoU between a detection and the last box of a
                       tube to allow extension.

    Returns:
        List of tube dictionaries, each with:
            'frames' : list of frame indices (int) where the tube is active.
            'boxes'  : list of boxes (same order as frames).
            'scores' : list of confidence scores (same order).
            'label'  : integer class label (the label of the first detection,
                       assumed constant for the whole tube).
    """
    tubes: List[Dict[str, Any]] = []

    for t, frame_dets in enumerate(detections):
        if len(frame_dets['boxes']) == 0:
            continue

        # Keep track of which detections in this frame are already matched
        matched = [False] * len(frame_dets['boxes'])

        # Try to match each detection to an existing tube
        for tube in tubes:
            # Only consider tubes that ended exactly on the previous frame
            if tube['frames'][-1] != t - 1:
                continue

            best_iou, best_idx = 0.0, -1
            for i, box in enumerate(frame_dets['boxes']):
                if matched[i]:
                    continue
                if int(frame_dets['labels'][i]) != tube['label']:
                    continue   # class mismatch
                iou_val = iou(box, tube['boxes'][-1])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = i

            if best_iou >= iou_threshold and best_idx != -1:
                # Extend the tube
                tube['frames'].append(t)
                tube['boxes'].append(frame_dets['boxes'][best_idx])
                tube['scores'].append(frame_dets['scores'][best_idx])
                matched[best_idx] = True

        # All detections that were not matched become new tubes
        for i, already_matched in enumerate(matched):
            if not already_matched:
                tubes.append({
                    'frames': [t],
                    'boxes':  [frame_dets['boxes'][i]],
                    'scores': [frame_dets['scores'][i]],
                    'label':  int(frame_dets['labels'][i]),
                })

    return tubes