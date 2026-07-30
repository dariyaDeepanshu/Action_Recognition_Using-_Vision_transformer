# =============================================================================
# Localisation Metrics (Part C.2)
# =============================================================================
# This module provides functions to evaluate spatio‑temporal action localisation
# on the JHMDB dataset. The metrics include:
#   - Frame‑level Average Precision (AP) for a single action class,
#   - Frame‑level mean AP (mAP) over all classes,
#   - Spatio‑temporal IoU for action tubes,
#   - Video‑level AP for a single class,
#   - Video‑level mAP over all classes.
#
# All AP implementations use 11‑point interpolation (standard Pascal VOC style).
# All IoU thresholds are 0.5, as specified in the project requirements.
# =============================================================================

import numpy as np
from collections import defaultdict

# The IoU function is reused from the dataset‑agnostic utility module.
from models.detection.tublar_utils import iou


def compute_frame_ap(gt_boxes_list, pred_boxes_list, iou_thresh=0.5):
    """
    Compute 11-point interpolated Average Precision for a single action class
    at the frame level.

    Args:
        gt_boxes_list   : list of T lists, each containing ground‑truth boxes
                          (x1,y1,x2,y2) in pixel coordinates for that frame.
        pred_boxes_list : list of T lists, each containing prediction dictionaries
                          with keys 'box' (x1,y1,x2,y2) and 'score' (float).
        iou_thresh      : IoU threshold for a detection to be considered correct
                          (default 0.5).

    Returns:
        ap : float in [0,1], the 11-point interpolated AP.
    """
    # Collect all predictions across frames as (score, frame_idx, box)
    all_pred = []
    for t, preds in enumerate(pred_boxes_list):
        for p in preds:
            all_pred.append((p['score'], t, p['box']))
    # Sort by decreasing score
    all_pred.sort(key=lambda x: -x[0])

    tp = np.zeros(len(all_pred))   # true positives
    fp = np.zeros(len(all_pred))   # false positives
    gt_used = defaultdict(set)     # per‑frame, which GT boxes have been matched

    for i, (score, t, box) in enumerate(all_pred):
        gt = gt_boxes_list[t]
        if len(gt) == 0:
            fp[i] = 1
            continue

        # Find the GT box with highest IoU (and not yet used in this frame)
        max_iou, best_j = 0, -1
        for j, g in enumerate(gt):
            iou_val = iou(box, g)
            if iou_val > max_iou:
                max_iou = iou_val
                best_j = j

        if max_iou >= iou_thresh and best_j not in gt_used[t]:
            tp[i] = 1
            gt_used[t].add(best_j)
        else:
            fp[i] = 1

    total_gt = sum(len(x) for x in gt_boxes_list)
    if total_gt == 0:
        return 0.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / (total_gt + 1e-8)
    precision = tp_cum / (tp_cum + fp_cum + 1e-8)

    # 11‑point interpolation (Pascal VOC style)
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        if np.any(recall >= thr):
            ap += np.max(precision[recall >= thr])
    return ap / 11


def compute_frame_map(gt_per_class, pred_per_class, iou_thresh=0.5):
    """
    Compute frame-level mean Average Precision over all action classes.

    Args:
        gt_per_class   : dict {class_id: gt_boxes_list}
        pred_per_class : dict {class_id: pred_boxes_list}
        iou_thresh     : IoU threshold for correct detection.

    Returns:
        mAP : float, the mean AP across all classes.
    """
    aps = []
    for c in gt_per_class:
        gt_list = gt_per_class[c]
        pred_list = pred_per_class.get(c, [[] for _ in gt_list])
        ap = compute_frame_ap(gt_list, pred_list, iou_thresh)
        aps.append(ap)
    return np.mean(aps) if aps else 0.0


def spatiotemporal_iou(tube1, tube2):
    """
    Calculate the spatio-temporal IoU between two action tubes.

    Each tube is a dictionary with:
        'frames' : list of frame indices,
        'boxes'  : list of (x1, y1, x2, y2) boxes (same length as frames).
    The spatio-temporal IoU is defined as the mean 2D IoU over the frames
    that both tubes share.

    Args:
        tube1, tube2 : tube dictionaries.

    Returns:
        float in [0,1]; 0 if there are no overlapping frames.
    """
    frames1 = set(tube1['frames'])
    frames2 = set(tube2['frames'])
    common = frames1 & frames2
    if not common:
        return 0.0

    ious = []
    for f in common:
        idx1 = tube1['frames'].index(f)
        idx2 = tube2['frames'].index(f)
        ious.append(iou(tube1['boxes'][idx1], tube2['boxes'][idx2]))
    return np.mean(ious)


def compute_video_ap(gt_tubes, pred_tubes, st_iou_thresh=0.5):
    """
    Compute video-level Average Precision for a single action class.

    Predicted tubes are sorted by their mean confidence score. A predicted tube
    is a true positive if its spatio-temporal IoU with an unmatched ground-truth
    tube meets the threshold. 11-point interpolation is used.

    Args:
        gt_tubes        : list of ground-truth tube dicts (each with 'frames',
                          'boxes', and 'label').
        pred_tubes      : list of predicted tube dicts (each with 'frames',
                          'boxes', 'scores', and 'label').
        st_iou_thresh   : spatio-temporal IoU threshold (default 0.5).

    Returns:
        ap : float in [0,1], the 11-point interpolated AP.
    """
    if len(gt_tubes) == 0:
        return 0.0

    # Sort predicted tubes by descending mean confidence
    pred_tubes = sorted(
        pred_tubes,
        key=lambda t: sum(t['scores']) / max(len(t['scores']), 1),
        reverse=True
    )

    tp = np.zeros(len(pred_tubes))
    fp = np.zeros(len(pred_tubes))
    gt_matched = set()

    for i, pred_tube in enumerate(pred_tubes):
        best_iou, best_j = 0, -1
        for j, gt_tube in enumerate(gt_tubes):
            if j in gt_matched:
                continue
            iou_val = spatiotemporal_iou(pred_tube, gt_tube)
            if iou_val > best_iou:
                best_iou = iou_val
                best_j = j
        if best_iou >= st_iou_thresh and best_j >= 0:
            tp[i] = 1
            gt_matched.add(best_j)
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / len(gt_tubes)
    precision = tp_cum / (tp_cum + fp_cum + 1e-8)

    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        if np.any(recall >= thr):
            ap += np.max(precision[recall >= thr])
    return ap / 11


def compute_video_map(gt_tubes, pred_tubes, st_iou_thresh=0.5):
    """
    Compute video-level mean Average Precision over all action classes.

    Args:
        gt_tubes        : list of all ground-truth tube dicts (each has 'label' key).
        pred_tubes      : list of all predicted tube dicts (each has 'label' key).
        st_iou_thresh   : spatio-temporal IoU threshold.

    Returns:
        mAP : float, the mean AP across all classes.
    """
    from collections import defaultdict
    gt_by_label = defaultdict(list)
    pred_by_label = defaultdict(list)

    for t in gt_tubes:
        gt_by_label[t['label']].append(t)
    for t in pred_tubes:
        pred_by_label[t['label']].append(t)

    aps = []
    for label, gt_list in gt_by_label.items():
        pred_list = pred_by_label.get(label, [])
        ap = compute_video_ap(gt_list, pred_list, st_iou_thresh)
        aps.append(ap)
    return np.mean(aps) if aps else 0.0