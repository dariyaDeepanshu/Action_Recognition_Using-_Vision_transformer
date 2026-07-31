# =============================================================================
# EVALUATE DETR LOCALISATION MODEL ON JHMDB TEST SET
# =============================================================================
# This script evaluates a trained DETR‑style spatio‑temporal action detector
# on the JHMDB dataset. It computes:
#   - Frame‑level mAP@0.5 (mean Average Precision over individual frames)
#   - Video‑level mAP@0.5 (mean AP over action tubes)
#   - Per‑class frame‑AP (top‑5 and bottom‑5 classes)
#   - Saves all metrics to outputs/evaluate_localisation/<run>/localisation_metrics.json
#   - Generates qualitative temporal strips (first 3 test videos) with
#     ground‑truth (green) and predicted (red) bounding boxes.
#
# Usage:
#     python scripts/evaluate_localisation.py \
#         --checkpoint checkpoints/detr_best.pt \
#         --jhmdb_root JHMDB_simp
# =============================================================================

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
import json

# Add project root to Python path for local imports
sys.path.append('.')

from data.jhmdb_dataset import get_jhmdb_dataloaders
from models.detection.detr_head import DETRLocalisation
from evaluation.localisation_metrics import compute_frame_ap, compute_frame_map, compute_video_map
from training.trainer import set_seed
from models.detection.tublar_utils import expand_teff_to_frames, link_detections_to_tubes
from utils.run_utils import make_run_dir

# Spatial size to which frames are resized (must match training)
IMG_SIZE = 224


def detect_video(
    model: torch.nn.Module,
    frames: torch.Tensor,
    device: torch.device,
    conf_threshold: float = 0.5,
) -> list:
    """
    Run the DETR model on a single video and return detections per temporal slice.

    The model outputs (pred_logits, pred_boxes, _) where:
        - pred_logits: (1, Q, num_classes+1)
        - pred_boxes : (1, Q, T_eff, 4)   (cx, cy, w, h) normalised to [0,1]

    This function:
        - Keeps only queries with confidence > conf_threshold and label not
          background (label < model.num_classes).
        - Converts boxes from (cx,cy,w,h) to (x1,y1,x2,y2) pixel coordinates.
        - Returns a list of T_eff dictionaries, each containing:
            'boxes'  : list of bounding boxes in pixel coordinates,
            'scores' : list of confidence scores,
            'labels' : list of class indices.

    Args:
        model           : DETRLocalisation instance (in evaluation mode).
        frames          : Video clip tensor of shape (1, T, C, H, W).
        device          : torch.device ('cuda' or 'cpu').
        conf_threshold  : Minimum confidence for a detection to be kept.

    Returns:
        List of T_eff dicts (one per temporal slice).
    """
    model.eval()
    with torch.no_grad():
        pred_logits, pred_boxes, _ = model(frames.to(device))

    # pred_logits: (1, Q, C+1)
    # pred_boxes : (1, Q, T_eff, 4)
    T_eff = pred_boxes.shape[2]

    # Class probabilities for the first (only) video in the batch
    probs = torch.softmax(pred_logits[0], dim=-1)   # (Q, C+1)
    scores, labels = probs.max(dim=-1)              # (Q,), (Q,)

    # Keep foreground detections (label != num_classes) above threshold
    keep = (scores > conf_threshold) & (labels < model.num_classes)
    kept_boxes = pred_boxes[0, keep]               # (K, T_eff, 4)
    kept_scores = scores[keep].cpu().numpy()       # (K,)
    kept_labels = labels[keep].cpu().numpy()       # (K,)

    # Initialise per‑slice detection containers
    per_slice_dets = [{'boxes': [], 'scores': [], 'labels': []} for _ in range(T_eff)]

    # For each kept query, convert its tubelet boxes to pixel coordinates
    for q in range(len(kept_scores)):
        for t in range(T_eff):
            cx, cy, w, h = kept_boxes[q, t].cpu().numpy()
            x1 = (cx - w / 2) * IMG_SIZE
            y1 = (cy - h / 2) * IMG_SIZE
            x2 = (cx + w / 2) * IMG_SIZE
            y2 = (cy + h / 2) * IMG_SIZE
            per_slice_dets[t]['boxes'].append([x1, y1, x2, y2])
            per_slice_dets[t]['scores'].append(kept_scores[q])
            per_slice_dets[t]['labels'].append(kept_labels[q])

    return per_slice_dets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DETR localisation on JHMDB test set."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained DETR model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--jhmdb_root",
        type=str,
        default="JHMDB_simp",
        help="Root directory of the JHMDB dataset.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Number of frames sampled per video clip.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (should be 1 for evaluation).",
    )
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=0.5,
        help="Confidence threshold for keeping predictions.",
    )
    parser.add_argument(
        "--iou_link",
        type=float,
        default=0.4,
        help="IoU threshold for greedy tube linking.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run evaluation on ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default=None,
        help="Optional extra tag identifying this run, used in the output "
             "directory name.",
    )
    args = parser.parse_args()

    set_seed(42)                      # reproducible
    device = torch.device(args.device)
    run_dir = make_run_dir("evaluate_localisation", "detr", args.run_tag)

    # -------------------------------------------------------------------------
    # Load test data
    # -------------------------------------------------------------------------
    _, test_loader, class_names = get_jhmdb_dataloaders(
        args.jhmdb_root,
        num_frames=args.num_frames,
        batch_size=args.batch_size,
        num_workers=2,
    )
    num_classes = len(class_names)

    # -------------------------------------------------------------------------
    # Build model and load checkpoint
    # -------------------------------------------------------------------------
    model = DETRLocalisation(num_classes=num_classes)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    tube_t = model.tube_t   # temporal tube size (e.g., 2)

    # -------------------------------------------------------------------------
    # Containers for ground truth and predictions (frame‑level and tube‑level)
    # -------------------------------------------------------------------------
    gt_frames_by_class = {c: [] for c in range(num_classes)}
    pred_frames_by_class = {c: [] for c in range(num_classes)}
    all_tubes_gt = []
    all_tubes_pred = []

    # -------------------------------------------------------------------------
    # Iterate over test videos
    # -------------------------------------------------------------------------
    for batch in test_loader:
        frames = batch['frames']                # (1, T, C, H, W)
        bboxes_gt = batch['bboxes'][0].numpy()  # (T, 4) normalised xyxy
        label = batch['label'][0].item()
        T = frames.shape[1]                     # number of input frames

        # Record ground‑truth boxes per frame (scaled to pixel coordinates)
        for t in range(T):
            gt_box = bboxes_gt[t]
            if gt_box[2] > gt_box[0] and gt_box[3] > gt_box[1]:
                # Valid box (non‑zero area)
                gt_frames_by_class[label].append([gt_box * IMG_SIZE])
            else:
                # No box in this frame (padding)
                gt_frames_by_class[label].append([])

        # Obtain per‑slice detections and expand to per‑frame detections
        per_slice_dets = detect_video(model, frames, device, args.conf_thresh)
        per_frame_dets = expand_teff_to_frames(per_slice_dets, T, tube_t)

        # For frame‑level mAP, we only consider predictions with the correct label
        for frame_det in per_frame_dets:
            frame_preds = [
                {'box': box, 'score': score}
                for box, score, lbl in zip(frame_det['boxes'], frame_det['scores'], frame_det['labels'])
                if int(lbl) == label
            ]
            pred_frames_by_class[label].append(frame_preds)

        # Ground‑truth action tube (one tube per video, covering all frames)
        all_tubes_gt.append({
            'frames': list(range(T)),
            'boxes': (bboxes_gt * IMG_SIZE).tolist(),
            'scores': [1.0] * T,
            'label': label,
        })

        # Predicted tubes (linking detections across frames)
        pred_tubes = link_detections_to_tubes(per_frame_dets, iou_threshold=args.iou_link)
        all_tubes_pred.extend(pred_tubes)

    # -------------------------------------------------------------------------
    # Compute global metrics
    # -------------------------------------------------------------------------
    frame_map = compute_frame_map(gt_frames_by_class, pred_frames_by_class)
    video_map = compute_video_map(all_tubes_gt, all_tubes_pred)
    print(f"\n{'='*50}")
    print(f"Frame-level mAP@0.5 : {frame_map:.4f}")
    print(f"Video-level mAP@0.5 : {video_map:.4f}")
    print(f"{'='*50}\n")

    # -------------------------------------------------------------------------
    # Per‑class frame‑AP
    # -------------------------------------------------------------------------
    per_class_ap = {}
    for c in range(num_classes):
        ap = compute_frame_ap(gt_frames_by_class[c], pred_frames_by_class[c])
        per_class_ap[class_names[c]] = round(float(ap), 4)
    sorted_ap = sorted(per_class_ap.items(), key=lambda x: -x[1])

    # Save all metrics to JSON (including top‑5 / bottom‑5)
    loc_results = {
        "checkpoint": args.checkpoint,
        "frame_map_at_05": round(float(frame_map), 4),
        "video_map_at_05": round(float(video_map), 4),
        "per_class_frame_ap": per_class_ap,
        "top5_classes": [{"class": k, "ap": v} for k, v in sorted_ap[:5]],
        "bottom5_classes": [{"class": k, "ap": v} for k, v in sorted_ap[-5:]],
    }
    loc_save_path = os.path.join(run_dir, "localisation_metrics.json")
    with open(loc_save_path, "w") as f:
        json.dump(loc_results, f, indent=2)
    print(f"[INFO] Localisation metrics saved to {loc_save_path}")

    # Print summary to console
    print("Frame-mAP — Top 5 classes:")
    for name, ap in sorted_ap[:5]:
        print(f"  {name}: {ap:.4f}")
    print("Frame-mAP — Bottom 5 classes:")
    for name, ap in sorted_ap[-5:]:
        print(f"  {name}: {ap:.4f}")

    # -------------------------------------------------------------------------
    # Qualitative visualisations (first 3 test videos)
    # -------------------------------------------------------------------------
    vis_dataset = test_loader.dataset
    vis_dataset.items = vis_dataset.items[:3]
    vis_loader = torch.utils.data.DataLoader(vis_dataset, batch_size=1, shuffle=False)

    for vid_idx, batch in enumerate(vis_loader):
        frames = batch['frames']
        bboxes_gt = batch['bboxes'][0].cpu().numpy()
        label = batch['label'][0].item()
        T = frames.shape[1]

        per_slice_dets = detect_video(model, frames, device, args.conf_thresh)
        per_frame_dets = expand_teff_to_frames(per_slice_dets, T, tube_t)
        frames_np = frames[0].cpu().numpy()          # (T, C, H, W)

        # Sample 5 equally spaced frames for the temporal strip
        sample_indices = np.linspace(0, T - 1, 5, dtype=int)
        fig, axes = plt.subplots(1, 5, figsize=(15, 4))
        for i, t in enumerate(sample_indices):
            # Convert frame from (C, H, W) to (H, W, C) and clip to [0,1]
            frame = np.clip(frames_np[t].transpose(1, 2, 0), 0, 1)
            ax = axes[i]
            ax.imshow(frame)

            # Ground‑truth box (green)
            gt = bboxes_gt[t] * IMG_SIZE
            rect_gt = plt.Rectangle(
                (gt[0], gt[1]), gt[2] - gt[0], gt[3] - gt[1],
                linewidth=2, edgecolor='g', facecolor='none'
            )
            ax.add_patch(rect_gt)

            # Predicted boxes (red)
            for box in per_frame_dets[t]['boxes']:
                rect_pr = plt.Rectangle(
                    (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                    linewidth=2, edgecolor='r', facecolor='none'
                )
                ax.add_patch(rect_pr)

            ax.set_title(f"t={t}", fontsize=9)
            ax.axis('off')

        # Aggregate predicted labels across slices to determine the final prediction
        all_pred_labels = [int(lbl) for det in per_slice_dets for lbl in det['labels']]
        if all_pred_labels:
            pred_label = max(set(all_pred_labels), key=all_pred_labels.count)
            pred_name = class_names[pred_label]
        else:
            pred_name = "None"

        plt.suptitle(f"Video {vid_idx} | GT: {class_names[label]} | Pred: {pred_name}", fontsize=11)

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='g', linewidth=2, label='GT'),
            Line2D([0], [0], color='r', linewidth=2, label='Pred')
        ]
        axes[0].legend(handles=legend_elements, loc='upper left', fontsize=7)

        plt.tight_layout()
        out_path = os.path.join(run_dir, f"localisation_example_{vid_idx}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")

    print("Evaluation complete.")


if __name__ == "__main__":
    main()