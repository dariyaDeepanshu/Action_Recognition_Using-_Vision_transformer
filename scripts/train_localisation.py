# =============================================================================
# TRAIN DETR SPATIO‑TEMPORAL ACTION LOCALISATION MODEL ON JHMDB
# =============================================================================
# This script trains the DETRLocalisation model (defined in detr_head.py) on
# the JHMDB dataset. It uses Hungarian matching + set prediction loss (losses.py)
# and saves the best checkpoint (lowest validation loss) to checkpoints/detr_best.pt.
#
# Key features:
#   - Supports freezing the VideoMAE backbone (default: frozen).
#   - Derives temporal labels from bounding‑box validity (real annotations vs.
#     fallback full‑frame boxes).
#   - Aligns ground‑truth boxes to the temporal slices used by the model.
#   - Logs training and validation loss to TensorBoard (logs/detr).
#   - Saves only the model with the lowest validation loss.
#
# Usage:
#     python scripts/train_localisation.py --jhmdb_root JHMDB_simp
#         --batch_size 4 --epochs 30 --lr 1e-4 --num_queries 10
# =============================================================================

import argparse
import os
import sys
import torch
from torch.utils.tensorboard import SummaryWriter

# Add project root to Python path for local imports
sys.path.append('.')

from data.jhmdb_dataset import get_jhmdb_dataloaders
from models.detection.detr_head import DETRLocalisation
from training.losses import HungarianMatcher, SetCriterion
from training.trainer import set_seed


# -----------------------------------------------------------------------------
# Helper: convert bounding boxes from xyxy to cxcywh (normalised)
# -----------------------------------------------------------------------------
def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-aligned boxes from (x1, y1, x2, y2) to (cx, cy, w, h).

    All coordinates are assumed to be normalised to [0, 1].

    Args:
        boxes: Tensor of shape (..., 4).

    Returns:
        Tensor of same shape with (cx, cy, w, h).
    """
    cx = (boxes[..., 0] + boxes[..., 2]) / 2
    cy = (boxes[..., 1] + boxes[..., 3]) / 2
    w = boxes[..., 2] - boxes[..., 0]
    h = boxes[..., 3] - boxes[..., 1]
    return torch.stack([cx, cy, w, h], dim=-1)


# -----------------------------------------------------------------------------
# Target builder for the set criterion
# -----------------------------------------------------------------------------
def build_targets(
    bboxes_teff: torch.Tensor,
    temporal_labels: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> list:
    """
    Convert per-batch and per-temporal-slice ground-truth boxes and temporal
    extents into the list-of-dicts format expected by SetCriterion.

    Each video in the batch produces exactly one ground-truth tube (since JHMDB
    clips contain a single actor). The function expands the batch into a list
    of B dictionaries, each with:
        'labels'   : (1,) tensor - class index.
        'boxes'    : (1, T_eff, 4) tensor - boxes for each temporal slice.
        'temporal' : (1, 2) tensor - (t_start, t_end) normalised.

    Args:
        bboxes_teff     : (B, T_eff, 4) normalised (cx, cy, w, h) boxes.
        temporal_labels : (B, 2) normalised (t_start, t_end) per video.
        labels          : (B,) action class indices.
        device          : Torch device for the tensors.

    Returns:
        List of B dictionaries, each containing the keys 'labels', 'boxes', 'temporal'.
    """
    B, T_eff, _ = bboxes_teff.shape
    targets = []
    for b in range(B):
        targets.append({
            'labels': labels[b:b+1].to(device),                # (1,)
            'boxes': bboxes_teff[b:b+1, :, :].to(device),      # (1, T_eff, 4)
            'temporal': temporal_labels[b:b+1, :].to(device),  # (1, 2)
        })
    return targets


# -----------------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train DETR localisation model on JHMDB."
    )
    parser.add_argument(
        "--jhmdb_root",
        type=str,
        default="JHMDB_simp",
        help="Root directory of the JHMDB dataset.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size (videos per batch).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for the DETR heads (backbone is frozen by default).",
    )
    parser.add_argument(
        "--num_queries",
        type=int,
        default=10,
        help="Number of learnable object queries.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Number of frames sampled per video clip.",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_false",
        default=True,
        help="Freeze the VideoMAE backbone (default: frozen). "
             "Pass this flag to unfreeze the backbone.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to train on ('cuda' or 'cpu').",
    )
    args = parser.parse_args()

    set_seed(42)                      # reproducibility
    device = torch.device(args.device)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs/detr", exist_ok=True)   # TensorBoard log directory

    # -------------------------------------------------------------------------
    # DataLoaders
    # -------------------------------------------------------------------------
    train_loader, test_loader, class_names = get_jhmdb_dataloaders(
        args.jhmdb_root,
        num_frames=args.num_frames,
        batch_size=args.batch_size,
        num_workers=2,
    )
    num_classes = len(class_names)

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    model = DETRLocalisation(
        num_classes=num_classes,
        num_queries=args.num_queries,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    # -------------------------------------------------------------------------
    # Loss and optimiser
    # -------------------------------------------------------------------------
    matcher = HungarianMatcher(
        cost_class=1.0, cost_bbox=5.0, cost_giou=2.0, cost_temporal=1.0
    )
    weight_dict = {
        'loss_ce': 1.0,
        'loss_bbox': 5.0,
        'loss_giou': 2.0,
        'loss_temporal': 1.0,
    }
    criterion = SetCriterion(num_classes, matcher, weight_dict)
    criterion.to(device)

    # Only trainable parameters are the DETR heads and positional embeddings
    # (the backbone is frozen by default)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    writer = SummaryWriter(log_dir="logs/detr")

    # -------------------------------------------------------------------------
    # Training loop with best‑model saving (by validation loss)
    # -------------------------------------------------------------------------
    best_val_loss = float('inf')
    checkpoint_path = "checkpoints/detr_best.pt"

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            frames = batch['frames'].to(device)         # (B, T, C, H, W)
            bboxes_xyxy = batch['bboxes'].to(device)    # (B, T, 4) normalised xyxy
            labels = batch['label'].to(device)          # (B,)

            B, T = frames.shape[:2]
            T_eff = T // model.tube_t   # number of temporal slices (e.g., 16 // 2 = 8)

            # ---- Align ground‑truth boxes to temporal slices ----
            # Pick the centre frame of each temporal tube
            t_indices = [
                min(t * model.tube_t + model.tube_t // 2, T - 1)
                for t in range(T_eff)
            ]
            bboxes_teff_xyxy = bboxes_xyxy[:, t_indices, :]   # (B, T_eff, 4)
            bboxes_teff = xyxy_to_cxcywh(bboxes_teff_xyxy)     # (B, T_eff, 4)

            # ---- Temporal labels derived from bounding‑box validity ----
            # In JHMDB, ground‑truth bounding boxes are only provided for frames
            # where the actor is present. The dataset fills missing frames with
            # a full‑frame fallback box [0,0,1,1] (area ≈ 1.0). We treat frames
            # with area < 0.98 as “annotated” and use the first and last such
            # frames to define the action temporal extent.
            with torch.no_grad():
                # Compute area of each box (normalised)
                areas = ((bboxes_xyxy[..., 2] - bboxes_xyxy[..., 0]) *
                         (bboxes_xyxy[..., 3] - bboxes_xyxy[..., 1]))   # (B, T)
                has_annotation = areas < 0.98                           # (B, T) bool

                temporal_labels = torch.zeros(B, 2, device=device)
                for b in range(B):
                    valid_t = has_annotation[b].nonzero(as_tuple=True)[0]
                    if len(valid_t) > 0:
                        temporal_labels[b, 0] = valid_t[0].float() / max(T - 1, 1)
                        temporal_labels[b, 1] = valid_t[-1].float() / max(T - 1, 1)
                    else:
                        # Fallback: treat whole clip as action interval
                        temporal_labels[b, 0] = 0.0
                        temporal_labels[b, 1] = 1.0

            # ---- Forward pass ----
            pred_logits, pred_boxes, pred_temporal = model(frames)
            # pred_boxes   : (B, Q, T_eff, 4)
            # pred_temporal: (B, Q, 2)

            # ---- Build targets and compute loss ----
            targets = build_targets(bboxes_teff, temporal_labels, labels, device)
            outputs = {
                'pred_logits': pred_logits,
                'pred_boxes': pred_boxes,
                'pred_temporal': pred_temporal,
            }
            loss_dict = criterion(outputs, targets)
            loss = sum(weight_dict[k] * loss_dict[k] for k in weight_dict)

            # ---- Backward pass and optimisation ----
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.1)
            optimizer.step()

            total_train_loss += loss.item()

            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch+1:03d} | Batch {batch_idx:04d} | "
                    f"loss={loss.item():.4f}  "
                    f"ce={loss_dict['loss_ce'].item():.4f}  "
                    f"bbox={loss_dict['loss_bbox'].item():.4f}  "
                    f"giou={loss_dict['loss_giou'].item():.4f}  "
                    f"temp={loss_dict['loss_temporal'].item():.4f}"
                )

        avg_train_loss = total_train_loss / len(train_loader)

        # ---------------------------------------------------------------------
        # Validation (loss on the test loader)
        # ---------------------------------------------------------------------
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                frames = batch['frames'].to(device)
                bboxes_xyxy = batch['bboxes'].to(device)
                labels = batch['label'].to(device)

                B, T = frames.shape[:2]
                T_eff = T // model.tube_t
                t_indices = [
                    min(t * model.tube_t + model.tube_t // 2, T - 1)
                    for t in range(T_eff)
                ]
                bboxes_teff_xyxy = bboxes_xyxy[:, t_indices, :]
                bboxes_teff = xyxy_to_cxcywh(bboxes_teff_xyxy)

                # Recompute temporal labels from validation boxes
                with torch.no_grad():
                    areas = ((bboxes_xyxy[..., 2] - bboxes_xyxy[..., 0]) *
                             (bboxes_xyxy[..., 3] - bboxes_xyxy[..., 1]))
                    has_annotation = areas < 0.98
                    temporal_labels = torch.zeros(B, 2, device=device)
                    for b in range(B):
                        valid_t = has_annotation[b].nonzero(as_tuple=True)[0]
                        if len(valid_t) > 0:
                            temporal_labels[b, 0] = valid_t[0].float() / max(T - 1, 1)
                            temporal_labels[b, 1] = valid_t[-1].float() / max(T - 1, 1)
                        else:
                            temporal_labels[b, 0] = 0.0
                            temporal_labels[b, 1] = 1.0

                pred_logits, pred_boxes, pred_temporal = model(frames)
                outputs = {
                    'pred_logits': pred_logits,
                    'pred_boxes': pred_boxes,
                    'pred_temporal': pred_temporal,
                }
                targets = build_targets(bboxes_teff, temporal_labels, labels, device)
                loss_dict = criterion(outputs, targets)
                loss = sum(weight_dict[k] * loss_dict[k] for k in weight_dict)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(test_loader)

        scheduler.step()
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        print(
            f"\nEpoch {epoch+1}/{args.epochs} | "
            f"train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f}\n"
        )

        # Save the model only if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ New best model saved (val_loss={avg_val_loss:.4f})")

    writer.close()
    print(f"Training complete. Best model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()