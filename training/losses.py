# =============================================================================
# DETR SET PREDICTION LOSS FOR SPATIO‑TEMPORAL LOCALISATION (Part D, Option 3)
# =============================================================================
# This module implements the Hungarian matching and set prediction losses for
# the DETR‑style spatio‑temporal action detector (detr_head.py).
#
# The loss combines:
#   - Classification cross‑entropy (with down‑weighted background class).
#   - Bounding box loss: L1 + (1 - GIoU) averaged over temporal slices.
#   - Temporal extent loss: L1 on (t_start, t_end).
#
# Hungarian matching uses a cost matrix that integrates classification, box,
# and temporal differences to find the optimal bipartite assignment between
# predicted queries and ground‑truth action tubes.
#
# All tensors are assumed to be on the same device.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# -----------------------------------------------------------------------------
# Box format utilities
# -----------------------------------------------------------------------------
def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format.

    Args:
        x: Tensor of shape (..., 4), where the last dimension contains
           (cx, cy, w, h) normalised to [0, 1].

    Returns:
        Tensor of same shape with (x1, y1, x2, y2) in normalised coordinates.
    """
    x1 = x[..., 0] - x[..., 2] / 2
    y1 = x[..., 1] - x[..., 3] / 2
    x2 = x[..., 0] + x[..., 2] / 2
    y2 = x[..., 1] + x[..., 3] / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute the Generalised Intersection over Union (GIoU) matrix between two
    sets of boxes.

    GIoU = IoU - (enclosing_area - union) / enclosing_area, which is always in
    the range [-1, 1]. It provides a smoother gradient when boxes do not overlap.

    Args:
        boxes1: (N, 4) bounding boxes in (cx, cy, w, h) format, normalised.
        boxes2: (M, 4) bounding boxes in (cx, cy, w, h) format, normalised.

    Returns:
        giou: (N, M) tensor of GIoU values.
    """
    # Convert to (x1, y1, x2, y2) for easier geometry
    b1 = box_cxcywh_to_xyxy(boxes1)   # (N, 4)
    b2 = box_cxcywh_to_xyxy(boxes2)   # (M, 4)

    # Intersection rectangle
    lt = torch.max(b1[:, None, :2], b2[:, :2])          # (N, M, 2)
    rb = torch.min(b1[:, None, 2:], b2[:, 2:])          # (N, M, 2)
    wh = (rb - lt).clamp(min=0)                         # (N, M, 2)
    inter = wh[..., 0] * wh[..., 1]                     # (N, M)

    # Areas of each box
    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])  # (N,)
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])  # (M,)
    union = area1[:, None] + area2[None, :] - inter     # (N, M)
    iou = inter / (union + 1e-7)

    # Enclosing box
    en_lt = torch.min(b1[:, None, :2], b2[:, :2])       # (N, M, 2)
    en_rb = torch.max(b1[:, None, 2:], b2[:, 2:])       # (N, M, 2)
    en_wh = (en_rb - en_lt).clamp(min=0)                # (N, M, 2)
    en_area = en_wh[..., 0] * en_wh[..., 1]             # (N, M)

    giou = iou - (en_area - union) / (en_area + 1e-7)
    return giou


# -----------------------------------------------------------------------------
# Hungarian matcher
# -----------------------------------------------------------------------------
class HungarianMatcher(nn.Module):
    """
    Bipartite matching between predicted queries and ground-truth objects.

    For each ground-truth tube (class label, a sequence of T_eff boxes, and a
    temporal interval), we find the query that minimises the weighted sum of:
        1) Classification cost: -P(class = gt_class)
        2) Box cost:            L1 + (1 - GIoU) averaged over all T_eff slices
        3) Temporal cost:       L1 between predicted (t_start, t_end) and GT
    """

    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0,
                 cost_giou: float = 2.0, cost_temporal: float = 1.0):
        """
        Args:
            cost_class   : Weight for the classification cost.
            cost_bbox    : Weight for the L1 box cost.
            cost_giou    : Weight for the GIoU cost (1 - GIoU).
            cost_temporal: Weight for the L1 temporal cost.
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_temporal = cost_temporal

    @torch.no_grad()
    def forward(self, outputs: dict, targets: list) -> list:
        """
        Compute the Hungarian assignment for each batch element.

        Args:
            outputs: dict with keys:
                'pred_logits'   : (B, Q, C+1)
                'pred_boxes'    : (B, Q, T_eff, 4)
                'pred_temporal' : (B, Q, 2)
            targets: list of B dicts, each with:
                'labels'    : (num_gt,) LongTensor
                'boxes'     : (num_gt, T_eff, 4) FloatTensor
                'temporal'  : (num_gt, 2) FloatTensor

        Returns:
            indices: list of B tuples (src_idx, tgt_idx) where src_idx are the
                     query indices matched to each ground-truth object.
        """
        bs = outputs['pred_logits'].shape[0]
        out_prob = outputs['pred_logits'].softmax(-1)   # (B, Q, C+1)
        out_bbox = outputs['pred_boxes']                # (B, Q, T_eff, 4)
        out_temporal = outputs['pred_temporal']         # (B, Q, 2)

        indices = []
        for b in range(bs):
            tgt_labels = targets[b]['labels']          # (num_gt,)
            tgt_boxes = targets[b]['boxes']            # (num_gt, T_eff, 4)
            tgt_temporal = targets[b]['temporal']      # (num_gt, 2)
            num_gt = len(tgt_labels)

            if num_gt == 0:
                indices.append((torch.tensor([]), torch.tensor([])))
                continue

            # Classification cost
            cost_class = -out_prob[b, :, tgt_labels]   # (Q, num_gt)

            # Box cost (L1 + GIoU), averaged over temporal slices
            Q = out_bbox.shape[1]
            T_eff = out_bbox.shape[2]
            if tgt_boxes.ndim == 2:
                # If only one box is given (no temporal dimension), repeat it
                tgt_boxes = tgt_boxes.unsqueeze(1).expand(-1, T_eff, -1)
            cost_bbox_mat = torch.zeros(Q, num_gt, device=out_bbox.device)
            cost_giou_mat = torch.zeros(Q, num_gt, device=out_bbox.device)
            for t in range(T_eff):
                # L1 cost per slice
                cost_bbox_slice = torch.cdist(
                    out_bbox[b, :, t, :], tgt_boxes[:, t, :], p=1
                )  # (Q, num_gt)
                # GIoU cost per slice
                cost_giou_slice = torch.stack([
                    1 - generalized_box_iou(
                        out_bbox[b, :, t, :],
                        tgt_boxes[g, t, :].unsqueeze(0)
                    ).mean(dim=1)
                    for g in range(num_gt)
                ], dim=1)  # (Q, num_gt)
                cost_bbox_mat += cost_bbox_slice
                cost_giou_mat += cost_giou_slice
            cost_bbox_mat /= T_eff
            cost_giou_mat /= T_eff

            # Temporal cost
            cost_temporal = torch.cdist(out_temporal[b], tgt_temporal, p=1)  # (Q, num_gt)

            # Total cost matrix
            C = (self.cost_class * cost_class +
                 self.cost_bbox * cost_bbox_mat +
                 self.cost_giou * cost_giou_mat +
                 self.cost_temporal * cost_temporal)

            # Hungarian assignment (linear sum assignment)
            src_idx, tgt_idx = linear_sum_assignment(C.cpu().numpy())
            indices.append(
                (torch.as_tensor(src_idx, dtype=torch.long),
                 torch.as_tensor(tgt_idx, dtype=torch.long))
            )
        return indices


# -----------------------------------------------------------------------------
# Set criterion
# -----------------------------------------------------------------------------
class SetCriterion(nn.Module):
    """
    DETR set-prediction loss aggregating classification, box, and temporal losses.

    The loss is computed only over matched (query, ground-truth) pairs.
    Unmatched queries are pushed to the background class via classification loss
    but do not contribute to box or temporal losses.
    """

    def __init__(self, num_classes: int, matcher: HungarianMatcher,
                 weight_dict: dict, eos_coef: float = 0.1):
        """
        Args:
            num_classes: Number of foreground action classes (background added automatically).
            matcher     : HungarianMatcher instance.
            weight_dict : Dictionary mapping loss names to their weights.
            eos_coef    : Down-weighting factor for the background class in CE loss.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef

        # Create class weight vector: background gets eos_coef, all others get 1.0
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[num_classes] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs: dict, targets: list, indices: list) -> dict:
        """
        Compute the cross-entropy loss for class labels.

        For each matched query, the target class is the ground-truth class.
        Unmatched queries are assigned the background class (index = num_classes).
        """
        device = outputs['pred_logits'].device
        empty_weight = self.empty_weight.to(device)
        loss_ce_terms = []

        for b, (src_idx, tgt_idx) in enumerate(indices):
            src_logits = outputs['pred_logits'][b]      # (Q, C+1)
            # Default all queries to background
            target_classes = torch.full(
                (src_logits.shape[0],), self.num_classes,
                dtype=torch.long, device=device
            )
            if src_idx.numel() > 0:
                target_classes[src_idx] = targets[b]['labels'][tgt_idx].to(device)
            loss_ce_terms.append(
                F.cross_entropy(src_logits, target_classes, weight=empty_weight)
            )

        if loss_ce_terms:
            return {'loss_ce': torch.stack(loss_ce_terms).mean()}
        return {'loss_ce': torch.tensor(0.0, device=device, requires_grad=True)}

    def loss_boxes(self, outputs: dict, targets: list, indices: list) -> dict:
        """
        Compute L1 + (1 - GIoU) loss for matched bounding boxes, averaged over
        temporal slices.
        """
        device = outputs['pred_boxes'].device
        loss_bbox_terms = []
        loss_giou_terms = []

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue

            src_boxes = outputs['pred_boxes'][b][src_idx]   # (K, T_eff, 4)
            tgt_boxes = targets[b]['boxes'][tgt_idx].to(device)  # (K, T_eff, 4)

            T_eff = src_boxes.shape[1]
            bbox_loss = 0.0
            giou_loss = 0.0
            for t in range(T_eff):
                bbox_loss += F.l1_loss(src_boxes[:, t, :], tgt_boxes[:, t, :])
                giou_mat = generalized_box_iou(src_boxes[:, t, :], tgt_boxes[:, t, :])
                giou_loss += (1 - torch.diag(giou_mat)).mean()
            loss_bbox_terms.append(bbox_loss / T_eff)
            loss_giou_terms.append(giou_loss / T_eff)

        if loss_bbox_terms:
            return {
                'loss_bbox': torch.stack(loss_bbox_terms).mean(),
                'loss_giou': torch.stack(loss_giou_terms).mean(),
            }
        return {
            'loss_bbox': torch.tensor(0.0, device=device, requires_grad=True),
            'loss_giou': torch.tensor(0.0, device=device, requires_grad=True),
        }

    def loss_temporal(self, outputs: dict, targets: list, indices: list) -> dict:
        """
        Compute L1 loss on (t_start, t_end) for matched pairs.
        """
        device = outputs['pred_temporal'].device
        loss_temporal_terms = []

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            src_temporal = outputs['pred_temporal'][b][src_idx]    # (K, 2)
            tgt_temporal = targets[b]['temporal'][tgt_idx].to(device)  # (K, 2)
            loss_temporal_terms.append(F.l1_loss(src_temporal, tgt_temporal))

        if loss_temporal_terms:
            return {'loss_temporal': torch.stack(loss_temporal_terms).mean()}
        return {'loss_temporal': torch.tensor(0.0, device=device, requires_grad=True)}

    def forward(self, outputs: dict, targets: list) -> dict:
        """
        Combine all losses.

        Args:
            outputs: dict with 'pred_logits', 'pred_boxes', 'pred_temporal'
            targets: list of dicts with 'labels', 'boxes', 'temporal'

        Returns:
            losses: dict mapping loss names to scalar tensors.
        """
        indices = self.matcher(outputs, targets)
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices))
        losses.update(self.loss_boxes(outputs, targets, indices))
        losses.update(self.loss_temporal(outputs, targets, indices))
        return losses