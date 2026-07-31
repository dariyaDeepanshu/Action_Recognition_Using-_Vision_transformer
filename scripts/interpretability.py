# =============================================================================
# INTERPRETABILITY, VISUALISATION, AND ERROR ANALYSIS (Part E)
# =============================================================================
# This script provides three interpretability analyses for the best VideoMAE
# classification model (trained on HMDB_simp) and optionally uses the JHMDB
# dataset for cross‑dataset feature visualisation and temporal contribution.
#
# Analyses implemented:
#   1. Attention map visualisation (Option 1):
#        - Extracts attention weights from a specified transformer layer.
#        - Overlays the average attention from CLS to patches on a middle frame.
#        - Compares attention patterns between correctly and incorrectly
#          classified test samples.
#   2. t‑SNE of CLS embeddings (Option 2):
#        - Extracts CLS token embeddings from the test sets of HMDB_simp and JHMDB.
#        - Projects them into 2D space with t‑SNE.
#        - Plots with markers for dataset origin (circle = HMDB, cross = JHMDB)
#          and colours by class label to assess feature overlap / transferability.
#   3. Temporal contribution analysis (Option 3):
#        - For a few JHMDB test videos, gradually reveals frames and records
#          model confidence.
#        - Overlays the ground‑truth action span derived from bounding box
#          annotations (non‑full‑frame boxes).
#        - Shows how quickly the model becomes confident relative to the
#          annotated action interval.
#
# Usage:
#     python scripts/interpretability.py                 # run all three
#     python scripts/interpretability.py --attention_only
#     python scripts/interpretability.py --tsne_only
#     python scripts/interpretability.py --temporal_only
# =============================================================================

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import cv2
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")

sys.path.append('.')

from data.hmdb_dataset import get_dataloaders
from data.jhmdb_dataset import get_jhmdb_dataloaders
from models.videomae import VideoMAE
from training.trainer import set_seed
from configs.config_utils import load_config
from utils.run_utils import make_run_dir


# -----------------------------------------------------------------------------
# Helper: Wrapper for attention extraction
# -----------------------------------------------------------------------------
class VideoMAEWrapper(nn.Module):
    """
    Wrapper around the Hugging Face VideoMAE model that captures attention
    weights from a specified transformer layer and returns the CLS embedding.
    """

    def __init__(self, model: torch.nn.Module, layer_idx: int = -1):
        """
        Args:
            model    : Trained VideoMAE model (instance of VideoMAE).
            layer_idx: Index of the encoder layer from which to extract attention
                       (0-11, default -1 = last layer).
        """
        super().__init__()
        # Direct reference to the underlying Hugging Face model
        self.model = model.model
        self.layer_idx = layer_idx
        self.attentions = []           # store attention weights
        self._register_hooks()

    def _register_hooks(self):
        """Register a forward hook on the target encoder layer."""
        encoder_layers = self.model.videomae.encoder.layer
        target_layer = encoder_layers[self.layer_idx]

        def forward_hook(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                # output[1] contains the attention probabilities
                self.attentions.append(output[1].detach())

        target_layer.register_forward_hook(forward_hook)

    def forward(self, pixel_values: torch.Tensor):
        """
        Forward pass: obtain logits, CLS embedding, and attention from the
        hooked layer.

        Returns:
            logits  : classification logits (B, num_classes)
            cls_emb : CLS token embedding (B, d_model)
            attn    : attention weights of the target layer, or None if not captured.
        """
        outputs = self.model(
            pixel_values=pixel_values,
            output_attentions=True,
            output_hidden_states=True,
        )
        logits = outputs.logits
        last_hidden = outputs.hidden_states[-1]   # last layer hidden states
        cls_emb = last_hidden[:, 0, :]            # CLS token
        attn = self.attentions[-1] if self.attentions else None
        return logits, cls_emb, attn


# -----------------------------------------------------------------------------
# Helper: Extract CLS embeddings for t‑SNE
# -----------------------------------------------------------------------------
def get_cls_embeddings(
    model: VideoMAE,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    is_hmdb: bool = True
) -> tuple:
    """
    Extract the CLS token embeddings from the underlying Hugging Face model
    for all samples in the dataloader.

    Args:
        model     : Trained VideoMAE instance.
        dataloader: DataLoader that yields (videos, labels) for HMDB or
                    dict with 'frames' and 'label' for JHMDB.
        device    : torch device.
        is_hmdb   : If True, the dataloader is for HMDB (tuple); otherwise JHMDB (dict).

    Returns:
        embeddings: numpy array of shape (N, d_model).
        labels    : numpy array of shape (N,).
    """
    embeddings = []
    labels = []
    hf_model = model.model   # underlying Hugging Face model
    hf_model.eval()

    with torch.no_grad():
        if is_hmdb:
            for videos, lbls in dataloader:
                videos = videos.to(device)
                outputs = hf_model(pixel_values=videos, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]      # (B, seq_len, d_model)
                cls_emb = last_hidden[:, 0, :].cpu().numpy() # (B, d_model)
                embeddings.extend(cls_emb)
                labels.extend(lbls.cpu().numpy())
        else:
            for batch in dataloader:
                videos = batch['frames'].to(device)
                outputs = hf_model(pixel_values=videos, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]
                cls_emb = last_hidden[:, 0, :].cpu().numpy()
                embeddings.extend(cls_emb)
                labels.extend(batch['label'].cpu().numpy())

    return np.array(embeddings), np.array(labels)


# -----------------------------------------------------------------------------
# 1. Attention map visualisation (correct vs. incorrect)
# -----------------------------------------------------------------------------
def visualize_attention(
    model_wrapper: VideoMAEWrapper,
    test_loader: torch.utils.data.DataLoader,
    class_names: list,
    device: torch.device,
    save_dir: str,
    num_samples: int = 10
) -> None:
    """
    Extract attention maps for a batch of test samples, split into correctly
    and incorrectly classified examples, and save a side-by-side comparison.

    The attention map is the average over heads of CLS-to-patch attention from
    the specified layer, averaged over the temporal dimension (for VideoMAE).
    The result is upsampled to the frame size and overlaid on the middle frame
    of the clip.

    Args:
        model_wrapper: VideoMAEWrapper instance (with hooks).
        test_loader  : DataLoader for HMDB_simp test set.
        class_names  : list of class names (for titles).
        device       : torch device.
        save_dir     : directory to save the output image.
        num_samples  : number of correct and incorrect examples to show
                       (the script collects up to `num_samples` of each).
    """
    os.makedirs(save_dir, exist_ok=True)
    model_wrapper.eval()

    correct_attns = []
    incorrect_attns = []
    collected = {'correct': 0, 'incorrect': 0}

    with torch.no_grad():
        for videos, labels in test_loader:
            if collected['correct'] >= num_samples and collected['incorrect'] >= num_samples:
                break
            videos = videos.to(device)
            logits, cls_emb, attn = model_wrapper(videos)
            preds = logits.argmax(dim=-1)

            for b in range(videos.size(0)):
                if collected['correct'] >= num_samples and collected['incorrect'] >= num_samples:
                    break
                is_correct = (preds[b] == labels[b]).item()
                if is_correct and collected['correct'] >= num_samples:
                    continue
                if not is_correct and collected['incorrect'] >= num_samples:
                    continue

                # --- Extract middle frame (denormalised) ---
                video = videos[b]                     # (T, C, H, W)
                t_mid = video.shape[0] // 2
                frame = video[t_mid].cpu().numpy().transpose(1, 2, 0)  # (H,W,C)

                mean = np.array([0.485, 0.456, 0.406])
                std  = np.array([0.229, 0.224, 0.225])
                frame = frame * std + mean
                frame = np.clip(frame, 0, 1)

                # --- Compute attention map ---
                if attn is not None:
                    H_p = W_p = 14                     # spatial patches per slice (224/16)
                    tube_t = 2                         # VideoMAE base tube temporal size
                    T_eff = video.shape[0] // tube_t    # number of temporal slices
                    att_map = attn[b].mean(dim=0)       # (seq_len, seq_len) avg over heads
                    cls_att = att_map[0, 1:]            # CLS → patches

                    spatial_tokens = H_p * W_p
                    expected = T_eff * spatial_tokens
                    actual = cls_att.shape[0]

                    # Defensive reshape: if token count mismatches, fall back to first 196 tokens
                    if actual != expected:
                        cls_att = cls_att[:spatial_tokens]
                        att_2d = cls_att.reshape(H_p, W_p).cpu().numpy()
                    else:
                        att_3d = cls_att.reshape(T_eff, H_p, W_p)
                        att_2d = att_3d.mean(axis=0).cpu().numpy()

                    att_grid = cv2.resize(att_2d, (frame.shape[1], frame.shape[0]))
                else:
                    att_grid = np.zeros_like(frame[:, :, 0])

                # --- Store ---
                item = {
                    'frame': frame,
                    'att': att_grid,
                    'label': labels[b].item(),
                    'pred': preds[b].item()
                }
                if is_correct:
                    correct_attns.append(item)
                    collected['correct'] += 1
                else:
                    incorrect_attns.append(item)
                    collected['incorrect'] += 1

    # --- Create figure: 2 rows (correct, incorrect) × num_samples columns ---
    fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 6))

    for i in range(num_samples):
        # Row 0: correct examples (frame + attention overlay)
        if i < len(correct_attns):
            c = correct_attns[i]
            axes[0, i].imshow(c['frame'])
            axes[0, i].imshow(c['att'], alpha=0.5, cmap='jet')
            axes[0, i].set_title(
                f"GT: {class_names[c['label']]}\nPred: {class_names[c['pred']]}",
                fontsize=8
            )
            axes[0, i].axis('off')
        else:
            axes[0, i].axis('off')

        # Row 1: incorrect examples
        if i < len(incorrect_attns):
            ic = incorrect_attns[i]
            axes[1, i].imshow(ic['frame'])
            axes[1, i].imshow(ic['att'], alpha=0.5, cmap='jet')
            axes[1, i].set_title(
                f"GT: {class_names[ic['label']]}\nPred: {class_names[ic['pred']]}",
                fontsize=8
            )
            axes[1, i].axis('off')
        else:
            axes[1, i].axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, "attention_correct_vs_incorrect.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved attention comparison to {save_path}")


# -----------------------------------------------------------------------------
# 2. t‑SNE feature space visualisation (cross‑dataset)
# -----------------------------------------------------------------------------
def tsne_features(
    model: VideoMAE,
    hmdb_loader: torch.utils.data.DataLoader,
    jhmdb_loader: torch.utils.data.DataLoader,
    device: torch.device,
    save_dir: str
) -> None:
    """
    Extract CLS embeddings from HMDB_simp and JHMDB test sets, run t-SNE,
    and plot the 2D projection. Colours indicate class label; markers indicate
    dataset (circle = HMDB, cross = JHMDB). This visualisation shows how well
    the learned features align across the two datasets (transferability).

    Args:
        model       : Trained VideoMAE model.
        hmdb_loader : DataLoader for HMDB_simp test set.
        jhmdb_loader: DataLoader for JHMDB test set.
        device      : torch device.
        save_dir    : Directory to save the output image.
    """
    print("Extracting CLS embeddings from HMDB_simp...")
    hmdb_emb, hmdb_labels = get_cls_embeddings(model, hmdb_loader, device, is_hmdb=True)

    print("Extracting CLS embeddings from JHMDB...")
    jhmdb_emb, jhmdb_labels = get_cls_embeddings(model, jhmdb_loader, device, is_hmdb=False)

    # Combine embeddings and labels
    embeddings = np.vstack([hmdb_emb, jhmdb_emb])
    labels = np.concatenate([hmdb_labels, jhmdb_labels])
    dataset_names = ['HMDB'] * len(hmdb_emb) + ['JHMDB'] * len(jhmdb_emb)

    print(f"Total samples: {len(embeddings)}. Running t‑SNE (may take a while)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb_2d = tsne.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(14, 10))
    unique_classes = sorted(set(labels))
    cmap = matplotlib.colormaps['tab20'].resampled(len(unique_classes))

    for cls in unique_classes:
        mask = labels == cls
        # HMDB points (circles)
        idx_hmdb = mask & (np.array(dataset_names) == 'HMDB')
        if np.any(idx_hmdb):
            plt.scatter(
                emb_2d[idx_hmdb, 0], emb_2d[idx_hmdb, 1],
                marker='o', s=15, alpha=0.7, color=cmap(cls),
                label=f'Class {cls} (HMDB)'
            )
        # JHMDB points (crosses)
        idx_jhmdb = mask & (np.array(dataset_names) == 'JHMDB')
        if np.any(idx_jhmdb):
            plt.scatter(
                emb_2d[idx_jhmdb, 0], emb_2d[idx_jhmdb, 1],
                marker='x', s=15, alpha=0.7, color=cmap(cls),
                label=f'Class {cls} (JHMDB)'
            )

    plt.title(
        "t-SNE of CLS embeddings (HMDB_simp + JHMDB)\n"
        "Same colour = same class, circle = HMDB, cross = JHMDB"
    )
    # Deduplicate legend entries
    handles, labels_leg = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_leg, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=8, ncol=2)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "tsne_cross_dataset.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved cross-dataset t-SNE to {save_path}")


# -----------------------------------------------------------------------------
# 3. Temporal contribution analysis (using JHMDB)
# -----------------------------------------------------------------------------
def temporal_contribution(
    model: VideoMAE,
    jhmdb_loader: torch.utils.data.DataLoader,
    device: torch.device,
    save_dir: str,
    num_videos: int = 5
) -> None:
    """
    For a small number of JHMDB test videos, gradually increase the number of
    frames shown and plot the model's confidence (maximum class probability).

    The ground-truth action span is derived from the bounding box annotations:
    frames where at least one non-fallback box is present (area < 0.98) are
    considered the action interval. This interval is shaded in green to show
    when the action actually occurs. The plot helps assess whether the model
    becomes confident early, coinciding with the action onset.

    Args:
        model        : Trained VideoMAE model.
        jhmdb_loader : DataLoader for JHMDB test set.
        device       : torch device.
        save_dir     : Directory to save the plots.
        num_videos   : Number of videos to analyse.
    """
    os.makedirs(save_dir, exist_ok=True)
    hf_model = model.model   # underlying Hugging Face model
    hf_model.eval()

    # Use a single‑sample loader to process videos one by one
    from torch.utils.data import DataLoader
    single_loader = DataLoader(
        jhmdb_loader.dataset, batch_size=1, shuffle=False, num_workers=0
    )

    # Collect the first `num_videos` videos
    videos_sampled, labels_sampled, bboxes_sampled = [], [], []
    for i, batch in enumerate(single_loader):
        if i >= num_videos:
            break
        videos_sampled.append(batch['frames'].to(device))
        labels_sampled.append(batch['label'])
        bboxes_sampled.append(batch['bboxes'])      # (1, T, 4)

    for idx, (video, lbl, bboxes) in enumerate(
            zip(videos_sampled, labels_sampled, bboxes_sampled)):
        T = video.shape[1]   # number of frames (e.g., 16)
        confidences = []

        with torch.no_grad():
            for t in range(1, T + 1):
                # Use only the first t frames; pad with last frame to keep length T
                if t < T:
                    subclip = video[:, :t]
                    pad = video[:, -1:].repeat(1, T - t, 1, 1, 1)
                    subclip = torch.cat([subclip, pad], dim=1)
                else:
                    subclip = video
                outputs = hf_model(pixel_values=subclip)
                probs = torch.softmax(outputs.logits, dim=-1)
                conf = probs.max(dim=-1).values.item()
                confidences.append(conf)

        # Derive ground‑truth action interval from bounding boxes
        bboxes_np = bboxes[0].cpu().numpy()   # (T, 4) normalised xyxy
        # Box area = (x2-x1)*(y2-y1)
        areas = (bboxes_np[:, 2] - bboxes_np[:, 0]) * (bboxes_np[:, 3] - bboxes_np[:, 1])
        # Fallback full‑frame boxes have area ~1.0; real annotations have area < 0.98
        valid_frames = np.where(areas < 0.98)[0]
        if len(valid_frames) > 0:
            t_start_gt = int(valid_frames[0]) + 1   # 1‑indexed for x‑axis
            t_end_gt   = int(valid_frames[-1]) + 1
        else:
            t_start_gt, t_end_gt = 1, T

        # Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, T + 1), confidences, marker='o', linestyle='-',
                label='Model confidence')
        ax.axvspan(t_start_gt, t_end_gt, alpha=0.12, color='green',
                   label=f'GT action span (frames {t_start_gt}–{t_end_gt})')
        ax.axvline(t_start_gt, color='green', linestyle='--', linewidth=1)
        ax.axvline(t_end_gt,   color='green', linestyle='--', linewidth=1)
        ax.set_xlabel("Number of frames revealed")
        ax.set_ylabel("Confidence (max class prob)")
        ax.set_title(
            f"Temporal contribution – JHMDB video {idx + 1} "
            f"(GT class {lbl[0].item()})"
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        save_path = os.path.join(save_dir, f"temporal_contrib_video{idx + 1}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved temporal contribution plot to {save_path}")

    print("\nNote: JHMDB clips are trimmed to the action, so the GT span covers most")
    print("or all frames. The shaded region shows when the model is 'in the action'.")
    print("Confidence rising before or at the GT boundary confirms early recognition.")


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Interpretability analysis (Part E)")
    parser.add_argument(
        "--config", type=str, default="configs/videomae.yaml",
        help="Model configuration file (YAML)."
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/videomae_best.pt",
        help="Trained VideoMAE checkpoint (.pt)."
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on ('cuda' or 'cpu')."
    )
    parser.add_argument(
        "--attention_only", action="store_true",
        help="Run only attention map visualisation."
    )
    parser.add_argument(
        "--tsne_only", action="store_true",
        help="Run only t‑SNE visualisation."
    )
    parser.add_argument(
        "--temporal_only", action="store_true",
        help="Run only temporal contribution analysis."
    )
    parser.add_argument(
        "--num_samples", type=int, default=5,
        help="Number of samples for attention or temporal analysis."
    )
    parser.add_argument(
        "--layer", type=int, default=11,
        help="Transformer layer index for attention extraction (0‑11)."
    )
    parser.add_argument(
        "--run_tag", type=str, default=None,
        help="Optional extra tag identifying this run, used in the output "
             "directory name."
    )
    args = parser.parse_args()

    set_seed(42)
    device = torch.device(args.device)

    # Load configuration and dataset parameters
    config = load_config(args.config)
    data_cfg = config.get("data", {})
    model_name = config.get("model_name", "videomae")
    num_frames = config.get("num_frames", 16)
    batch_size = 4        # small for interpretability

    run_dir = make_run_dir("interpretability", model_name, args.run_tag)
    attention_dir = os.path.join(run_dir, "attention")
    temporal_dir = os.path.join(run_dir, "temporal")

    # HMDB_simp test loader (classification)
    _, _, hmdb_test_loader, class_names = get_dataloaders(
        dataset_root=data_cfg.get("dataset_root", "HMDB_simp"),
        model_name=model_name,
        batch_size=batch_size,
        seed=42,
        sampling="uniform",
        num_frames=num_frames,
        num_workers=2,
        augment_mode=None,
    )

    # JHMDB test loader (for cross‑dataset t‑SNE and temporal contribution)
    _, jhmdb_test_loader, _ = get_jhmdb_dataloaders(
        jhmdb_root="JHMDB_simp",
        num_frames=num_frames,
        batch_size=batch_size,
        num_workers=2,
    )

    # Load the trained VideoMAE model
    model = VideoMAE(num_classes=len(class_names))
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)

    # Wrap for attention extraction
    wrapper = VideoMAEWrapper(model, layer_idx=args.layer).to(device)

    # Run selected analyses
    if args.attention_only:
        visualize_attention(
            wrapper, hmdb_test_loader, class_names, device,
            attention_dir, args.num_samples
        )
    elif args.tsne_only:
        tsne_features(model, hmdb_test_loader, jhmdb_test_loader, device, run_dir)
    elif args.temporal_only:
        temporal_contribution(
            model, jhmdb_test_loader, device, temporal_dir, args.num_samples
        )
    else:
        print("Running all interpretability analyses...")
        visualize_attention(
            wrapper, hmdb_test_loader, class_names, device,
            attention_dir, args.num_samples
        )
        tsne_features(model, hmdb_test_loader, jhmdb_test_loader, device, run_dir)
        temporal_contribution(
            model, jhmdb_test_loader, device, temporal_dir, args.num_samples
        )
        print(f"All done. Results saved in '{run_dir}'.")


if __name__ == "__main__":
    main()