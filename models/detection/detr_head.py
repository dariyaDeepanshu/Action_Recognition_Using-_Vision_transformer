# =============================================================================
# DETR‑style Spatio‑temporal Action Localisation Head (Part D, Option 3)
# =============================================================================
# This module implements an end‑to‑end spatio‑temporal action detector using
# a DETR‑style transformer decoder. It builds upon a frozen VideoMAE backbone
# (taken from the best Part A classification model) and adds:
#   - Learnable spatiotemporal positional encodings.
#   - A transformer decoder with learnable action queries.
#   - Three prediction heads: classification, bounding‑box tubelet, temporal extent.
#
# The design follows the specification of Part D, Option 3 (DETR‑style detector)
# and satisfies the requirement to output bounding boxes, class labels, and
# temporal extents in a single forward pass.
# =============================================================================

import os
import warnings

import torch
import torch.nn as nn
from transformers import AutoModelForVideoClassification, AutoConfig

# Path to the Part A fine‑tuned VideoMAE classification checkpoint.
# The backbone weights are loaded from here so that the localisation head
# benefits from the action‑specific representations learned in Part A.
VIDEOMAE_CHECKPOINT = "checkpoints/videomae_best.pt"


class DETRLocalisation(nn.Module):
    """
    End-to-end spatio-temporal action detector (DETR‑style).

    Architecture overview:
        1. VideoMAE backbone (frozen) produces a flat sequence of spatiotemporal
           patch tokens: (B, 1 + T_eff*S, d_model). The CLS token is discarded.
        2. Spatiotemporal positional encoding PE(t, s) = temporal_PE[t] + spatial_PE[s]
           is added to each patch token, giving unique identities across space and time.
        3. A transformer decoder receives learnable action queries (num_queries) and
           the positionally encoded patch tokens as memory. Queries attend to all
           patches simultanously (true spatiotemporal attention).
        4. Three prediction heads are applied to the decoder outputs:
           - Classification head: one class label per query (including background).
           - Bounding-box tubelet head: one normalised (cx, cy, w, h) box per temporal slice.
           - Temporal extent head: one normalised (t_start, t_end) interval per query.

    Args:
        backbone_name      : HuggingFace model ID for VideoMAE (used as fallback).
        num_classes        : Number of foreground action classes. Background class
                             automatically added (index = num_classes).
        num_queries        : Number of learnable action queries.
        d_model            : Transformer hidden dimension (must match backbone).
        nhead              : Number of attention heads in decoder layers.
        num_decoder_layers : Depth of the DETR transformer decoder.
        dropout            : Dropout rate inside decoder layers.
        freeze_backbone    : If True (default), backbone weights are frozen.
        tube_t             : Temporal tube size of the VideoMAE patch embed (default 2).
        patch_size         : Spatial patch size in pixels (default 16).
        img_size           : Spatial resolution of input frames (default 224).
        max_t_eff          : Maximum number of temporal slices (supports variable clip length).
    """

    def __init__(
        self,
        backbone_name: str = "MCG-NJU/videomae-base-finetuned-kinetics",
        num_classes: int = 21,
        num_queries: int = 10,
        d_model: int = 768,
        nhead: int = 8,
        num_decoder_layers: int = 3,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        tube_t: int = 2,
        patch_size: int = 16,
        img_size: int = 224,
        max_t_eff: int = 32,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.d_model = d_model
        self.tube_t = tube_t
        self.max_t_eff = max_t_eff

        # Number of spatial patches per temporal slice (e.g., 224/16 = 14 -> 196)
        self.spatial_patches = (img_size // patch_size) ** 2

        # ---------------------------------------------------------------------
        # 1. VideoMAE backbone (with hidden state output)
        # ---------------------------------------------------------------------
        config = AutoConfig.from_pretrained(backbone_name)
        config.num_labels = num_classes
        config.output_hidden_states = True   # needed to access patch tokens
        self.backbone = AutoModelForVideoClassification.from_pretrained(
            backbone_name,
            config=config,
            ignore_mismatched_sizes=True,
        )

        # Load Part A fine‑tuned backbone weights (if available)
        if os.path.isfile(VIDEOMAE_CHECKPOINT):
            state = torch.load(VIDEOMAE_CHECKPOINT, map_location="cpu")
            # Handle possible "backbone." prefix in state dict
            if any(k.startswith("backbone.") for k in state):
                state = {
                    k.replace("backbone.", ""): v
                    for k, v in state.items()
                    if k.startswith("backbone.")
                }
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            print(
                f"[DETRLocalisation] Loaded Part A backbone from "
                f"'{VIDEOMAE_CHECKPOINT}'. "
                f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}"
            )
        else:
            warnings.warn(
                f"[DETRLocalisation] Part A checkpoint '{VIDEOMAE_CHECKPOINT}' "
                "not found. Using HuggingFace pretrained weights as fallback. "
                "Run Part A training first for best localisation performance.",
                UserWarning,
            )

        # Freeze backbone if required (default: frozen)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ---------------------------------------------------------------------
        # 2. Spatiotemporal positional encodings
        # ---------------------------------------------------------------------
        # PE(t, s) = temporal_pe[t] + spatial_pe[s]
        self.temporal_pe = nn.Embedding(max_t_eff, d_model)
        self.spatial_pe = nn.Embedding(self.spatial_patches, d_model)

        # ---------------------------------------------------------------------
        # 3. DETR transformer decoder
        # ---------------------------------------------------------------------
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.query_embed = nn.Parameter(torch.randn(num_queries, d_model))

        # ---------------------------------------------------------------------
        # 4. Prediction heads
        # ---------------------------------------------------------------------
        # Class head (background class is index num_classes)
        self.class_embed = nn.Linear(d_model, num_classes + 1)

        # Bounding‑box tubelet head – predicts max_t_eff boxes per query,
        # then trimmed to actual T_eff at runtime.
        self.bbox_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, max_t_eff * 4),
        )

        # Temporal extent head – predicts (t_start, t_end) normalised to [0,1]
        self.temporal_embed = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2),
        )

    # -------------------------------------------------------------------------
    # Helper: build flat spatiotemporal memory with positional encoding
    # -------------------------------------------------------------------------
    def get_spatiotemporal_memory(self, pixel_values: torch.Tensor):
        """
        Extract patch tokens from the VideoMAE encoder and add spatiotemporal
        positional encodings, producing a flat memory tensor.

        Args:
            pixel_values : (B, T, C, H, W) input video clip.

        Returns:
            memory : (B, T_eff * S, d_model) positionally encoded patch tokens.
            T_eff  : number of temporal slices (T // tube_t).
        """
        B, T = pixel_values.shape[:2]
        T_eff = T // self.tube_t
        expected_total = T_eff * self.spatial_patches

        # Forward through VideoMAE encoder
        outputs = self.backbone.videomae(
            pixel_values=pixel_values,
            output_hidden_states=True,
        )
        # Remove CLS token -> (B, T_eff * S, d_model)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]
        total_tokens = patch_tokens.shape[1]

        # Defensive trim / pad to match expected total (graceful handling)
        if total_tokens != expected_total:
            warnings.warn(
                f"[DETRLocalisation] Token count mismatch: got {total_tokens}, "
                f"expected {expected_total} (T={T}, tube_t={self.tube_t}, "
                f"T_eff={T_eff}, spatial_patches={self.spatial_patches}). "
                "Trimming/padding - ensure num_frames is divisible by tube_t.",
                UserWarning,
            )
            if total_tokens > expected_total:
                patch_tokens = patch_tokens[:, :expected_total, :]
            else:
                pad = torch.zeros(
                    B, expected_total - total_tokens, self.d_model,
                    device=patch_tokens.device,
                    dtype=patch_tokens.dtype,
                )
                patch_tokens = torch.cat([patch_tokens, pad], dim=1)

        # Create spatiotemporal positional encoding
        # Sequence order: t=0: s=0..S-1, t=1: s=0..S-1, ...
        device = patch_tokens.device
        t_ids = torch.arange(T_eff, device=device).repeat_interleave(self.spatial_patches)
        s_ids = torch.arange(self.spatial_patches, device=device).repeat(T_eff)
        # Clamp to avoid index out of range (max_t_eff)
        t_ids = t_ids.clamp(max=self.max_t_eff - 1)

        pos_enc = self.temporal_pe(t_ids) + self.spatial_pe(s_ids)   # (T_eff*S, d_model)
        memory = patch_tokens + pos_enc.unsqueeze(0)                # (B, T_eff*S, d_model)
        return memory, T_eff

    # -------------------------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------------------------
    def forward(self, pixel_values: torch.Tensor):
        """
        Run end-to-end spatio-temporal detection.

        Args:
            pixel_values : (B, T, C, H, W) input video clip.

        Returns:
            pred_logits   : (B, Q, num_classes+1) class logits.
            pred_boxes    : (B, Q, T_eff, 4) bounding boxes (cx,cy,w,h) normalised.
            pred_temporal : (B, Q, 2) temporal extents (t_start, t_end) normalised.
        """
        # 1. Build spatiotemporal memory
        memory, T_eff = self.get_spatiotemporal_memory(pixel_values)   # (B, T_eff*S, d_model)
        B = memory.shape[0]

        # 2. Expand learnable queries across batch
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)     # (B, Q, d_model)

        # 3. Transformer decoder – queries attend to all spatiotemporal tokens
        hs = self.decoder(queries, memory)                            # (B, Q, d_model)

        # 4. Prediction heads
        pred_logits = self.class_embed(hs)                            # (B, Q, C+1)

        # Bounding‑box tubelet: reshape and trim to actual T_eff
        boxes_flat = self.bbox_embed(hs)                              # (B, Q, max_t_eff*4)
        boxes_all = boxes_flat.view(B, self.num_queries, self.max_t_eff, 4)
        pred_boxes = boxes_all[:, :, :T_eff, :].sigmoid()             # (B, Q, T_eff, 4)

        # Temporal extents
        pred_temporal = self.temporal_embed(hs).sigmoid()             # (B, Q, 2)

        return pred_logits, pred_boxes, pred_temporal