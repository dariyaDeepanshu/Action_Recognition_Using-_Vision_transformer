# =============================================================================
# JHMDB Dataset Loader (for spatio‑temporal localisation)
# =============================================================================
# This module loads the JHMDB dataset with per‑frame bounding box annotations
# derived from joint positions. It supports:
#   - Reading video files (avi, mp4, mov) from the "JHMDB_video" folder.
#   - Loading per‑frame bounding boxes from MATLAB .mat joint files.
#   - Using official train/test splits (from the "splits" folder).
#   - Providing PyTorch DataLoaders with video clips, bounding boxes, and labels.
#
# The dataset structure (inside JHMDB_simp):
#   JHMDB_video/
#       brush_hair/
#           video1.avi
#           video2.avi
#           ...
#       catch/...
#   joint_positions/
#       brush_hair/
#           video1.mat
#           video2.mat
#           ...
#   splits/
#       brush_hair_test_split1.txt
#       ...
# =============================================================================

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import scipy.io
from sklearn.model_selection import train_test_split


# -----------------------------------------------------------------------------
# Helper: load official splits (if available)
# -----------------------------------------------------------------------------
def load_jhmdb_splits(splits_root: str, class_name: str, split_num: int = 1):
    """
    Read the official JHMDB test split file for a given class.

    Each split file (e.g., 'brush_hair_test_split1.txt') contains lines:
        video_name.avi label
    where label = 1 for train, 2 for test, 0 for ignored.

    Args:
        splits_root: Path to the directory containing the split .txt files.
        class_name:  Name of the action class (e.g., 'brush_hair').
        split_num:   Split number (default 1).

    Returns:
        train_vids: List of video names (without .avi) for training.
        test_vids:  List of video names (without .avi) for testing.
        If the split file does not exist, returns (None, None).
    """
    split_file = os.path.join(splits_root, f"{class_name}_test_split{split_num}.txt")
    if not os.path.exists(split_file):
        return None, None

    train_vids, test_vids = [], []
    with open(split_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            vid = parts[0].replace('.avi', '')   # remove extension
            label = int(parts[1])
            if label == 1:
                train_vids.append(vid)
            elif label == 2:
                test_vids.append(vid)
    return train_vids, test_vids


# -----------------------------------------------------------------------------
# Bounding box extraction from joint positions (.mat files)
# -----------------------------------------------------------------------------
def load_bounding_boxes_from_joints(joint_folder: str, num_frames: int) -> np.ndarray:
    """
    Load per‑frame bounding boxes from a MATLAB .mat joint file.
    The bounding box is computed as the minimal rectangle enclosing all valid
    joints (x>0 or y>0). Coordinates are normalised to [0,1] using the original
    JHMDB frame size (320×240).

    Args:
        joint_folder: Path to the folder containing the .mat file for a video.
        num_frames:   Number of frames to sample (the returned array will have
                      this many bounding boxes, uniformly sampled from the
                      available joint frames).

    Returns:
        np.ndarray of shape (num_frames, 4) with (x1, y1, x2, y2) normalised.
        If the joint file is missing or cannot be read, returns a full‑frame box
        (0,0,1,1) for all frames.
    """
    bboxes = []

    # No joint folder → fallback full‑frame boxes
    if not os.path.exists(joint_folder):
        return np.tile([0.0, 0.0, 1.0, 1.0], (num_frames, 1)).astype(np.float32)

    # Find the .mat file (there should be exactly one per video)
    mat_files = [f for f in os.listdir(joint_folder) if f.endswith('.mat')]
    if not mat_files:
        return np.tile([0.0, 0.0, 1.0, 1.0], (num_frames, 1)).astype(np.float32)

    mat_path = os.path.join(joint_folder, mat_files[0])
    try:
        mat = scipy.loadmat(mat_path)
        pos = mat['pos_img']            # joint positions, shape (F,2,15) or (2,15,F)
    except Exception:
        return np.tile([0.0, 0.0, 1.0, 1.0], (num_frames, 1)).astype(np.float32)

    # Normalise the orientation to (F, 2, 15)
    if pos.ndim == 3:
        # Case 1: (F, 2, 15)
        if pos.shape[1] == 2 and pos.shape[2] == 15:
            total_frames = pos.shape[0]
            joints = pos
        # Case 2: (2, 15, F) → transpose to (F, 2, 15)
        elif pos.shape[0] == 2 and pos.shape[1] == 15:
            joints = np.transpose(pos, (2, 0, 1))
            total_frames = joints.shape[0]
        else:
            total_frames = 1
            joints = np.array([pos])
    else:
        total_frames = 1
        joints = np.array([pos])

    # Uniformly sample num_frames indices from the available joint frames
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for idx in indices:
        if idx < total_frames:
            fjoints = joints[idx]               # (2, 15)
            x = fjoints[0, :]
            y = fjoints[1, :]
            valid = (x > 0) | (y > 0)           # joints with positive coordinates
            if valid.any():
                # Compute bounding box in original (320×240) coordinates, then normalise
                x_min = np.min(x[valid]) / 320.0
                y_min = np.min(y[valid]) / 240.0
                x_max = np.max(x[valid]) / 320.0
                y_max = np.max(y[valid]) / 240.0
                # Clamp to [0,1]
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(1, x_max), min(1, y_max)
            else:
                # No valid joints – full frame
                x_min, y_min, x_max, y_max = 0.0, 0.0, 1.0, 1.0
        else:
            x_min, y_min, x_max, y_max = 0.0, 0.0, 1.0, 1.0
        bboxes.append([x_min, y_min, x_max, y_max])

    return np.array(bboxes, dtype=np.float32)


# -----------------------------------------------------------------------------
# Video frame loading
# -----------------------------------------------------------------------------
def load_jhmdb_frames(video_path: str, num_frames: int, image_size: int = 224) -> np.ndarray:
    """
    Extract uniformly sampled frames from a video file (.avi, .mp4, .mov).

    Args:
        video_path:  Path to the video file.
        num_frames:  Number of frames to sample.
        image_size:  Target spatial size (resized to square).

    Returns:
        np.ndarray of shape (num_frames, image_size, image_size, 3), dtype=uint8.
        Returns black frames if video cannot be read.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return np.zeros((num_frames, image_size, image_size, 3), dtype=np.uint8)

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (image_size, image_size))
            frames.append(frame)
        else:
            frames.append(np.zeros((image_size, image_size, 3), dtype=np.uint8))
    cap.release()

    # Pad if necessary (should not happen, but safe)
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((image_size, image_size, 3),
                                                         dtype=np.uint8))
    return np.stack(frames[:num_frames])


# -----------------------------------------------------------------------------
# Helper: locate the real video directory inside JHMDB_video
# -----------------------------------------------------------------------------
def _find_video_class_dir(videos_root: str, joint_classes: set) -> str:
    """
    Some JHMDB zip extractions add an extra wrapper folder (e.g.
    'ReCompress_Videos') and/or a junk '__MACOSX' folder inside 'JHMDB_video'.
    This walks down from videos_root (BFS) to find the directory whose
    subdirectories actually match the known action-class names.
    """
    queue = [videos_root]
    while queue:
        current = queue.pop(0)
        subdirs = [
            d for d in os.listdir(current)
            if os.path.isdir(os.path.join(current, d))
        ]
        if subdirs and len(set(subdirs) & joint_classes) >= max(1, len(subdirs) // 2):
            return current
        for d in subdirs:
            queue.append(os.path.join(current, d))
    return videos_root


# -----------------------------------------------------------------------------
# Dataset scanning and split creation
# -----------------------------------------------------------------------------
def scan_jhmdb(jhmdb_root: str):
    """
    Scan the JHMDB dataset structure, collect all video items, and split them
    into train / test sets using the official splits (or a fallback 80/20 split
    if official splits are missing).

    Args:
        jhmdb_root: Path to the JHMDB dataset root (contains 'JHMDB_video',
                    'joint_positions', and 'splits' subdirectories).

    Returns:
        train_items: List of dicts for training videos.
        test_items:  List of dicts for test videos.
        class_names: Sorted list of class names.
        class_to_idx: Dictionary mapping class name → integer index.
    """
    videos_root = os.path.join(jhmdb_root, "JHMDB_video")
    joints_root = os.path.join(jhmdb_root, "joint_positions")
    splits_root = os.path.join(jhmdb_root, "splits")

    if not os.path.exists(videos_root):
        raise FileNotFoundError(f"JHMDB_video not found at {videos_root}")

    # Classes are known from joint_positions; used to locate the real video
    # directory in case JHMDB_video has an extra wrapper/junk folder.
    joint_classes = {
        d for d in os.listdir(joints_root)
        if os.path.isdir(os.path.join(joints_root, d))
    } if os.path.exists(joints_root) else set()
    videos_root = _find_video_class_dir(videos_root, joint_classes)

    # List all action classes (subdirectories in JHMDB_video)
    class_names = [
        d for d in os.listdir(videos_root)
        if os.path.isdir(os.path.join(videos_root, d)) and not d.startswith('.')
        and not d.startswith('__')
    ]
    class_names.sort()
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    all_items = []   # list of dicts for all videos
    for class_name in class_names:
        class_video_dir = os.path.join(videos_root, class_name)
        class_joint_dir = os.path.join(joints_root, class_name)

        # Find all video files (avi, mp4, mov)
        video_files = [
            f for f in os.listdir(class_video_dir)
            if f.lower().endswith(('.avi', '.mp4', '.mov'))
        ]
        print(f"  {class_name}: {len(video_files)} videos")

        for vf in video_files:
            vid_name = os.path.splitext(vf)[0]
            video_path = os.path.join(class_video_dir, vf)
            joint_folder = os.path.join(class_joint_dir, vid_name)
            all_items.append({
                "video_path": video_path,
                "joint_folder": joint_folder,
                "label": class_to_idx[class_name],
                "class_name": class_name,
                "video_name": vid_name,
            })

    # Group items by class for split management
    from collections import defaultdict
    class_to_items = defaultdict(list)
    for item in all_items:
        class_to_items[item["class_name"]].append(item)

    train_items, test_items = [], []
    for class_name, items in class_to_items.items():
        train_vids, test_vids = load_jhmdb_splits(splits_root, class_name)
        if train_vids is not None:
            # Use official split
            for item in items:
                if item["video_name"] in train_vids:
                    train_items.append(item)
                elif item["video_name"] in test_vids:
                    test_items.append(item)
                else:
                    # If a video appears in neither list, put it in training
                    train_items.append(item)
        else:
            # No official split file – fallback to a stratified 80/20 split
            labels = [item["label"] for item in items]
            train_sub, test_sub = train_test_split(
                items, test_size=0.2, stratify=labels, random_state=42
            )
            train_items.extend(train_sub)
            test_items.extend(test_sub)

    print(f"\nTotal: {len(train_items)} train, {len(test_items)} test, "
          f"{len(class_names)} classes")
    return train_items, test_items, class_names, class_to_idx


# -----------------------------------------------------------------------------
# PyTorch Dataset
# -----------------------------------------------------------------------------
class JHMDBDataset(Dataset):
    """
    PyTorch Dataset for JHMDB with video frames, bounding boxes, and labels.

    Args:
        items:       List of item dicts (from scan_jhmdb).
        num_frames:  Number of frames per clip (default 16).
        image_size:  Spatial size (default 224).
    """
    def __init__(self, items: list, num_frames: int = 16, image_size: int = 224):
        self.items = items
        self.num_frames = num_frames
        self.image_size = image_size
        print(f"JHMDBDataset: {len(items)} videos, {num_frames} frames")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        # Load frames (np.uint8, shape (T, H, W, 3))
        frames = load_jhmdb_frames(item["video_path"], self.num_frames, self.image_size)

        # Load bounding boxes (normalised, shape (T, 4))
        bboxes = load_bounding_boxes_from_joints(item["joint_folder"], self.num_frames)

        # Ensure bboxes length matches num_frames (resample if necessary)
        if len(bboxes) != self.num_frames:
            indices = np.linspace(0, len(bboxes)-1, self.num_frames, dtype=int)
            bboxes = bboxes[indices]

        # Normalise frames to [0,1] and convert to (T, C, H, W)
        frames = frames.astype(np.float32) / 255.0
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)

        # ImageNet normalisation
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        frames = (frames - mean) / std

        bboxes = torch.from_numpy(bboxes).float()
        label = torch.tensor(item["label"], dtype=torch.long)

        return {
            "frames": frames,
            "bboxes": bboxes,
            "label": label,
            "class_name": item["class_name"],
            "video_name": item["video_name"],
        }


# -----------------------------------------------------------------------------
# Convenience function: get train and test DataLoaders
# -----------------------------------------------------------------------------
def get_jhmdb_dataloaders(jhmdb_root: str, num_frames: int = 16,
                          batch_size: int = 2, num_workers: int = 2):
    """
    Create DataLoaders for training and testing on JHMDB.

    Args:
        jhmdb_root:  Path to the JHMDB dataset root.
        num_frames:  Number of frames per clip.
        batch_size:  Batch size (videos per batch).
        num_workers: Number of subprocesses for data loading.

    Returns:
        train_loader, test_loader, class_names.
    """
    train_items, test_items, class_names, _ = scan_jhmdb(jhmdb_root)

    train_ds = JHMDBDataset(train_items, num_frames)
    test_ds  = JHMDBDataset(test_items, num_frames)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader, class_names


# -----------------------------------------------------------------------------
# Test block (when executed directly)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick test: load first batch from the test set
    train_loader, test_loader, class_names = get_jhmdb_dataloaders(
        "JHMDB_simp", num_frames=16, batch_size=2, num_workers=0
    )
    print("\n[TEST] Sample batch from test loader:")
    for batch in test_loader:
        print("  Frames shape:", batch["frames"].shape)
        print("  BBoxes shape:", batch["bboxes"].shape)
        print("  Labels:", batch["label"])
        break