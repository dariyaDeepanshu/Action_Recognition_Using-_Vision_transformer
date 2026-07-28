# =============================================================================
# HMDB_simp Dataset Loader
# =============================================================================
# This module handles the complete data pipeline for the HMDB_simp dataset:
#   1. Scanning the folder structure (class folders, video folders, JPG frames)
#   2. Stratified train/val/test split (70/15/15)
#   3. Frame sampling with three strategies (uniform, random, dense)
#   4. Spatial / temporal augmentations (for ablation studies)
#   5. PyTorch Dataset and DataLoader creation
#
# The same loader works for TimeSFormer and VideoMAE – model‑specific
# parameters (number of frames, image size) are taken from MODEL_CONFIGS.
# =============================================================================

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import cv2


# -----------------------------------------------------------------------------
# MODEL CONFIGURATIONS
# -----------------------------------------------------------------------------
# Each model requires a different clip length (number of frames) and input size.
# These defaults are used unless overridden by the `num_frames` / `image_size`
# arguments when creating the dataset (useful for ablation studies).
MODEL_CONFIGS = {
    "timesformer": {"num_frames": 8,   "image_size": 224, "sampling": "uniform"},
    "videomae":    {"num_frames": 16,  "image_size": 224, "sampling": "uniform"},
}


# -----------------------------------------------------------------------------
# 1. SCAN DATASET FOLDER
# -----------------------------------------------------------------------------
def scan_dataset(dataset_root: str):
    """
    Recursively scan the HMDB_simp root directory and collect all video folders
    with their corresponding class labels.

    Expected folder structure:
        dataset_root/
            brush_hair/
                video1/          # folder containing JPG frames
                    frame00001.jpg
                    frame00002.jpg
                    ...
                video2/...
            catch/...

    Args:
        dataset_root: Path to the root directory of HMDB_simp.

    Returns:
        video_paths: List of full paths to every video folder.
        labels:      List of integer class labels (one per video).
        class_names: Sorted list of class name strings.
        class_to_idx: Dictionary mapping class name → integer label.
    """
    print(f"\n[INFO] Scanning dataset at: {dataset_root}")

    # List all top‑level subdirectories (each is a class)
    class_names = sorted([
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    ])

    if not class_names:
        raise ValueError(f"No subfolders found in {dataset_root}. "
                         "Make sure the dataset is placed correctly.")

    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

    video_paths = []
    labels = []

    for class_name in class_names:
        class_folder = os.path.join(dataset_root, class_name)
        videos_in_class = 0

        # Each immediate subfolder inside the class folder is one video
        for video_folder in os.listdir(class_folder):
            video_path = os.path.join(class_folder, video_folder)
            if os.path.isdir(video_path):
                video_paths.append(video_path)
                labels.append(class_to_idx[class_name])
                videos_in_class += 1

        print(f"  {class_name:30s} → {videos_in_class} videos "
              f"(label={class_to_idx[class_name]})")

    print(f"\n[INFO] Total videos found: {len(video_paths)}")
    print(f"[INFO] Total classes found: {len(class_names)}")
    print(f"[INFO] Classes: {class_names}\n")

    return video_paths, labels, class_names, class_to_idx


# -----------------------------------------------------------------------------
# 2. STRATIFIED SPLIT (TRAIN / VAL / TEST)
# -----------------------------------------------------------------------------
def split_dataset(video_paths: list, labels: list, seed: int = 42):
    """
    Split the dataset into training (70%), validation (15%), and test (15%).
    The split is stratified by class labels to preserve the class distribution.

    Args:
        video_paths: List of video folder paths.
        labels:      List of integer labels (same length as video_paths).
        seed:        Random seed for reproducibility.

    Returns:
        Dictionary with keys 'train', 'val', 'test', each containing a dict
        with 'paths' and 'labels' lists.
    """
    # Split 70% train, 30% temporary (will be further split into val/test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        video_paths, labels,
        test_size=0.30,
        stratify=labels,
        random_state=seed
    )

    # Split the temporary 30% into two equal halves (15% val, 15% test)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=seed
    )

    print("[INFO] Dataset split:")
    print(f"  Train : {len(X_train)} videos "
          f"({len(X_train)/len(video_paths)*100:.1f}%)")
    print(f"  Val   : {len(X_val)} videos "
          f"({len(X_val)/len(video_paths)*100:.1f}%)")
    print(f"  Test  : {len(X_test)} videos "
          f"({len(X_test)/len(video_paths)*100:.1f}%)")

    return {
        "train": {"paths": X_train, "labels": y_train},
        "val":   {"paths": X_val,   "labels": y_val},
        "test":  {"paths": X_test,  "labels": y_test},
    }


# -----------------------------------------------------------------------------
# 3. FRAME LOADING HELPERS
# -----------------------------------------------------------------------------
def get_image_frames(folder_path: str, num_frames: int, image_size: int):
    """
    Load a sequence of uniformly spaced frames from a folder of JPG images.

    Args:
        folder_path: Path to a video folder containing sequentially named images
                     (e.g., 0001.jpg, 0002.jpg, ...).
        num_frames:  Number of frames to sample.
        image_size:  Target spatial size (height and width) for resizing.

    Returns:
        numpy.ndarray of shape (num_frames, image_size, image_size, 3), dtype=uint8.
        Returns black frames if no images are found.
    """
    # Collect all image files, sorted alphanumerically
    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if not image_files:
        # No images in folder – return black frames as a fallback
        return np.zeros((num_frames, image_size, image_size, 3), dtype=np.uint8)

    total = len(image_files)
    # Uniformly spaced indices (ensures we always take exactly `num_frames` frames)
    indices = np.linspace(0, total - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        img_path = os.path.join(folder_path, image_files[idx])
        frame = cv2.imread(img_path)
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # OpenCV BGR → RGB
            frame = cv2.resize(frame, (image_size, image_size))
            frames.append(frame)
        else:
            # Corrupted or unreadable image – insert a black frame
            frames.append(np.zeros((image_size, image_size, 3), dtype=np.uint8))

    # Pad if we somehow got fewer frames than requested (should not happen)
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((image_size, image_size, 3),
                                                         dtype=np.uint8))

    return np.stack(frames[:num_frames])


def load_frames_uniform(video_path: str, num_frames: int, image_size: int):
    """Uniform sampling (default)."""
    return get_image_frames(video_path, num_frames, image_size)


def load_frames_random(video_path: str, num_frames: int, image_size: int):
    """Random sampling (without replacement)."""
    image_files = sorted([
        f for f in os.listdir(video_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    total = len(image_files)
    if total == 0:
        return np.zeros((num_frames, image_size, image_size, 3), dtype=np.uint8)

    # Choose random indices and sort them to preserve temporal order
    indices = sorted(random.sample(range(total), min(num_frames, total)))
    frames = []
    for idx in indices:
        img_path = os.path.join(video_path, image_files[idx])
        frame = cv2.imread(img_path)
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (image_size, image_size))
            frames.append(frame)
        else:
            frames.append(np.zeros((image_size, image_size, 3), dtype=np.uint8))

    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((image_size, image_size, 3),
                                                         dtype=np.uint8))
    return np.stack(frames[:num_frames])


def load_frames_dense(video_path: str, num_frames: int, image_size: int):
    """
    Dense sampling: takes `num_frames` consecutive frames from the middle of the video.
    Useful for actions that happen quickly (e.g., catching a ball).
    """
    image_files = sorted([
        f for f in os.listdir(video_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    total = len(image_files)
    if total == 0:
        return np.zeros((num_frames, image_size, image_size, 3), dtype=np.uint8)

    # Centre the window
    start = max(0, total // 2 - num_frames // 2)
    indices = list(range(start, min(start + num_frames, total)))
    frames = []
    for idx in indices:
        img_path = os.path.join(video_path, image_files[idx])
        frame = cv2.imread(img_path)
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (image_size, image_size))
            frames.append(frame)
        else:
            frames.append(np.zeros((image_size, image_size, 3), dtype=np.uint8))

    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((image_size, image_size, 3),
                                                         dtype=np.uint8))
    return np.stack(frames[:num_frames])


# Mapping from string names to the actual sampling functions.
SAMPLING_FUNCTIONS = {
    "uniform": load_frames_uniform,
    "random":  load_frames_random,
    "dense":   load_frames_dense,
}


# -----------------------------------------------------------------------------
# 4. AUGMENTATION HELPERS
# -----------------------------------------------------------------------------
def augment_frames(frames: np.ndarray, augment_mode: str = None) -> np.ndarray:
    """
    Apply data augmentation (spatial or temporal) to a batch of frames.

    Args:
        frames: numpy array of shape (T, H, W, 3), uint8, values in [0, 255].
        augment_mode: One of:
            - None or "none" → no augmentation.
            - "spatial" → random horizontal flip, random crop+resize, brightness jitter.
            - "temporal" → randomly pick one of: temporal jitter, speed perturbation,
                           or reverse playback (applied to the frame order).

    Returns:
        Augmented frames (same shape, uint8, values clipped to [0,255]).
    """
    if augment_mode is None or augment_mode == "none":
        return frames

    T, H, W, C = frames.shape

    # --- Spatial augmentation ---
    if augment_mode == "spatial":
        # Horizontal flip (consistent across all frames)
        if random.random() > 0.5:
            frames = frames[:, :, ::-1, :].copy()

        # Random crop + resize (crop to 85‑100% of size, then resize back)
        if random.random() > 0.5:
            crop_ratio = random.uniform(0.85, 1.0)
            crop_h = int(H * crop_ratio)
            crop_w = int(W * crop_ratio)
            start_h = random.randint(0, H - crop_h)
            start_w = random.randint(0, W - crop_w)
            cropped = frames[:, start_h:start_h + crop_h, start_w:start_w + crop_w, :]
            # Resize each cropped frame back to (H, W)
            resized = np.stack([cv2.resize(f, (W, H)) for f in cropped])
            frames = resized

        # Brightness jitter (multiply by random factor between 0.8 and 1.2)
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            frames = np.clip(frames.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        return frames

    # --- Temporal augmentation ---
    if augment_mode == "temporal":
        # Randomly choose one of three temporal transforms
        choice = random.choice(["jitter", "speed", "reverse"])

        if choice == "jitter":
            # Shift the sequence by a small offset (±2 frames)
            shift = random.randint(-2, 2)
            indices = np.clip(np.arange(T) + shift, 0, T - 1)
            frames = frames[indices]

        elif choice == "speed":
            # Simulate speed change by resampling the timeline
            factor = random.uniform(0.8, 1.2)
            new_len = max(1, int(T * factor))
            new_indices = np.linspace(0, T - 1, new_len, dtype=int)
            # Interpolate back to original length (nearest neighbour)
            resampled = np.interp(np.arange(T),
                                  np.linspace(0, T - 1, new_len),
                                  new_indices).astype(int)
            frames = frames[resampled]

        else:   # "reverse"
            frames = frames[::-1].copy()

        return frames

    raise ValueError(f"Unknown augment_mode: {augment_mode}")


# -----------------------------------------------------------------------------
# 5. PYTORCH DATASET CLASS
# -----------------------------------------------------------------------------
class HMDBDataset(Dataset):
    """
    PyTorch Dataset for HMDB_simp (folder of JPG frames per video).

    Args:
        video_paths: List of paths to video folders.
        labels:      List of integer labels (same length).
        model_name:  One of 'timesformer', 'videomae'. Used to get default
                     values for number of frames and image size.
        sampling:    Sampling strategy ('uniform', 'random', 'dense').
        augment_mode: Augmentation mode (None, 'spatial', 'temporal') - applied only
                      to training set (the dataset does not know whether it's train;
                      the caller should set augment_mode accordingly).
        num_frames:  Override the default number of frames (for ablation studies).
        image_size:  Override the default spatial size.
    """
    def __init__(
        self,
        video_paths: list,
        labels: list,
        model_name: str = "timesformer",
        sampling: str = "uniform",
        augment_mode: str = None,
        num_frames: int = None,
        image_size: int = None,
    ):
        self.video_paths = video_paths
        self.labels = labels
        self.augment_mode = augment_mode

        # Get default config for the given model
        cfg = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["timesformer"])
        self.num_frames = num_frames if num_frames is not None else cfg["num_frames"]
        self.image_size = image_size if image_size is not None else cfg["image_size"]
        self.sampling_fn = SAMPLING_FUNCTIONS.get(sampling, load_frames_uniform)

        print("[INFO] HMDBDataset created:")
        print(f"  Model      : {model_name}")
        print(f"  Num frames : {self.num_frames}")
        print(f"  Image size : {self.image_size}x{self.image_size}")
        print(f"  Sampling   : {sampling}")
        print(f"  Augment    : {augment_mode}")
        print(f"  Videos     : {len(video_paths)}")

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # Load frames (numpy array of shape (T, H, W, 3), uint8)
        frames = self.sampling_fn(video_path, self.num_frames, self.image_size)

        # Apply augmentation (if any). The caller decides when augmentation is active.
        frames = augment_frames(frames, self.augment_mode)

        # Normalise to [0, 1] and convert to float
        frames = frames.astype(np.float32) / 255.0

        # Convert to PyTorch tensor: (T, H, W, C) -> (T, C, H, W)
        frames = torch.from_numpy(frames)
        frames = frames.permute(0, 3, 1, 2)

        # ImageNet normalisation (mean and std for RGB)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        frames = (frames - mean) / std

        label = torch.tensor(label, dtype=torch.long)
        return frames, label


# -----------------------------------------------------------------------------
# 6. CONVENIENCE FUNCTION: GET ALL THREE DATALOADERS
# -----------------------------------------------------------------------------
def get_dataloaders(
    dataset_root: str,
    model_name: str = "timesformer",
    batch_size: int = 4,
    seed: int = 42,
    sampling: str = "uniform",
    num_frames: int = None,
    num_workers: int = 2,
    augment_mode: str = None,
):
    """
    One-stop function to obtain train, validation, and test DataLoaders.

    Args:
        dataset_root : Path to HMDB_simp data.
        model_name   : Model name (used to get default number of frames and image size).
        batch_size   : Number of videos per batch.
        seed         : Random seed for reproducible data splits.
        sampling     : Frame sampling strategy ('uniform', 'random', 'dense').
        num_frames   : Override number of frames (for ablation studies).
        num_workers  : Number of subprocesses for data loading.
        augment_mode : Augmentation mode (None, 'spatial', 'temporal') – only applied
                       to the training loader.

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    # Scan and split the dataset
    video_paths, labels, class_names, _ = scan_dataset(dataset_root)
    splits = split_dataset(video_paths, labels, seed=seed)

    # Create Datasets
    train_dataset = HMDBDataset(
        splits["train"]["paths"],
        splits["train"]["labels"],
        model_name=model_name,
        sampling=sampling,
        augment_mode=augment_mode,   # augmentation only for training
        num_frames=num_frames,
    )

    val_dataset = HMDBDataset(
        splits["val"]["paths"],
        splits["val"]["labels"],
        model_name=model_name,
        sampling=sampling,
        augment_mode=None,           # no augmentation for validation
        num_frames=num_frames,
    )

    test_dataset = HMDBDataset(
        splits["test"]["paths"],
        splits["test"]["labels"],
        model_name=model_name,
        sampling=sampling,
        augment_mode=None,           # no augmentation for test
        num_frames=num_frames,
    )

    # Build DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,                # important for training
        num_workers=num_workers,
        pin_memory=True,             # faster GPU transfer
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print("\n[INFO] DataLoaders ready:")
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")
    print(f"  Test batches  : {len(test_loader)}")

    return train_loader, val_loader, test_loader, class_names


# -----------------------------------------------------------------------------
# TEST BLOCK (run only when this script is executed directly)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick test of the data loaders
    dataset_path = "./HMDB_simp"
    train_loader, _, _, _ = get_dataloaders(
        dataset_path, "timesformer", batch_size=2, num_workers=0
    )
    for frames, labels in train_loader:
        print(f"Test batch - frames shape: {frames.shape}, labels: {labels}")
        break