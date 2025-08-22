"""Utility functions for event-based data preprocessing and augmentation.

Includes voxelization, histogram conversion, augmentation strategies and
video frame composition helpers used across training and visualization.
"""
import torch
import cv2
from typing import Tuple, List
import os 
import urllib.request

def _download_checkpoint(url: str, dest: str) -> bool:
    """Download a checkpoint file with a simple progress indicator.

    Returns True if successful, False otherwise.
    """
    # try:
    print(f"Downloading pretrained checkpoint from {url} to {dest}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with urllib.request.urlopen(url) as response, open(dest, "wb") as out_file:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 8192
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = downloaded / total * 100
                print(
                    f"\rDownloading checkpoint: {percent:5.1f}% ({downloaded}/{total} bytes)",
                    end="",
                )
    print("\nDownload complete.")
    return True
    # except Exception as e:  # pragma: no cover - network errors
    #     print(f"Failed to download checkpoint: {e}")
    return False

def apply_event_augmentations(
    events: torch.Tensor,
    training: bool = True,
    aug_prob: float = 0.5,
    width: int = 346,
    height: int = 260,
) -> torch.Tensor:
    """Stochastic data augmentations for raw event tensors.

    Args:
        events: [B, N, 4] with columns (t, x, y, p)
        training: Toggle augmentations
        aug_prob: Probability to apply any augmentations at all
        width/height: Sensor resolution used for clamping coordinates

    Returns:
        Augmented tensor (shape preserved).
    """
    if not training or torch.rand(1).item() > aug_prob:
        return events
    B, N, _ = events.shape
    device = events.device
    augmented = events.clone()

    # Temporal jitter (simulate timestamp noise)
    if torch.rand(1).item() < 0.4:
        time_jitter = torch.randn_like(augmented[:, :, 0]) * 0.01
        augmented[:, :, 0] = (augmented[:, :, 0] + time_jitter).clamp(0, 1)

    # Spatial jitter
    if torch.rand(1).item() < 0.3:
        spatial_jitter_x = torch.randn_like(augmented[:, :, 1]) * 1.5
        spatial_jitter_y = torch.randn_like(augmented[:, :, 2]) * 1.5
        augmented[:, :, 1] = (augmented[:, :, 1] + spatial_jitter_x).clamp(0, width - 1)
        augmented[:, :, 2] = (augmented[:, :, 2] + spatial_jitter_y).clamp(0, height - 1)

    # Event dropout (random or clustered) simulating dead pixels / low activity
    if torch.rand(1).item() < 0.4:
        dropout_rate = torch.rand(1).item() * 0.1 + 0.05
        if torch.rand(1).item() < 0.5:
            keep_mask = torch.rand(B, N, device=device) > dropout_rate
        else:
            keep_mask = torch.ones(B, N, dtype=torch.bool, device=device)
            for b in range(B):
                n_clusters = torch.randint(1, 5, (1,)).item()
                for _ in range(n_clusters):
                    center_x = torch.rand(1).item() * width
                    center_y = torch.rand(1).item() * height
                    radius = (torch.rand(1).item() * 0.06 + 0.02) * min(width, height)
                    dist_x = (augmented[b, :, 1] - center_x).abs()
                    dist_y = (augmented[b, :, 2] - center_y).abs()
                    cluster_mask = (dist_x < radius) & (dist_y < radius)
                    keep_mask[b] &= ~cluster_mask
        new_augmented: List[torch.Tensor] = []
        for b in range(B):
            new_augmented.append(augmented[b][keep_mask[b]])
        min_events = min(batch.shape[0] for batch in new_augmented)
        final_augmented = torch.zeros(B, min(N, min_events), 4, device=device)
        for b in range(B):
            n_events = min(min_events, new_augmented[b].shape[0])
            final_augmented[b, :n_events] = new_augmented[b][:n_events]
        if final_augmented.shape[1] < N:
            padding = torch.zeros(B, N - final_augmented.shape[1], 4, device=device)
            augmented = torch.cat([final_augmented, padding], dim=1)
        else:
            augmented = final_augmented

    # Polarity flips (sensor noise)
    if torch.rand(1).item() < 0.2:
        flip_rate = torch.rand(1).item() * 0.04 + 0.01
        flip_mask = torch.rand(B, N, device=device) < flip_rate
        augmented[:, :, 3] = torch.where(flip_mask, -augmented[:, :, 3], augmented[:, :, 3])

    # Event rate variation (simulate lighting differences)
    if torch.rand(1).item() < 0.25:
        rate_factor = torch.rand(1).item() * 1.0 + 0.5
        if rate_factor < 1.0:
            keep_ratio = rate_factor
            keep_mask = torch.rand(B, N, device=device) < keep_ratio
            new_augmented: List[torch.Tensor] = []
            for b in range(B):
                new_augmented.append(augmented[b][keep_mask[b]])
            min_events = min(batch.shape[0] for batch in new_augmented)
            final_augmented = torch.zeros(B, min(N, min_events), 4, device=device)
            for b in range(B):
                n_events = min(min_events, new_augmented[b].shape[0])
                final_augmented[b, :n_events] = new_augmented[b][:n_events]
            if final_augmented.shape[1] < N:
                padding = torch.zeros(B, N - final_augmented.shape[1], 4, device=device)
                augmented = torch.cat([final_augmented, padding], dim=1)
            else:
                augmented = final_augmented
    return augmented


def add_hot_pixels(events: torch.Tensor, device: torch.device, width: int, height: int) -> torch.Tensor:
    """Inject synthetic hot pixels.

    Hot pixels are pixels that constantly fire events; this augmentation
    improves robustness to sensor defects.
    """
    B, N, _ = events.shape
    n_hot_pixels = torch.randint(5, 1000, (1,)).item()
    hot_events = torch.zeros(B, n_hot_pixels, 4, device=device)
    hot_events[:, :, 0] = torch.rand(B, n_hot_pixels, device=device)
    hot_events[:, :, 1] = torch.rand(B, n_hot_pixels, device=device) * (width - 1)
    hot_events[:, :, 2] = torch.rand(B, n_hot_pixels, device=device) * (height - 1)
    hot_events[:, :, 3] = torch.randint(0, 2, (B, n_hot_pixels), device=device) * 2 - 1
    return torch.cat([events, hot_events], dim=1)


def eventstovoxel(
    events: torch.Tensor,
    height: int = 260,
    width: int = 346,
    bins: int = 5,
    training: bool = True,
    hotpixel: bool = False,
    aug_prob: float = 0.5,
) -> torch.Tensor:
    """Convert variable-length event streams into a dense voxel grid.

    Channels are split by polarity across time bins: first ``bins`` for positive
    polarity, next ``bins`` for negative.
    """
    B, N, _ = events.shape
    device = events.device
    if hotpixel:
        events = add_hot_pixels(events, device, width, height)
    if training:
        events = apply_event_augmentations(events, training=True, aug_prob=aug_prob, width=width, height=height)
        B, N, _ = events.shape

    x = events[:, :, 1].long().clamp(0, width - 1)
    y = events[:, :, 2].long().clamp(0, height - 1)
    t = (events[:, :, 0] * bins).long().clamp(0, bins - 1)
    p = events[:, :, 3].long()
    neg_p = (p < 0).long()
    pos_p = (p > 0).long()

    voxel = torch.zeros(B, 2 * bins, height, width, device=device)
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, N)
    voxel.index_put_((batch_idx, t, y, x), pos_p * torch.ones_like(t, dtype=torch.float), accumulate=True)
    voxel.index_put_((batch_idx, t + bins, y, x), neg_p * torch.ones_like(t, dtype=torch.float), accumulate=True)
    return voxel


def eventstohistogram(events: torch.Tensor, height: int = 260, width: int = 346) -> torch.Tensor:
    """Accumulate events into a 2-channel polarity histogram."""
    B, N, _ = events.shape
    x = (events[:, :, 1] * width).long().clamp(0, width - 1)
    y = (events[:, :, 2] * height).long().clamp(0, height - 1)
    p = events[:, :, 3].long().clamp(0, 1)
    hist = torch.zeros(B, 2, height, width, device=events.device)
    batch_idx = torch.arange(B, device=events.device).unsqueeze(1).expand(-1, N)
    hist.index_put_((batch_idx, p, y, x), torch.abs(events[:, :, 3]), accumulate=True)
    return hist


def add_frame_to_video(video_writer, images: List[torch.Tensor]):
    """Composite a frame (events + depth + prediction) and append to a video.

    The first element of ``images`` is either a raw event tensor (shape[-1] == 4)
    or a voxel grid. Subsequent tensors are assumed to be single-channel images.
    """
    if images[0].shape[-1] == 4:  # raw events
        y = torch.round(images[0][0, :, 1])
        x = torch.round(images[0][0, :, 2])
        img = torch.zeros(260, 346).to(images[0].device)
        img[x.long(), y.long()] = 1
    else:
        img = 1 * (torch.sum(images[0][0], dim=0) > 0)
    images[0] = img
    merged = torch.cat([im for im in images], dim=1).detach().cpu().numpy()
    merged = (merged * 255).astype("uint8")
    merged = cv2.cvtColor(merged, cv2.COLOR_GRAY2BGR)
    video_writer.write(merged)


def calc_topk_accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> List[torch.Tensor]:
    """Compute top-k classification accuracy.

    Returns a list of accuracies corresponding to each ``k`` in ``topk``.
    """
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))
    return res
