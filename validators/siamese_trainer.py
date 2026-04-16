"""
Trainer for the Siamese signature verification network.

Data pipeline
-------------
1. Load all reference signature images from DB (via file paths).
2. Preprocess each image with SignatureValidator._load_and_preprocess().
3. Augment each image N times (random rotation, elastic distortion, etc.)
   to counteract the small number of training samples.
4. Build pair dataset:
     Positive pairs (y=1): two signatures from the same employee.
     Negative pairs (y=0): signatures from different employees.
   Pairs are balanced (equal positives and negatives).
5. Train SiameseCNN with ContrastiveLoss using Adam + cosine LR schedule.
6. Save the best model (lowest validation loss) to  models/siamese.pt.

Typical usage
-------------
trainer = SiameseTrainer(db_session)
report  = trainer.train(epochs=60, augment_factor=12)
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from validators.siamese_network import (
    SiameseCNN,
    ContrastiveLoss,
    SiameseValidator,
    _best_device,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Result dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingReport:
    status: str                          # "ok" | "skipped" | "error"
    message: str = ""
    epochs_run: int = 0
    best_val_loss: float = 0.0
    train_loss_history: list[float] = field(default_factory=list)
    val_loss_history: list[float]   = field(default_factory=list)
    n_employees: int = 0
    n_pairs: int = 0
    duration_seconds: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Augmentation
# ═══════════════════════════════════════════════════════════════════════════

def _augment(img: np.ndarray) -> np.ndarray:
    """
    Apply random augmentations to a binary (0/255) grayscale signature image.
    Returns an augmented copy of the same shape.
    """
    h, w = img.shape

    # 1. Random slight rotation (±6°)
    angle  = random.uniform(-6, 6)
    M_rot  = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img    = cv2.warpAffine(img, M_rot, (w, h), borderValue=0)

    # 2. Random translation (±4% of each dimension)
    tx = random.uniform(-0.04 * w, 0.04 * w)
    ty = random.uniform(-0.04 * h, 0.04 * h)
    M_tr = np.float32([[1, 0, tx], [0, 1, ty]])
    img  = cv2.warpAffine(img, M_tr, (w, h), borderValue=0)

    # 3. Random scale (0.92 – 1.08)
    scale = random.uniform(0.92, 1.08)
    M_sc  = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
    img   = cv2.warpAffine(img, M_sc, (w, h), borderValue=0)

    # 4. Elastic distortion — mimics natural handwriting variation
    if random.random() < 0.6:
        img = _elastic_distort(img, alpha=random.uniform(4, 12), sigma=random.uniform(2, 4))

    # 5. Random Gaussian noise (very light)
    if random.random() < 0.4:
        noise = np.random.normal(0, 8, img.shape).astype(np.int16)
        img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Re-binarize after distortion
    _, img = cv2.threshold(img, 64, 255, cv2.THRESH_BINARY)
    return img


def _elastic_distort(img: np.ndarray, alpha: float, sigma: float) -> np.ndarray:
    from scipy.ndimage import map_coordinates, gaussian_filter

    h, w  = img.shape
    rand_x = np.random.rand(h, w) * 2 - 1
    rand_y = np.random.rand(h, w) * 2 - 1
    dx = gaussian_filter(rand_x, sigma) * alpha
    dy = gaussian_filter(rand_y, sigma) * alpha

    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    ix = np.clip(xs + dx, 0, w - 1)
    iy = np.clip(ys + dy, 0, h - 1)

    distorted = map_coordinates(img, [iy.ravel(), ix.ravel()], order=1).reshape(h, w)
    return distorted.astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

class PairDataset(Dataset):
    """
    Dataset of (img_a, img_b, label) signature pairs.

    Parameters
    ----------
    pairs:
        List of (img_a, img_b, label) tuples where images are uint8 numpy
        arrays and label is 0 or 1.
    """

    def __init__(self, pairs: list[tuple[np.ndarray, np.ndarray, int]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_a, img_b, label = self.pairs[idx]
        t_a = torch.from_numpy(img_a.astype(np.float32) / 255.0).unsqueeze(0)
        t_b = torch.from_numpy(img_b.astype(np.float32) / 255.0).unsqueeze(0)
        return t_a, t_b, torch.tensor(float(label))


# ═══════════════════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════════════════

class SiameseTrainer:
    """
    Builds training pairs from the database and trains SiameseCNN.

    Parameters
    ----------
    reference_data:
        Dict mapping  employee_id → list[file_path_str].
        Caller (router) is responsible for querying the DB and passing this.
    model_path:
        Where to save the best model weights.
    """

    MIN_EMPLOYEES = 2   # need at least 2 different signers for negative pairs

    def __init__(
        self,
        reference_data: dict[int, list[str]],
        model_path: Path,
    ) -> None:
        self.reference_data = reference_data   # {emp_id: [path1, path2, ...]}
        self.model_path     = model_path
        self.device         = _best_device()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def train(
        self,
        epochs: int = 60,
        augment_factor: int = 15,
        lr: float = 1e-3,
        batch_size: int = 32,
        val_split: float = 0.15,
    ) -> TrainingReport:
        t0 = time.time()

        n_employees = len(self.reference_data)
        if n_employees < self.MIN_EMPLOYEES:
            return TrainingReport(
                status="skipped",
                message=(
                    f"Cần ít nhất {self.MIN_EMPLOYEES} nhân viên có chữ ký mẫu để huấn luyện. "
                    f"Hiện tại chỉ có {n_employees}."
                ),
                n_employees=n_employees,
            )

        # 1. Load & preprocess images
        logger.info("Loading and preprocessing signature images …")
        emp_images: dict[int, list[np.ndarray]] = {}
        for emp_id, paths in self.reference_data.items():
            imgs = []
            for p in paths:
                img = self._preprocess(p)
                if img is not None:
                    imgs.append(img)
            if imgs:
                emp_images[emp_id] = imgs

        if len(emp_images) < self.MIN_EMPLOYEES:
            return TrainingReport(
                status="skipped",
                message="Không đọc được đủ ảnh chữ ký từ disk.",
                n_employees=n_employees,
            )

        # 2. Augment
        logger.info("Augmenting (×%d) …", augment_factor)
        for emp_id in emp_images:
            originals = emp_images[emp_id]
            augmented = list(originals)
            for img in originals:
                for _ in range(augment_factor - 1):
                    augmented.append(_augment(img))
            emp_images[emp_id] = augmented

        # 3. Build pairs
        pairs = self._build_pairs(emp_images)
        random.shuffle(pairs)
        logger.info("Built %d pairs from %d employees.", len(pairs), len(emp_images))

        # 4. Train/val split
        dataset  = PairDataset(pairs)
        n_val    = max(1, int(len(dataset) * val_split))
        n_train  = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

        # 5. Model + optimiser + scheduler
        model     = SiameseCNN().to(self.device)
        criterion = ContrastiveLoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

        best_val_loss   = float("inf")
        train_history: list[float] = []
        val_history:   list[float] = []

        # 6. Training loop
        logger.info("Training on %s for %d epochs …", self.device, epochs)
        for epoch in range(1, epochs + 1):
            train_loss = self._run_epoch(model, train_loader, criterion, optimiser, train=True)
            val_loss   = self._run_epoch(model, val_loader,   criterion, None,      train=False)
            scheduler.step()

            train_history.append(round(train_loss, 5))
            val_history.append(round(val_loss, 5))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), str(self.model_path))

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d/%d — train=%.4f  val=%.4f  best_val=%.4f",
                    epoch, epochs, train_loss, val_loss, best_val_loss,
                )

        # Force singleton reload with new weights
        SiameseValidator.reset()

        duration = time.time() - t0
        logger.info(
            "Training complete in %.1fs. Best val loss: %.4f. Model saved to '%s'.",
            duration, best_val_loss, self.model_path,
        )

        return TrainingReport(
            status="ok",
            message="Huấn luyện hoàn tất.",
            epochs_run=epochs,
            best_val_loss=round(best_val_loss, 5),
            train_loss_history=train_history,
            val_loss_history=val_history,
            n_employees=len(emp_images),
            n_pairs=len(pairs),
            duration_seconds=round(duration, 1),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_epoch(
        self,
        model: SiameseCNN,
        loader: DataLoader,
        criterion: ContrastiveLoss,
        optimiser,
        train: bool,
    ) -> float:
        model.train(train)
        total_loss = 0.0
        ctx = torch.enable_grad if train else torch.no_grad

        with ctx():
            for t_a, t_b, labels in loader:
                t_a    = t_a.to(self.device)
                t_b    = t_b.to(self.device)
                labels = labels.to(self.device)

                e1, e2 = model(t_a, t_b)
                loss   = criterion(e1, e2, labels)

                if train:
                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()

                total_loss += loss.item() * len(labels)

        return total_loss / max(len(loader.dataset), 1)

    @staticmethod
    def _preprocess(path: str) -> np.ndarray | None:
        """Reuse SignatureValidator preprocessing."""
        try:
            from validators.signature_validator import SignatureValidator
            return SignatureValidator()._load_and_preprocess(path)
        except Exception as exc:
            logger.warning("Cannot preprocess '%s': %s", path, exc)
            return None

    @staticmethod
    def _build_pairs(
        emp_images: dict[int, list[np.ndarray]],
    ) -> list[tuple[np.ndarray, np.ndarray, int]]:
        """
        Build balanced positive (y=1) and negative (y=0) pairs.

        Strategy
        --------
        * Positive: all combinations within each employee's images.
        * Negative: random cross-employee pairs, limited to 1× the number
          of positive pairs to keep the dataset balanced.
        """
        pairs: list[tuple[np.ndarray, np.ndarray, int]] = []
        emp_ids = list(emp_images.keys())

        # Positive pairs
        for imgs in emp_images.values():
            for i in range(len(imgs)):
                for j in range(i + 1, len(imgs)):
                    pairs.append((imgs[i], imgs[j], 1))

        n_pos = len(pairs)

        # Negative pairs (balanced)
        neg_pairs: list[tuple[np.ndarray, np.ndarray, int]] = []
        attempts = 0
        while len(neg_pairs) < n_pos and attempts < n_pos * 20:
            attempts += 1
            id_a, id_b = random.sample(emp_ids, 2)
            img_a = random.choice(emp_images[id_a])
            img_b = random.choice(emp_images[id_b])
            neg_pairs.append((img_a, img_b, 0))

        pairs.extend(neg_pairs)
        return pairs
