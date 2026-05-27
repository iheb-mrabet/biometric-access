import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "spoofgate_cdcn_dataset"
MODEL_PATH = PROJECT_ROOT / "models" / "spoofgate_cdcn" / "spoofgate_cdcn.pt"
IMAGE_SIZE = 224


def ensure_src_imports():
    src = str(PROJECT_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def list_images(folder):
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files = []
    for pattern in patterns:
        files.extend(folder.glob(pattern))
    return sorted(files)


def split_paths(paths, val_ratio, seed):
    rng = random.Random(seed)
    paths = list(paths)
    rng.shuffle(paths)
    val_count = max(1, int(len(paths) * val_ratio))
    return paths[val_count:], paths[:val_count]


def augment_image(image):
    if random.random() < 0.65:
        alpha = random.uniform(0.72, 1.35)
        beta = random.uniform(-22, 22)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    if random.random() < 0.35:
        k = random.choice([3, 5])
        image = cv2.GaussianBlur(image, (k, k), 0)

    if random.random() < 0.50:
        image = cv2.flip(image, 1)

    if random.random() < 0.30:
        noise = np.random.normal(0, random.uniform(2.0, 7.0), image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return image


class SpoofDataset(Dataset):
    def __init__(self, items, train):
        self.items = items
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {path}")

        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        if self.train:
            image = augment_image(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(image), torch.tensor(label, dtype=torch.float32)


def make_loaders(batch_size, val_ratio, seed):
    real = [(path, 1.0) for path in list_images(DATASET_DIR / "real")]
    spoof = [(path, 0.0) for path in list_images(DATASET_DIR / "spoof")]

    if len(real) < 50 or len(spoof) < 50:
        raise RuntimeError(
            f"Need at least 50 real and 50 spoof images. Found real={len(real)}, spoof={len(spoof)}."
        )

    real_train, real_val = split_paths(real, val_ratio, seed)
    spoof_train, spoof_val = split_paths(spoof, val_ratio, seed + 1)

    train_items = real_train + spoof_train
    val_items = real_val + spoof_val

    random.Random(seed + 2).shuffle(train_items)
    random.Random(seed + 3).shuffle(val_items)

    train_loader = DataLoader(
        SpoofDataset(train_items, train=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        SpoofDataset(val_items, train=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader, len(real), len(spoof)


def find_best_threshold(scores, labels):
    best_threshold = 0.75
    best_acc = -1.0
    for threshold in np.linspace(0.30, 0.95, 66):
        preds = (scores >= threshold).astype(np.float32)
        acc = float(np.mean(preds == labels))
        if acc > best_acc:
            best_acc = acc
            best_threshold = float(threshold)
    return best_threshold, best_acc


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    scores = []
    labels = []
    total_loss = 0.0
    loss_fn = nn.BCEWithLogitsLoss()

    for images, batch_labels in loader:
        images = images.to(device)
        batch_labels = batch_labels.to(device)
        logits, map_logits = model(images)
        cls_loss = loss_fn(logits.view(-1), batch_labels)
        total_loss += float(cls_loss.item()) * images.size(0)
        probs = torch.sigmoid(logits.view(-1)).detach().cpu().numpy()
        scores.extend(probs.tolist())
        labels.extend(batch_labels.detach().cpu().numpy().tolist())

    scores = np.array(scores, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    threshold, acc = find_best_threshold(scores, labels)
    live_scores = scores[labels == 1]
    spoof_scores = scores[labels == 0]
    return {
        "loss": total_loss / max(1, len(labels)),
        "threshold": threshold,
        "accuracy": acc,
        "live_median": float(np.median(live_scores)) if live_scores.size else 0.0,
        "spoof_median": float(np.median(spoof_scores)) if spoof_scores.size else 0.0,
        "scores": scores,
        "labels": labels,
    }


def train(args):
    ensure_src_imports()
    from spoofgate_cdcn_model import SpoofGateCDCN, make_depth_targets, save_checkpoint

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_loader, val_loader, real_count, spoof_count = make_loaders(
        args.batch_size,
        args.val_ratio,
        args.seed,
    )

    model = SpoofGateCDCN(theta=args.theta).to(device)
    cls_loss_fn = nn.BCEWithLogitsLoss()
    map_loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Training CDCN SpoofGate on {device}. real={real_count}, spoof={spoof_count}")
    print(f"Checkpoint: {MODEL_PATH}")

    best = {"accuracy": -1.0, "threshold": 0.75}
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, map_logits = model(images)
            map_targets = make_depth_targets(labels, map_logits.shape, device)

            cls_loss = cls_loss_fn(logits.view(-1), labels)
            map_loss = map_loss_fn(map_logits, map_targets)
            loss = cls_loss + args.map_weight * map_loss
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * images.size(0)
            seen += images.size(0)

        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        train_loss = running_loss / max(1, seen)

        print(
            f"epoch={epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={metrics['loss']:.4f} "
            f"val_acc={metrics['accuracy']:.3f} threshold={metrics['threshold']:.2f} "
            f"live_med={metrics['live_median']:.3f} spoof_med={metrics['spoof_median']:.3f}"
        )

        if metrics["accuracy"] > best["accuracy"]:
            best = metrics
            save_checkpoint(
                MODEL_PATH,
                model,
                metrics["threshold"],
                IMAGE_SIZE,
                {
                    "real_count": real_count,
                    "spoof_count": spoof_count,
                    "epoch": epoch,
                    "val_accuracy": metrics["accuracy"],
                    "live_median": metrics["live_median"],
                    "spoof_median": metrics["spoof_median"],
                    "theta": args.theta,
                },
            )

    elapsed = time.time() - start
    print("=" * 80)
    print(
        f"BEST | accuracy={best['accuracy']:.3f} threshold={best['threshold']:.2f} "
        f"live_median={best['live_median']:.3f} spoof_median={best['spoof_median']:.3f} "
        f"elapsed={elapsed:.1f}s"
    )
    print(f"Saved model to {MODEL_PATH}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--map-weight", type=float, default=0.35)
    parser.add_argument("--theta", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
