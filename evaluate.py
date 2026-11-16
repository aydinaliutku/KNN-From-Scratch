"""Evaluate from-scratch k-NN on the digits demo or supplied chest X-ray folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from knn import KNearestNeighbors

SEED = 42
XRAY_CLASSES = ("covid", "normal", "pneumonia")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_xrays(folder: Path, size: int) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    images, labels = [], []
    for category in XRAY_CLASSES:
        class_dir = folder / category
        if not class_dir.is_dir():
            raise ValueError(f"Expected an image directory at {class_dir}")
        files = sorted(p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not files:
            raise ValueError(f"No supported images in {class_dir}")
        for path in files:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not read {path}")
            images.append(cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA))
            labels.append(category)
    return np.stack(images), np.array(labels)


def xray_features(images: np.ndarray, filtering: str) -> np.ndarray:
    import cv2

    if filtering == "box":
        images = np.stack([cv2.blur(img, (5, 5)) for img in images])
    if filtering == "gaussian-erode":
        kernel = np.ones((3, 3), dtype=np.uint8)
        images = np.stack([cv2.erode(cv2.GaussianBlur(img, (5, 5), 0), kernel) for img in images])
    return images.reshape(len(images), -1)


def evaluate(data_dir: Path | None, output_dir: Path) -> dict:
    if data_dir is None:
        digits = load_digits()
        train_x, test_x, train_y, test_y = train_test_split(
            digits.data, digits.target, test_size=0.2, random_state=SEED, stratify=digits.target,
        )
        variants = [("raw 8x8", train_x, test_x)]
        classes = np.arange(10)
    else:
        train_dirs = {(data_dir / "train" / label).resolve() for label in XRAY_CLASSES}
        test_dirs = {(data_dir / "test" / label).resolve() for label in XRAY_CLASSES}
        if train_dirs & test_dirs:
            raise ValueError("Supply separate train and test class directories")
        variants = []
        for size in (32, 64):
            train_images, train_y = load_xrays(data_dir / "train", size)
            test_images, test_y = load_xrays(data_dir / "test", size)
            for filtering in ("raw", "box", "gaussian-erode"):
                variants.append((
                    f"{filtering} {size}x{size}",
                    xray_features(train_images, filtering),
                    xray_features(test_images, filtering),
                ))
        classes = np.array(XRAY_CLASSES)

    fit, validation = train_test_split(
        np.arange(len(train_y)), test_size=0.25, random_state=SEED, stratify=train_y,
    )
    if len(np.unique(train_y[validation])) != len(classes) or len(np.unique(train_y[fit])) != len(classes):
        raise ValueError("Supply enough training images per class to create a stratified validation set")
    best = None
    validation_scores = []
    for name, training, test in variants:
        classifier = KNearestNeighbors().fit(training[fit], train_y[fit])
        max_k = min(10, len(fit))
        neighbors, distances = classifier.nearest(training[validation], max_k)
        for k in range(1, max_k + 1):
            for weights in ("uniform", "distance"):
                prediction = classifier.vote(neighbors[:, :k], distances[:, :k], weights)
                score = float(f1_score(train_y[validation], prediction, average="macro", labels=classes))
                row = {"features": name, "k": k, "weights": weights, "macro_f1": score}
                validation_scores.append(row)
                if best is None or score > best[0]["macro_f1"]:
                    best = row, training, test

    selection, training, test = best
    classifier = KNearestNeighbors(selection["k"], selection["weights"]).fit(training, train_y)
    prediction = classifier.predict(test)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = ConfusionMatrixDisplay.from_predictions(
        test_y, prediction, labels=classes, cmap="Blues", colorbar=False,
    )
    matrix.ax_.set_title("k-NN on the held-out test set")
    matrix.figure_.tight_layout()
    matrix.figure_.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(matrix.figure_)
    return {
        "dataset": "scikit-learn digits" if data_dir is None else "user-supplied chest X-rays",
        "selection": selection,
        "fit_examples": len(fit),
        "validation_examples": len(validation),
        "test_examples": len(test),
        "test_accuracy": float(accuracy_score(test_y, prediction)),
        "test_macro_f1": float(f1_score(test_y, prediction, average="macro", labels=classes)),
        "test_report": classification_report(test_y, prediction, labels=classes, output_dict=True, zero_division=0),
        "validation_scores": validation_scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xray-dir", type=Path, help="Optional directory with train/ and test/ class folders")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    try:
        result = evaluate(args.xray_dir, args.output)
        (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Dataset: {result['dataset']}")
    print(f"Validation selection: {result['selection']}")
    print(f"Test accuracy: {result['test_accuracy']:.3f} | macro F1: {result['test_macro_f1']:.3f}")


if __name__ == "__main__":
    main()
