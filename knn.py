"""Brute-force Manhattan-distance k-nearest neighbors implemented with NumPy."""

from __future__ import annotations

import numpy as np


class KNearestNeighbors:
    def __init__(self, k: int = 5, weights: str = "uniform"):
        self.k = k
        self.weights = weights

    def fit(self, samples: np.ndarray, labels: np.ndarray) -> KNearestNeighbors:
        samples = np.asarray(samples, dtype=np.float64)
        labels = np.asarray(labels)
        if samples.ndim != 2 or min(samples.shape) == 0 or not np.isfinite(samples).all():
            raise ValueError("Training samples must be a nonempty, finite two-dimensional matrix")
        if labels.ndim != 1 or len(labels) != len(samples):
            raise ValueError("Provide one label per training sample")
        self.samples_ = samples
        self.classes_, self.class_indices_ = np.unique(labels, return_inverse=True)
        return self

    def nearest(self, samples: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "samples_"):
            raise ValueError("Fit the classifier before predicting")
        if not 1 <= k <= len(self.samples_):
            raise ValueError("k must be between 1 and the training sample count")
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[1] != self.samples_.shape[1] or not np.isfinite(samples).all():
            raise ValueError("Query samples must be finite and match the training feature count")

        neighbors = np.empty((len(samples), k), dtype=np.intp)
        distances = np.empty((len(samples), k), dtype=np.float64)
        for i, sample in enumerate(samples):
            # Convert before subtraction so unsigned image pixels cannot wrap around.
            manhattan = np.abs(self.samples_ - sample).sum(axis=1)
            nearest = np.argsort(manhattan, kind="stable")[:k]
            neighbors[i] = self.class_indices_[nearest]
            distances[i] = manhattan[nearest]
        return neighbors, distances

    def vote(self, neighbors: np.ndarray, distances: np.ndarray, weights: str) -> np.ndarray:
        if weights not in {"uniform", "distance"}:
            raise ValueError("weights must be 'uniform' or 'distance'")
        predictions = np.empty(len(neighbors), dtype=np.intp)
        for i, (classes, row_distances) in enumerate(zip(neighbors, distances, strict=True)):
            votes = None
            if weights == "distance":
                exact = row_distances == 0
                votes = exact.astype(float) if exact.any() else 1.0 / row_distances
            counts = np.bincount(classes, weights=votes, minlength=len(self.classes_))
            predictions[i] = counts.argmax()
        return self.classes_[predictions]

    def predict(self, samples: np.ndarray) -> np.ndarray:
        neighbors, distances = self.nearest(samples, self.k)
        return self.vote(neighbors, distances, self.weights)
