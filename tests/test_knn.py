import unittest

import numpy as np

from knn import KNearestNeighbors


class KNearestNeighborsTests(unittest.TestCase):
    def test_manhattan_distance_does_not_wrap_unsigned_pixels(self):
        train = np.array([[0], [250]], dtype=np.uint8)
        model = KNearestNeighbors(k=1).fit(train, ["near", "far"])
        np.testing.assert_array_equal(model.predict(np.array([[1]], dtype=np.uint8)), ["near"])

    def test_weighting_uses_exact_matches_only_when_they_exist(self):
        model = KNearestNeighbors(k=3, weights="distance").fit([[4], [5], [6]], ["z", "a", "a"])
        np.testing.assert_array_equal(model.predict([[4]]), ["z"])

    def test_uniform_ties_are_reproducible(self):
        model = KNearestNeighbors(k=2).fit([[0], [2]], ["z", "a"])
        np.testing.assert_array_equal(model.predict([[1]]), ["a"])
