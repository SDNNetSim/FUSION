
import unittest
from helper_scripts.snr_helpers import get_overlapping_lightpaths, edge_set

class TestSnrHelpers(unittest.TestCase):

    def test_edge_set_bidirectional(self):
        path = [1, 2, 3]
        expected = {frozenset({1, 2}), frozenset({2, 3})}
        result = edge_set(path, bidirectional=True)
        self.assertEqual(result, expected)

    def test_edge_set_unidirectional(self):
        path = [1, 2, 3]
        expected = {(1, 2), (2, 3)}
        result = edge_set(path, bidirectional=False)
        self.assertEqual(result, expected)

    def test_core_and_slot_overlap(self):
        new_lp = {"path": [1, 2, 3], "spectrum": (10, 20), "core": 1, "band": "C"}
        existing = [
            {"id": "a", "path": [2, 3, 4], "spectrum": (15, 25), "core": 1, "band": "C"},
            {"id": "b", "path": [2, 3, 4], "spectrum": (21, 30), "core": 1, "band": "C"},
            {"id": "c", "path": [5, 6],     "spectrum": (15, 25), "core": 1, "band": "C"},
        ]
        result = get_overlapping_lightpaths(new_lp, existing, cores_per_link=7)
        ids = {lp["id"] for lp in result}
        self.assertIn("a", ids)
        self.assertNotIn("b", ids)
        self.assertNotIn("c", ids)

    def test_adjacent_core_toggle(self):
        new_lp = {"path": [1, 2], "spectrum": (10, 20), "core": 1, "band": "C"}
        lps = [{"id": "d", "path": [1, 2], "spectrum": (12, 18), "core": 2, "band": "C"}]
        result_with = get_overlapping_lightpaths(new_lp, lps, cores_per_link=7, include_adjacent_cores=True)
        result_without = get_overlapping_lightpaths(new_lp, lps, cores_per_link=7, include_adjacent_cores=False)
        self.assertEqual(len(result_with), 1)
        self.assertEqual(len(result_without), 0)

    def test_cross_band_toggle(self):
        new_lp = {"path": [1, 2], "spectrum": (10, 20), "core": 1, "band": "C"}
        lps = [{"id": "e", "path": [1, 2], "spectrum": (12, 18), "core": 1, "band": "L"}]
        result_with = get_overlapping_lightpaths(new_lp, lps, cores_per_link=7, include_all_bands=True)
        result_without = get_overlapping_lightpaths(new_lp, lps, cores_per_link=7, include_all_bands=False)
        self.assertEqual(len(result_with), 1)
        self.assertEqual(len(result_without), 0)

    def test_bidirectional_links(self):
        new_lp = {"path": [2, 1], "spectrum": (10, 20), "core": 1, "band": "C"}
        lps = [{"id": "f", "path": [1, 2], "spectrum": (12, 18), "core": 1, "band": "C"}]
        result_bi = get_overlapping_lightpaths(new_lp, lps, cores_per_link=7, bidirectional_links=True)
        result_uni = get_overlapping_lightpaths(new_lp, lps, cores_per_link=7, bidirectional_links=False)
        self.assertEqual(len(result_bi), 1)
        self.assertEqual(len(result_uni), 0)

    def test_multiple_overlaps(self):
        new_lp = {"path": [1, 2], "spectrum": (10, 20), "core": 1, "band": "C"}
        lps = [
            {"id": "a", "path": [1, 2], "spectrum": (12, 18), "core": 1, "band": "C"},
            {"id": "b", "path": [1, 2], "spectrum": (11, 19), "core": 1, "band": "C"},
            {"id": "c", "path": [1, 2], "spectrum": (13, 17), "core": 1, "band": "C"},
        ]
        result = get_overlapping_lightpaths(new_lp, lps, cores_per_link=7)
        self.assertEqual({lp["id"] for lp in result}, {"a", "b", "c"})

    def test_slot_boundary_case(self):
        new_lp = {"path": [1, 2], "spectrum": (10, 20), "core": 1, "band": "C"}
        lps = [
            {"id": "before", "path": [1, 2], "spectrum": (0, 9), "core": 1, "band": "C"},
            {"id": "touching", "path": [1, 2], "spectrum": (20, 25), "core": 1, "band": "C"},
            {"id": "overlap", "path": [1, 2], "spectrum": (19, 22), "core": 1, "band": "C"},
        ]
        result = get_overlapping_lightpaths(new_lp, lps, cores_per_link=7)
        ids = {lp["id"] for lp in result}
        self.assertNotIn("before", ids)
        self.assertIn("touching", ids)
        self.assertIn("overlap", ids)

if __name__ == '__main__':
    unittest.main()
