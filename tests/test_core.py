"""Regression tests for public, no-network parts of the generation pipeline."""

import os
import tempfile
import unittest

from orchestrator import Orchestrator
from seed_manager import SeedManager


class RendererValidationTests(unittest.TestCase):
    def test_allows_a_plot_saved_to_the_supplied_path(self) -> None:
        code = (
            "import matplotlib.pyplot as plt\n"
            "plt.plot([0, 1], [0, 1])\n"
            "plt.savefig(save_path)\n"
            "plt.close()\n"
        )
        Orchestrator._validate_generated_code(code)

    def test_rejects_non_plotting_imports(self) -> None:
        with self.assertRaises(ValueError):
            Orchestrator._validate_generated_code("import os\n")

    def test_rejects_writes_outside_save_path(self) -> None:
        code = "import matplotlib.pyplot as plt\nplt.savefig('other.png')\n"
        with self.assertRaises(ValueError):
            Orchestrator._validate_generated_code(code)

    def test_runs_a_valid_renderer_in_an_isolated_subprocess(self) -> None:
        code = (
            "import matplotlib.pyplot as plt\n"
            "plt.plot([0, 1], [0, 1])\n"
            "plt.savefig(save_path)\n"
            "plt.close()\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, "rendered.png")
            Orchestrator().execute_python_code(code, image_path)
            self.assertTrue(os.path.isfile(image_path))
            self.assertGreater(os.path.getsize(image_path), 0)


class SeedQuotaTests(unittest.TestCase):
    def test_reservations_do_not_exceed_assigned_quota(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state_path = os.path.join(directory, "state.json")
            manager = SeedManager(state_path, target_count_per_seed=1, target_counts={"a": 2})
            seed = {"name": "a"}
            selected = [manager.get_next_seed_to_generate([seed]) for _ in range(2)]
            self.assertTrue(all(selected))
            self.assertIsNone(manager.get_next_seed_to_generate([seed]))
            for item in selected:
                manager.record_success(item)
            self.assertEqual(manager.get_generation_progress([seed])["total_generated"], 2)


if __name__ == "__main__":
    unittest.main()
