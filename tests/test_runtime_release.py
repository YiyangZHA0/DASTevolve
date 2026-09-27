import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from astevolve.runtime.paths import resolve_path
from scripts.check_assets import check_case_assets


class ExternalInputTests(unittest.TestCase):
    def test_data_uri_stays_inside_configured_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch.dict(os.environ, {"ASTEVOLVE_DATA_ROOT": str(root)}):
                self.assertEqual(resolve_path("data://case/reference.pdb"), root / "case/reference.pdb")
                for uri in ("data://", "data:///file", "data://../file", "data://case/../../file",
                            "data://C:/file", "data://case//file", "data://case/./file"):
                    with self.subTest(uri=uri), self.assertRaises(ValueError):
                        resolve_path(uri)
                self.assertEqual(resolve_path("state.json", base=root), root / "state.json")

    def test_missing_config_evaluator_and_external_input_are_not_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = root / "case"
            case.mkdir()
            data = root / "external"
            (data / "case").mkdir(parents=True)
            manifest = case / "case.json"
            manifest.write_text(json.dumps({
                "case_id": "case", "design_state_path": "state.json",
                "entry_program": "initial.py", "config_path": "config.yaml",
                "outer_evaluator_path": "evaluate.py",
                "required_assets": ["data://case/reference.pdb"],
            }))
            for name in ("state.json", "initial.py", "config.yaml", "evaluate.py"):
                (case / name).write_text("{}")
            reference = data / "case/reference.pdb"
            reference.write_text("END\n")
            with patch.dict(os.environ, {"ASTEVOLVE_DATA_ROOT": str(data)}, clear=True):
                self.assertTrue(check_case_assets(manifest_path=manifest)["ready"])
                for path, role in ((case / "config.yaml", "config"),
                                   (case / "evaluate.py", "outer_evaluator_path"),
                                   (reference, "required_asset:0")):
                    content = path.read_bytes()
                    path.unlink()
                    report = check_case_assets(manifest_path=manifest)
                    self.assertFalse(report["ready"])
                    self.assertIn(role, report["missing_required"])
                    path.mkdir()
                    self.assertFalse(check_case_assets(manifest_path=manifest)["ready"])
                    path.rmdir()
                    path.write_bytes(content)

    def test_core_imports_do_not_require_gpu_or_posix_locking(self):
        for name in ("astevolve.evolution", "astevolve.adapters.evolution", "engine.case_builder"):
            with self.subTest(module=name):
                importlib.import_module(name)


if __name__ == "__main__":
    unittest.main()
