"""Offline checks for ESMFold missing metrics and ASYN evidence protocols.

Run from a checkout with NumPy installed: python scripts/check_esmfold_evidence.py
Folding and worker inference are mocked; no GPU, API, or model assets are used.
"""
from __future__ import annotations

from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astevolve.evaluation.contracts import EvaluatorContext
from astevolve.evaluation.plugins.registry import plugin_runtime_score_config
from astevolve.metrics.structure import scalar_confidence_metrics
from astevolve.providers import esmfold, esmfold_worker
from astevolve.search.structure_evidence import _aggregate_complex_state_metrics
from astevolve.search.structure_pipeline import _PROVIDER_EVIDENCE_VERSIONS
from cases.asyn_c_terminal_hotspot_binder.asyn_binder_evaluator import (
    ASYNHotspotBinderPlugin,
    TARGET_SEQUENCE,
)


class ESMFoldEvidenceChecks(unittest.TestCase):
    def tearDown(self):
        esmfold._CACHE.clear()

    def test_in_process_outputs_keep_iptm_missing_for_every_requested_metric(self):
        fake_bagel = SimpleNamespace(
            Chain=lambda **kwargs: kwargs, Residue=lambda **kwargs: kwargs
        )
        with tempfile.TemporaryDirectory() as directory:
            fake_result = SimpleNamespace(
                local_plddt=[0.7, 0.9],
                ptm=[0.6],
                to_cif=lambda path: Path(path).write_text("data_mock\n"),
            )
            model = SimpleNamespace(fold=lambda chains: fake_result)
            with patch.dict(sys.modules, {"bagel": fake_bagel}), \
                 patch.dict(esmfold.os.environ, {"ASTEVOLVE_ESMFOLD_TMP": directory}), \
                 patch.object(esmfold, "_worker_python", return_value=None), \
                 patch.object(esmfold, "_bagel_root", return_value=Path(directory)), \
                 patch.object(esmfold, "_load_model", return_value=model):
                for metric in ("plddt", "ptm", "iptm", "unsupported_metric"):
                    with self.subTest(metric=metric):
                        result = esmfold.run_esmfold_confidence_multichain(
                            pred_name=metric, chains=[("B", "AA")], metric=metric
                        )
                        self.assertEqual(result["metrics"]["plddt"], 80.0)
                        self.assertEqual(result["metrics"]["ptm"], 0.6)
                        self.assertIsNone(result["metrics"]["iptm"])
                        self.assertNotIn("unsupported_metric", result["metrics"])
                        self.assertEqual(
                            result["unavailable_metrics"]["iptm"],
                            "not_supported_by_esmfold_v1",
                        )
                        serialized = json.loads(Path(result["summary_json"]).read_text())
                        self.assertIsNone(serialized["metrics"]["iptm"])

    def test_worker_transport_and_parent_adapter_preserve_null(self):
        result = {
            "provider": "esmfold", "metrics": {"plddt": 80.0, "ptm": 0.6, "iptm": None},
            "unavailable_metrics": {"iptm": "not_supported_by_esmfold_v1"},
        }
        request = {"request_id": "offline", "chains": [["B", "AA"]], "metric": "iptm"}
        stream = io.StringIO()
        with patch.object(esmfold_worker.sys, "stdin", io.StringIO(json.dumps(request) + "\n")), \
             patch.object(esmfold_worker.sys, "stdout", stream), \
             patch.object(esmfold_worker, "run_esmfold_confidence_multichain", return_value=result):
            self.assertEqual(esmfold_worker.main(), 0)
        response = json.loads(stream.getvalue())
        self.assertEqual(response["status"], "ok")
        self.assertIsNone(response["result"]["metrics"]["iptm"])
        with patch.dict(esmfold.os.environ, {"ASTEVOLVE_ESMFOLD_IN_WORKER": "0"}), \
             patch.object(esmfold, "_worker_python", return_value=Path(sys.executable)), \
             patch.object(esmfold, "_worker_request", return_value=response["result"]), \
             patch.object(esmfold, "_load_model", side_effect=AssertionError("must not fold")):
            returned = esmfold.run_esmfold_confidence_multichain(chains=[("B", "AA")], metric="iptm")
        self.assertIsNone(returned["metrics"]["iptm"])

    def test_normalization_and_state_aggregation_do_not_create_iptm_zero(self):
        scalar = scalar_confidence_metrics({"metrics": {"plddt": 80.0, "ptm": 0.6, "iptm": None}})
        self.assertNotIn("iptm", scalar)
        aggregate = _aggregate_complex_state_metrics([{"structure_metrics": {"scalar": scalar}}])
        self.assertNotIn("iptm", aggregate["scalar"])
        self.assertEqual(scalar_confidence_metrics({"metrics": {"iptm": 0.0}})["iptm"], 0.0)
        self.assertEqual(_PROVIDER_EVIDENCE_VERSIONS["esmfold"], "astevolve.esmfold_adapter.v2")


class ASYNEvidenceChecks(unittest.TestCase):
    def context(self, *, provider="protenix", iptm=None, protocol=None):
        state = {
            "name": "binder_plus_ASYN", "provider": provider,
            "confidence_metrics": {"iptm": iptm},
            "structure_metrics": {
                "scalar": scalar_confidence_metrics({"metrics": {"iptm": iptm}}),
                "chain_plddt": {"B": 80.0},
                "node_plddt": {"B01": {"plddt_mean": 80.0, "plddt_min": 70.0}},
            },
        }
        return EvaluatorContext(
            out={"seqs": {"T": TARGET_SEQUENCE, "B": "EAKLQ" * 10}},
            structure={"states": [state], "multistate_objectives": {"objectives": {
                "binder_ASYN_interface": {"score": 0.8, "details": {"clash_count": 0}}
            }}},
            compiled=None, design_state={}, masks=None, template_seqs=None, fixed_residues=None,
            score_config={} if protocol is None else {"asyn_interface_evidence_protocol": protocol},
        )

    def terms(self, context):
        return {term.name: term for term in ASYNHotspotBinderPlugin().score_terms(context)}

    def assert_complete(self, terms, expected):
        gate = terms["asyn_state_evidence_complete"]
        self.assertTrue(gate.details["hard_gate"])
        self.assertEqual(gate.score, float(expected))
        self.assertEqual(gate.available, expected)
        self.assertEqual(terms["asyn_interface_quality"].available, expected)

    def test_missing_iptm_fails_required_protocol(self):
        for provider in ("protenix", "alphafold3", "esmfold2", "unknown", ""):
            with self.subTest(provider=provider):
                terms = self.terms(self.context(provider=provider))
                self.assert_complete(terms, False)
                self.assertTrue(terms["asyn_state_evidence_complete"].details["bound_iptm_required"])

    def test_real_zero_and_positive_iptm_remain_valid_evidence(self):
        for iptm in (0.0, 0.65):
            with self.subTest(iptm=iptm):
                terms = self.terms(self.context(iptm=iptm, protocol="iptm_required"))
                self.assert_complete(terms, True)
                self.assertAlmostEqual(terms["asyn_interface_quality"].score, 0.56 + 0.30 * iptm / 0.65)

    def test_invalid_iptm_fails_required_protocol(self):
        for iptm in (float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(iptm=iptm):
                self.assert_complete(self.terms(self.context(iptm=iptm)), False)

    def test_explicit_esmfold_protocol_uses_available_interface_evidence(self):
        terms = self.terms(self.context(provider="esmfold", protocol="esmfold_interface_only"))
        self.assert_complete(terms, True)
        self.assertAlmostEqual(terms["asyn_interface_quality"].score, 0.8)
        gate = terms["asyn_state_evidence_complete"]
        self.assertFalse(gate.details["bound_iptm_available"])
        self.assertFalse(gate.details["bound_iptm_required"])

    def test_legacy_esmfold_zero_cannot_masquerade_as_iptm_evidence(self):
        strict = self.terms(self.context(provider="esmfold", iptm=0.0))
        self.assert_complete(strict, False)
        explicit = self.terms(self.context(provider="esmfold", iptm=0.0, protocol="esmfold_interface_only"))
        self.assert_complete(explicit, True)
        self.assertFalse(explicit["asyn_state_evidence_complete"].details["bound_iptm_available"])
        self.assertIsNone(explicit["asyn_interface_quality"].details["bound_iptm"])

    def test_esmfold_exemption_does_not_apply_to_other_or_unknown_providers(self):
        for provider in ("protenix", "esmfold2", "unknown", ""):
            with self.subTest(provider=provider):
                context = self.context(provider=provider, iptm=0.65, protocol="esmfold_interface_only")
                self.assert_complete(self.terms(context), False)

    def test_esmfold_protocol_still_requires_interface_and_binder_confidence(self):
        base = self.context(provider="esmfold", protocol="esmfold_interface_only")
        missing_interface = deepcopy(base)
        missing_interface.structure["multistate_objectives"] = {}
        self.assert_complete(self.terms(missing_interface), False)
        missing_confidence = deepcopy(base)
        missing_confidence.structure["states"][0]["structure_metrics"] = {}
        terms = self.terms(missing_confidence)
        self.assertEqual(terms["asyn_state_evidence_complete"].score, 0.0)
        self.assertFalse(terms["asyn_state_evidence_complete"].available)

    def test_unknown_protocol_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported ASYN interface evidence protocol"):
            self.terms(self.context(iptm=0.65, protocol="optional"))

    def test_case_configuration_resolves_explicit_esmfold_protocol(self):
        path = Path(__file__).resolve().parents[1] / "cases/asyn_c_terminal_hotspot_binder/runtime_config.json"
        config = json.loads(path.read_text())
        resolved = plugin_runtime_score_config(config["score_config"], "asyn_hotspot_binder")
        self.assertEqual(config["inner_structure_model"], "esmfold")
        self.assertEqual(resolved["asyn_interface_evidence_protocol"], "esmfold_interface_only")
        context = self.context(provider="esmfold")
        context.score_config = resolved
        self.assert_complete(self.terms(context), True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
