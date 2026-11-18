

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional, Callable
import numpy as np
from astevolve.core.amino_acids import (
    AA,
    AA_SET,
    BETA_FAV,
    CHARGED,
    HELIX_FAV,
    HYDROPHOBIC,
)
from astevolve.runtime.conda import resolve_protenix_conda_env
from astevolve.providers.registry import (
    run_structure_confidence_multichain,
    run_structure_plddt_multichain,
)

class EnergyTerm:


    kind: str = "base"

    def penalty(self, seqs: Dict[str, str], compiled: Dict[str, Any]) -> float:
        raise NotImplementedError


def _segment_filter(seg, f: Optional[Dict[str, Any]]) -> bool:
    if not f:
        return True
    if "chain_id" in f and seg.chain_id != f["chain_id"]:
        return False
    if "kind" in f and seg.kind != f["kind"]:
        return False
    if "name" in f and seg.name != f["name"]:
        return False
    return True


def _extract_frag(seg, seqs: Dict[str, str]) -> str:

    return seg.extract(seqs.get(seg.chain_id, ""))


def _default_pred_name(chain_ids: List[str]) -> str:
    return "__".join(chain_ids) if chain_ids else "pred"


def _ensure_struct_cache(compiled: Dict[str, Any]) -> Dict[str, Any]:
    if "_struct_cache" not in compiled or compiled["_struct_cache"] is None:
        compiled["_struct_cache"] = {}
    return compiled["_struct_cache"]


def _seq_signature(seqs: Dict[str, str], chain_ids: List[str]) -> Tuple[Tuple[str, str], ...]:
    return tuple((cid, seqs.get(cid, "")) for cid in chain_ids)


def _get_residue_plddt_from_compiled(
    compiled: Dict[str, Any],
    chain_id: str,
) -> Optional[List[float]]:
    struct_cache = compiled.get("_struct_cache", {}) or {}
    key = ("residue_plddt", chain_id)
    if key in struct_cache:
        vals = struct_cache[key]
        return [float(x) for x in vals] if vals is not None else None

    legacy = compiled.get("_residue_plddt", {})
    if isinstance(legacy, dict) and chain_id in legacy:
        vals = legacy[chain_id]
        return [float(x) for x in vals] if vals is not None else None

    return None


def _maybe_fetch_residue_confidence(
    seqs: Dict[str, str],
    chain_ids: List[str],
    compiled: Dict[str, Any],
    device: Optional[str],
    model_name: str,
    conda_env: str,
    seed: int,
) -> None:

    if not chain_ids:
        return

    struct_cache = _ensure_struct_cache(compiled)
    signature = _seq_signature(seqs, chain_ids)
    signature_key = ("residue_plddt_signature", tuple(chain_ids))
    if struct_cache.get(signature_key) == signature and all(
        ("residue_plddt", cid) in struct_cache for cid in chain_ids
    ):
        return

    try:
        chains = []
        for cid in chain_ids:
            seq = seqs.get(cid, "")
            if not seq:
                return
            chains.append((cid, seq))

        out = run_structure_confidence_multichain(
            pred_name=_default_pred_name(chain_ids),
            chains=chains,
            device=device,
            seed=seed,
            model_name=model_name,
            conda_env=conda_env,
        )
        if not isinstance(out, dict):
            return

        metrics = out.get("metrics", {}) or {}
        residue_plddt = out.get("residue_plddt", {}) or {}

        for metric_name, value in metrics.items():
            try:
                struct_cache[("scalar", str(metric_name), tuple(chain_ids))] = float(value)
            except (TypeError, ValueError):
                continue

        for cid, vals in residue_plddt.items():
            struct_cache[("residue_plddt", str(cid))] = [float(x) for x in vals]
        struct_cache[signature_key] = signature
    except Exception:
        return


@dataclass
class AlphabetTerm(EnergyTerm):


    kind: str = "alphabet"
    allowed: set = None
    chain_ids: Optional[List[str]] = None

    def __post_init__(self):
        if self.allowed is None:
            self.allowed = set(AA_SET)

    def penalty(self, seqs: Dict[str, str], compiled: Dict[str, Any]) -> float:
        bad = 0
        selected = set(self.chain_ids or [])
        items = seqs.items() if not selected else [(cid, seqs.get(cid, "")) for cid in selected]
        for _, s in items:
            bad += sum(1 for c in s if c not in self.allowed)
        return float(bad)


@dataclass
class FixedChainSequenceTerm(EnergyTerm):


    kind: str = "fixed_chain_sequence"
    chain_id: str = "T"
    sequence: str = ""

    def penalty(self, seqs: Dict[str, str], compiled: Dict[str, Any]) -> float:
        cur = seqs.get(self.chain_id, "")
        if len(cur) != len(self.sequence):
            return 10.0
        return float(sum(1 for a, b in zip(cur, self.sequence) if a != b))


@dataclass
class FixedResiduesTerm(EnergyTerm):


    kind: str = "fixed_residues"
    fixed_residues: Dict[str, Dict[int, str]] = None

    def penalty(self, seqs: Dict[str, str], compiled: Dict[str, Any]) -> float:
        if not self.fixed_residues:
            return 0.0
        pen = 0.0
        for chain_id, req in self.fixed_residues.items():
            s = seqs.get(chain_id, "")
            for pos, aa in req.items():
                if pos < 0 or pos >= len(s):
                    pen += 2.0
                elif s[pos] != aa:
                    pen += 1.0
        return float(pen)


@dataclass
class SecondaryStructureProxyTerm(EnergyTerm):


    kind: str = "ss_proxy"
    target_map: Dict[Tuple[str, str], str] = None

    def penalty(self, seqs: Dict[str, str], compiled: Dict[str, Any]) -> float:
        if not self.target_map:
            return 0.0
        total = 0.0
        for seg in compiled["segments"]:
            key = (seg.chain_id, seg.name)
            if key not in self.target_map:
                continue
            target = self.target_map[key]
            frag = _extract_frag(seg, seqs)
            if not frag:
                total += 2.0
                continue
            if target == "H":
                frac = sum(1 for c in frag if c in HELIX_FAV) / len(frag)
                total += (1.0 - frac)
            elif target == "E":
                frac = sum(1 for c in frag if c in BETA_FAV) / len(frag)
                total += (1.0 - frac)
            else:
                frac = sum(1 for c in frag if (c in HELIX_FAV or c in BETA_FAV)) / len(frag)
                total += frac
        return float(total)


@dataclass
class HydrophobicPatternTerm(EnergyTerm):


    kind: str = "hydrophobic_pattern"
    domain_min_hydro: float = 0.35
    linker_max_hydro: float = 0.25

    def penalty(self, seqs: Dict[str, str], compiled: Dict[str, Any]) -> float:
        pen = 0.0
        for seg in compiled["segments"]:
            frag = _extract_frag(seg, seqs)
            if not frag:
                pen += 2.0
                continue
            frac_h = sum(1 for c in frag if c in HYDROPHOBIC) / len(frag)
            if seg.kind == "domain":
                pen += max(0.0, self.domain_min_hydro - frac_h)
            if seg.kind == "linker":
                pen += max(0.0, frac_h - self.linker_max_hydro)
        return float(pen)


@dataclass
class InterfaceProxyTerm(EnergyTerm):


    kind: str = "interface_proxy"
    binder_chain: str = "B"
    binder_segment: str = "iface"
    target_chain: str = "T"
    target_segment: str = "epitope"
    desired_binder_hydro: float = 0.45

    def penalty(self, seqs: Dict[str, str], compiled: Dict[str, Any]) -> float:
        bseg = None
        tseg = None
        for seg in compiled["segments"]:
            if seg.chain_id == self.binder_chain and seg.name == self.binder_segment:
                bseg = seg
            if seg.chain_id == self.target_chain and seg.name == self.target_segment:
                tseg = seg
        if bseg is None or tseg is None:
            return 2.0

        bfrag = _extract_frag(bseg, seqs)
        if not bfrag:
            return 2.0
        frac_h = sum(1 for c in bfrag if c in HYDROPHOBIC) / len(bfrag)
        return float(abs(frac_h - self.desired_binder_hydro))


@dataclass
class SegmentCompositionTerm(EnergyTerm):


    kind: str = "segment_composition"
    aa_set: set = None
    min_frac: float = 0.0
    max_frac: float = 1.0
    segment_filter: Optional[Dict[str, Any]] = None

    def penalty(self, seqs: Dict[str, str], compiled: Dict[str, Any]) -> float:
        if not self.aa_set:
            return 0.0
        pen = 0.0
        for seg in compiled["segments"]:
            if not _segment_filter(seg, self.segment_filter):
                continue
            frag = _extract_frag(seg, seqs)
            if not frag:
                pen += 2.0
                continue
            frac = sum(1 for c in frag if c in self.aa_set) / len(frag)
            if frac < self.min_frac:
                pen += (self.min_frac - frac)
            if frac > self.max_frac:
                pen += (frac - self.max_frac)
        return float(pen)


@dataclass
class MaxRunTerm(EnergyTerm):

    kind: str = "max_run"
    aa_set: set = None
    max_run: int = 3
    segment_filter: Optional[Dict[str, Any]] = None

    def penalty(self, seqs: Dict[str, str], compiled: Dict[str, Any]) -> float:
        if not self.aa_set or self.max_run <= 0:
            return 0.0
        pen = 0.0
        for seg in compiled["segments"]:
            if not _segment_filter(seg, self.segment_filter):
                continue
            for span_s, span_e in seg.spans:
                chain_seq = seqs.get(seg.chain_id, "")
                frag = chain_seq[span_s:span_e]
                if not frag:
                    pen += 2.0
                    continue
                run = 0
                max_found = 0
                for c in frag:
                    if c in self.aa_set:
                        run += 1
                        if run > max_found:
                            max_found = run
                    else:
                        run = 0
                if max_found > self.max_run:
                    pen += (max_found - self.max_run)
        return float(pen)


@dataclass
class ProtenixPlddtTerm(EnergyTerm):

    kind: str = "chai_plddt"
    chains: Optional[List[str]] = None
    mode: str = "high"
    target: float = 70.0
    scale: float = 100.0
    device: Optional[str] = None
    model_name: str = "protenix_mini_esm_v0.5.0"
    conda_env: str = field(default_factory=resolve_protenix_conda_env)
    seed: int = 101

    _cache: Dict[str, float] = field(
        default_factory=dict, init=False, repr=False
    )

    def penalty(self, seqs: Dict[str, str], compiled: Dict[str, Any]) -> float:
        if "_plddt" in compiled and compiled["_plddt"] is not None:
            plddt = float(compiled["_plddt"])
        else:
            chain_ids = self.chains or compiled.get("chain_order", [])
            if not chain_ids:
                return 2.0

            chains = []
            for cid in chain_ids:
                s = seqs.get(cid, "")
                if not s:
                    return 2.0
                chains.append((cid, s))

            key = (
                "|".join(f"{cid}:{seq}" for cid, seq in chains)
                + f"|{self.model_name}|{self.seed}|{self.mode}"
            )
            if key in self._cache:
                plddt = self._cache[key]
            else:
                plddt = run_structure_plddt_multichain(
                    pred_name=_default_pred_name(chain_ids),
                    chains=chains,
                    device=self.device,
                    seed=self.seed,
                    model_name=self.model_name,
                    conda_env=self.conda_env,
                )
                self._cache[key] = float(plddt)

        score = float(plddt) / float(self.scale)
        target_score = float(self.target) / float(self.scale)

        if self.mode == "low":
            return float(max(0.0, score - target_score))
        return float(max(0.0, target_score - score))


ChaiPlddtTerm = ProtenixPlddtTerm


@dataclass
class ProtenixPlddtDeltaTerm(EnergyTerm):

    kind: str = "chai_plddt_delta"
    chains_A: List[str] = field(default_factory=list)
    chains_B: List[str] = field(default_factory=list)
    metric: str = "plddt"
    direction: str = "B_gt_A"
    delta_threshold: float = 0.4
    scale: float = 3
    device: Optional[str] = None
    model_name: str = "protenix_mini_esm_v0.5.0"
    conda_env: str = field(default_factory=resolve_protenix_conda_env)
    seed: int = 101

    _cache: Dict[str, float] = field(
        default_factory=dict, init=False, repr=False
    )

    def _metric_for(self, seqs: Dict[str, str], chain_ids: List[str]) -> Optional[float]:
        if not chain_ids:
            return None
        chains = []
        for cid in chain_ids:
            s = seqs.get(cid, "")
            if not s:
                return None
            chains.append((cid, s))

        key = (
            "|".join(f"{cid}:{seq}" for cid, seq in chains)
            + f"|{self.model_name}|{self.seed}|{self.metric}|delta"
        )
        if key in self._cache:
            return self._cache[key]

        val = run_structure_plddt_multichain(
            pred_name=chain_ids[0],
            chains=chains,
            metric=self.metric,
            device=self.device,
            seed=self.seed,
            model_name=self.model_name,
            conda_env=self.conda_env,
        )

        self._cache[key] = float(val)
        return float(val)


    def _plddt_for(self, seqs: Dict[str, str], chain_ids: List[str]) -> Optional[float]:
        return self._metric_for(seqs, chain_ids)

    def penalty(self, seqs: Dict[str, str], compiled: Dict[str, Any]) -> float:
        mA = self._metric_for(seqs, self.chains_A)
        mB = self._metric_for(seqs, self.chains_B)

        if mA is None or mB is None:
            return 2.0

        if self.direction == "A_gt_B":

            delta = mA - mB
        else:

            delta = mB - mA


        loss = max(0.0, float(self.delta_threshold) - float(delta)) * float(self.scale)

        print(f"[{self.metric}] A={mA:.4f}, B={mB:.4f}, delta={delta:.4f}, loss={loss:.4f}")
        return float(loss)


ChaiPlddtDeltaTerm = ProtenixPlddtDeltaTerm


@dataclass
class ResiduePlddtTerm(EnergyTerm):

    kind: str = "residue_plddt"
    chain_id: Optional[str] = None
    segment_filter: Optional[Dict[str, Any]] = None
    positions: Optional[List[int]] = None
    mode: str = "high"
    target: float = 70.0
    scale: float = 100.0
    aggregation: str = "mean"
    softmin_temperature: float = 8.0
    missing_penalty: float = 2.0
    device: Optional[str] = None
    model_name: str = "protenix_mini_esm_v0.5.0"
    conda_env: str = field(default_factory=resolve_protenix_conda_env)
    seed: int = 101

    def _collect_positions(self, compiled: Dict[str, Any]) -> Dict[str, List[int]]:
        out: Dict[str, List[int]] = {}

        if self.positions is not None:
            if not self.chain_id:
                return out
            out[self.chain_id] = sorted(set(int(i) for i in self.positions))
            return out

        for seg in compiled.get("segments", []):
            if self.chain_id is not None and seg.chain_id != self.chain_id:
                continue
            if not _segment_filter(seg, self.segment_filter):
                continue
            out.setdefault(seg.chain_id, [])
            out[seg.chain_id].extend(seg.indices())

        for cid in list(out.keys()):
            out[cid] = sorted(set(int(i) for i in out[cid]))
        return out

    def _aggregate(self, vals: List[float]) -> float:
        arr = np.asarray(vals, dtype=float)
        if arr.size == 0:
            return float("nan")
        if self.aggregation == "min":
            return float(arr.min())
        if self.aggregation == "max":
            return float(arr.max())
        if self.aggregation == "softmin":
            temp = max(1e-6, float(self.softmin_temperature))
            weights = np.exp(-arr / temp)
            weights = weights / max(float(weights.sum()), 1e-12)
            return float((weights * arr).sum())
        return float(arr.mean())

    def penalty(self, seqs: Dict[str, str], compiled: Dict[str, Any]) -> float:
        pos_map = self._collect_positions(compiled)
        if not pos_map:
            return float(self.missing_penalty)

        chain_ids = list(pos_map.keys())
        _maybe_fetch_residue_confidence(
            seqs=seqs,
            chain_ids=chain_ids,
            compiled=compiled,
            device=self.device,
            model_name=self.model_name,
            conda_env=self.conda_env,
            seed=self.seed,
        )

        residue_vals: List[float] = []
        for cid, positions in pos_map.items():
            per_res = _get_residue_plddt_from_compiled(compiled, cid)
            if per_res is None:
                continue
            for i in positions:
                if 0 <= i < len(per_res):
                    residue_vals.append(float(per_res[i]))

        if not residue_vals:
            return float(self.missing_penalty)

        agg_val = self._aggregate(residue_vals)
        score = float(agg_val) / float(self.scale)
        target_score = float(self.target) / float(self.scale)

        if self.mode == "low":
            return float(max(0.0, score - target_score))
        return float(max(0.0, target_score - score))


_TERM_REGISTRY: Dict[str, Callable[[Dict[str, Any]], EnergyTerm]] = {
    "alphabet": lambda p: AlphabetTerm(
        allowed=set(p.get("allowed", AA_SET)),
        chain_ids=list(p.get("chain_ids", [])) if p.get("chain_ids") is not None else None,
    ),
    "fixed_chain_sequence": lambda p: FixedChainSequenceTerm(
        chain_id=str(p["chain_id"]),
        sequence=str(p["sequence"]),
    ),
    "fixed_residues": lambda p: FixedResiduesTerm(
        fixed_residues={
            k: {int(i): str(a) for i, a in v.items()}
            for k, v in p.get("fixed_residues", {}).items()
        }
    ),
    "ss_proxy": lambda p: SecondaryStructureProxyTerm(
        target_map={
            (k.split(":")[0], k.split(":")[1]): v
            for k, v in p.get("target_map", {}).items()
        }
    ),
    "hydrophobic_pattern": lambda p: HydrophobicPatternTerm(
        domain_min_hydro=float(p.get("domain_min_hydro", 0.35)),
        linker_max_hydro=float(p.get("linker_max_hydro", 0.25)),
    ),
    "interface_proxy": lambda p: InterfaceProxyTerm(
        binder_chain=str(p.get("binder_chain", "B")),
        binder_segment=str(p.get("binder_segment", "iface")),
        target_chain=str(p.get("target_chain", "T")),
        target_segment=str(p.get("target_segment", "epitope")),
        desired_binder_hydro=float(p.get("desired_binder_hydro", 0.45)),
    ),
    "segment_composition": lambda p: SegmentCompositionTerm(
        aa_set=set(p.get("aa_set", "")),
        min_frac=float(p.get("min_frac", 0.0)),
        max_frac=float(p.get("max_frac", 1.0)),
        segment_filter=p.get("segment_filter"),
    ),
    "max_run": lambda p: MaxRunTerm(
        aa_set=set(p.get("aa_set", "")),
        max_run=int(p.get("max_run", 3)),
        segment_filter=p.get("segment_filter"),
    ),
    "chai_plddt": lambda p: ProtenixPlddtTerm(
        chains=p.get("chains", None),
        mode=str(p.get("mode", "high")),
        target=float(p.get("target", 70.0)),
        scale=float(p.get("scale", 100.0)),
        device=p.get("device", None),
        model_name=str(p.get("model_name", "protenix_mini_esm_v0.5.0")),
        conda_env=resolve_protenix_conda_env(p.get("conda_env")),
        seed=int(p.get("seed", 101)),
    ),
    "chai_plddt_delta": lambda p: ProtenixPlddtDeltaTerm(
        chains_A=list(p.get("chains_A", [])),
        chains_B=list(p.get("chains_B", [])),
        metric=str(p.get("metric", "plddt")),
        direction=str(p.get("direction", "B_gt_A")),
        delta_threshold=float(p.get("delta_threshold", 0.4)),
        scale=float(p.get("scale", 100.0)),
        device=p.get("device", None),
        model_name=str(p.get("model_name", "protenix_mini_esm_v0.5.0")),
        conda_env=resolve_protenix_conda_env(p.get("conda_env")),
        seed=int(p.get("seed", 101)),
    ),
    "residue_plddt": lambda p: ResiduePlddtTerm(
        chain_id=p.get("chain_id", None),
        segment_filter=p.get("segment_filter", None),
        positions=[int(x) for x in p.get("positions", [])]
        if p.get("positions") is not None
        else None,
        mode=str(p.get("mode", "high")),
        target=float(p.get("target", 70.0)),
        scale=float(p.get("scale", 100.0)),
        aggregation=str(p.get("aggregation", "mean")),
        softmin_temperature=float(p.get("softmin_temperature", 8.0)),
        missing_penalty=float(p.get("missing_penalty", 2.0)),
        device=p.get("device", None),
        model_name=str(p.get("model_name", "protenix_mini_esm_v0.5.0")),
        conda_env=resolve_protenix_conda_env(p.get("conda_env")),
        seed=int(p.get("seed", 101)),
    ),
}


def build_terms_from_specs(
    specs: List[Dict[str, Any]], stage: Optional[str] = None
) -> List[tuple[float, EnergyTerm]]:


    terms: List[tuple[float, EnergyTerm]] = []
    for sp in specs:
        if stage is not None:
            sp_stage = sp.get("stage", "fast")
            if sp_stage != stage:
                continue

        kind = sp["kind"]
        w = float(sp.get("weight", 1.0))
        p = sp.get("params", {}) or {}

        if kind in _TERM_REGISTRY:
            terms.append((w, _TERM_REGISTRY[kind](p)))
        else:
            terms.append((w, AlphabetTerm()))
    return terms


def energy_breakdown(
    seqs: Dict[str, str],
    compiled: Dict[str, Any],
    weighted_terms: List[tuple[float, EnergyTerm]],
) -> Dict[str, float]:


    out: Dict[str, float] = {}
    for w, t in weighted_terms:
        val = w * t.penalty(seqs, compiled)
        out[t.kind] = out.get(t.kind, 0.0) + float(val)
    out["total"] = float(sum(out.values()))
    return out


def energy(
    seqs: Dict[str, str],
    compiled: Dict[str, Any],
    weighted_terms: List[tuple[float, EnergyTerm]],
) -> float:


    e = 0.0
    for w, t in weighted_terms:
        e += w * t.penalty(seqs, compiled)
    return float(e)
