import json
import math
import os
import re
import subprocess
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import cache_utils as cache
from askcos_tree_utils import parse_uds_paths, route_summary


TREE_SEARCH_CONTROLLER_URL = "http://127.0.0.1:9100/api/tree-search/controller/call-sync-without-token"
RISK_RULES_PATH = os.path.join(os.path.dirname(__file__), "risk_rules.json")
CONSTRAINT_LOOP_LOG_PATH = os.path.join(os.path.dirname(__file__), "runtime_jobs", "constraint_loop_logs.jsonl")
ROUTE_EVAL_LOG_PATH = os.path.join(os.path.dirname(__file__), "runtime_jobs", "route_eval_logs.jsonl")
ROUTE_FEEDBACK_LOG_PATH = os.path.join(os.path.dirname(__file__), "runtime_jobs", "route_feedback_logs.jsonl")


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_payload(
    *,
    target_smiles: str,
    backend: str,
    max_depth: int,
    max_paths: int,
    expansion_time: int,
    max_branching: int,
    max_num_templates: int,
    threshold: float,
    fast_filter_threshold: float,
) -> Dict[str, Any]:
    return {
        "backend": backend,
        "smiles": target_smiles,
        "description": target_smiles,
        "expand_one_options": {
            "retro_backend_options": [
                {
                    "retro_backend": "template_relevance",
                    "retro_model_name": "reaxys",
                    "max_num_templates": max_num_templates,
                    "max_cum_prob": 0.999,
                    "threshold": threshold,
                    "attribute_filter": [],
                }
            ],
            "banned_chemicals": [],
            "banned_reactions": [],
            "use_fast_filter": True,
            "filter_threshold": fast_filter_threshold,
            "retro_rerank_backend": "relevance_heuristic",
            "atom_map_backend": "rxnmapper",
            "cluster_precursors": False,
            "cluster_setting": {
                "feature": "original",
                "cluster_method": "hdbscan",
                "fp_type": "morgan",
                "fp_length": 512,
                "fp_radius": 1,
                "classification_threshold": 0.2,
            },
            "extract_template": False,
            "return_reacting_atoms": True,
            "selectivity_check": False,
        },
        "build_tree_options": {
            "buyable_logic": "and",
            "buyables_source": None,
            "expansion_time": expansion_time,
            "max_branching": max_branching,
            "max_depth": max_depth,
            "exploration_weight": 1.0,
            "return_first": False,
            "max_trees": 500,
            "max_chemicals": None,
            "max_reactions": None,
            "max_templates": None,
            "max_iterations": None,
            "max_ppg_logic": "none",
            "max_ppg": None,
            "max_scscore_logic": "none",
            "max_scscore": None,
            "chemical_property_logic": "none",
            "max_chemprop_c": None,
            "max_chemprop_n": None,
            "max_chemprop_o": None,
            "max_chemprop_h": None,
            "chemical_popularity_logic": "none",
            "min_chempop_reactants": 5,
            "min_chempop_products": 5,
        },
        "enumerate_paths_options": {
            "path_format": "json",
            "json_format": "nodelink",
            "sorting_metric": "plausibility",
            "validate_paths": True,
            "score_trees": False,
            "cluster_trees": False,
            "cluster_method": "hdbscan",
            "min_samples": 5,
            "min_cluster_size": 5,
            "paths_only": False,
            "max_paths": max_paths,
        },
    }


def _call_tree_search(payload: Dict[str, Any], expansion_time: int) -> Dict[str, Any]:
    result = subprocess.run(
        [
            "curl",
            "-sS",
            TREE_SEARCH_CONTROLLER_URL,
            "--header",
            "Content-Type: application/json",
            "--request",
            "POST",
            "--data",
            json.dumps(payload, ensure_ascii=False),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=max(300, expansion_time + 300),
    )
    data = json.loads(result.stdout)
    if int(data.get("status_code", 500)) != 200:
        raise RuntimeError(f"Tree search API 回傳失敗: {data.get('message', 'unknown')}")
    return data


def _norm_inverse(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 1.0
    z = (x - lo) / (hi - lo)
    return max(0.0, min(1.0, 1.0 - z))


def _norm_direct(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 1.0
    z = (x - lo) / (hi - lo)
    return max(0.0, min(1.0, z))


def _cost_agent(routes: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    costs = [_safe_float(r.get("precursor_cost"), 1e9) for r in routes]
    lo, hi = min(costs), max(costs)
    out = {}
    for r, c in zip(routes, costs):
        score = _norm_inverse(c, lo, hi)
        out[int(r["route_id"])] = {
            "score": score,
            "reason": f"成本估計={c:.4f}",
        }
    return out


def _success_agent(routes: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    vals = []
    for r in routes:
        p = r.get("plausibility_product")
        p = _safe_float(p, 0.0) if p is not None else 0.0
        vals.append(p)
    lo, hi = min(vals), max(vals)
    out = {}
    for r, p in zip(routes, vals):
        score = _norm_direct(p, lo, hi)
        out[int(r["route_id"])] = {
            "score": score,
            "reason": f"plausibility_product={p:.6f}",
        }
    return out


def _supply_agent(routes: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out = {}
    for r in routes:
        n_leaf = max(1, int(r.get("num_leaf_precursors") or 1))
        c = _safe_float(r.get("precursor_cost"), 0.0)
        # 葉節點越少、成本越低通常供應鏈壓力越小
        leaf_score = 1.0 / n_leaf
        cost_penalty = 1.0 / (1.0 + max(0.0, c))
        score = 0.7 * leaf_score + 0.3 * cost_penalty
        out[int(r["route_id"])] = {
            "score": max(0.0, min(1.0, score)),
            "reason": f"leaf={n_leaf}, cost={c:.4f}",
        }
    return out


def _load_risk_rules() -> Dict[str, Any]:
    default_rules: Dict[str, Any] = {
        "severity_weights": {"high": 1.0, "medium": 0.5, "low": 0.2, "unknown": 0.1},
        "high_risk_tokens": ["N=[N+]=[N-]", "N3"],
        "medium_risk_tokens": ["ClCl", "BrBr", "C[N+](C)=CCl"],
        "low_risk_tokens": [],
        "pubchem": {
            "enabled": True,
            "high_hcodes": ["H300", "H310", "H330", "H340", "H350", "H360", "H370"],
            "medium_hcodes": ["H301", "H311", "H331", "H314", "H317", "H334", "H351", "H372", "H373"],
            "low_hcodes": ["H315", "H319", "H335", "H336"],
            "high_keywords": ["fatal", "explosive", "carcinogen", "mutagen", "reproductive toxicity"],
            "medium_keywords": ["toxic", "corrosive", "sensitizer", "organ damage"],
            "low_keywords": ["irritant", "harmful", "drowsiness"],
        },
    }
    try:
        with open(RISK_RULES_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            return default_rules
        out = dict(default_rules)
        out.update({k: v for k, v in cfg.items() if k in out or k == "severity_weights"})
        if "severity_weights" in cfg and isinstance(cfg["severity_weights"], dict):
            merged = dict(default_rules["severity_weights"])
            merged.update(cfg["severity_weights"])
            out["severity_weights"] = merged
        return out
    except Exception:
        return default_rules


def _collect_strings(obj: Any) -> List[str]:
    out: List[str] = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_collect_strings(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_collect_strings(v))
    return out


def _pubchem_cid_from_smiles(smiles: str) -> int:
    key = cache.build_key("pubchem:cid:v1", smiles=smiles)
    hit = cache.get(key)
    if isinstance(hit, int):
        return hit
    if isinstance(hit, str) and hit.isdigit():
        return int(hit)
    result = subprocess.run(
        [
            "curl",
            "-sS",
            "--get",
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/cids/JSON",
            "--data-urlencode",
            f"smiles={smiles}",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    data = json.loads(result.stdout)
    cid = int((data.get("IdentifierList", {}).get("CID") or [0])[0])
    if cid > 0:
        cache.set(key, cid)
    return cid


def _pubchem_hazard_payload(cid: int) -> Dict[str, Any]:
    key = cache.build_key("pubchem:hazard_payload:v1", cid=cid)
    hit = cache.get(key)
    if isinstance(hit, dict):
        return hit
    result = subprocess.run(
        [
            "curl",
            "-sS",
            "--get",
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON",
            "--data-urlencode",
            "heading=Hazards Identification",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=25,
    )
    data = json.loads(result.stdout)
    if isinstance(data, dict):
        cache.set(key, data)
    return data if isinstance(data, dict) else {}


def _pubchem_hazard_assess(smiles: str, rules: Dict[str, Any]) -> Dict[str, Any]:
    key = cache.build_key("pubchem:hazard_assess:v1", smiles=smiles, rules=rules.get("pubchem", {}))
    hit = cache.get(key)
    if isinstance(hit, dict):
        return hit

    pubchem_rules = rules.get("pubchem", {}) if isinstance(rules.get("pubchem", {}), dict) else {}
    high_h = set(pubchem_rules.get("high_hcodes", []) or [])
    med_h = set(pubchem_rules.get("medium_hcodes", []) or [])
    low_h = set(pubchem_rules.get("low_hcodes", []) or [])
    high_kw = [x.lower() for x in (pubchem_rules.get("high_keywords", []) or [])]
    med_kw = [x.lower() for x in (pubchem_rules.get("medium_keywords", []) or [])]
    low_kw = [x.lower() for x in (pubchem_rules.get("low_keywords", []) or [])]

    try:
        cid = _pubchem_cid_from_smiles(smiles)
        if cid <= 0:
            out = {"severity": "unknown", "reason": "pubchem_no_cid", "hits": []}
            cache.set(key, out)
            return out
        payload = _pubchem_hazard_payload(cid)
        texts = [t for t in _collect_strings(payload) if isinstance(t, str)]
        merged = " ".join(texts).lower()
        hcodes = set(re.findall(r"\bH\\d{3}\b", " ".join(texts).upper()))

        high_hcode_hits = sorted(list(hcodes.intersection(high_h)))
        med_hcode_hits = sorted(list(hcodes.intersection(med_h)))
        low_hcode_hits = sorted(list(hcodes.intersection(low_h)))
        high_hits = list(high_hcode_hits)
        med_hits = list(med_hcode_hits)
        low_hits = list(low_hcode_hits)
        if not high_hits:
            high_hits.extend([kw for kw in high_kw if kw in merged][:3])
        if not med_hits:
            med_hits.extend([kw for kw in med_kw if kw in merged][:3])
        if not low_hits:
            low_hits.extend([kw for kw in low_kw if kw in merged][:3])

        if high_hits:
            out = {
                "severity": "high",
                "reason": "pubchem_high",
                "hits": high_hits,
                "high_hcode_hits": high_hcode_hits,
                "med_hcode_hits": med_hcode_hits,
                "low_hcode_hits": low_hcode_hits,
            }
        elif med_hits:
            out = {
                "severity": "medium",
                "reason": "pubchem_medium",
                "hits": med_hits,
                "high_hcode_hits": high_hcode_hits,
                "med_hcode_hits": med_hcode_hits,
                "low_hcode_hits": low_hcode_hits,
            }
        elif low_hits:
            out = {
                "severity": "low",
                "reason": "pubchem_low",
                "hits": low_hits,
                "high_hcode_hits": high_hcode_hits,
                "med_hcode_hits": med_hcode_hits,
                "low_hcode_hits": low_hcode_hits,
            }
        else:
            out = {
                "severity": "none",
                "reason": "pubchem_no_hazard_hit",
                "hits": [],
                "high_hcode_hits": high_hcode_hits,
                "med_hcode_hits": med_hcode_hits,
                "low_hcode_hits": low_hcode_hits,
            }
        cache.set(key, out)
        return out
    except Exception as e:
        out = {
            "severity": "unknown",
            "reason": f"pubchem_error:{str(e)[:120]}",
            "hits": [],
            "high_hcode_hits": [],
            "med_hcode_hits": [],
            "low_hcode_hits": [],
        }
        cache.set(key, out)
        return out


def _safety_agent(
    routes: List[Dict[str, Any]],
    banned_tokens: List[str],
    enable_pubchem_hazard: bool = True,
    hazard_leaf_only: bool = True,
    max_unique_hazard_checks: int = 120,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, List[str]], Dict[int, Dict[str, Any]]]:
    rules = _load_risk_rules()
    weights = rules.get("severity_weights", {})
    w_high = _safe_float(weights.get("high", 1.0), 1.0)
    w_medium = _safe_float(weights.get("medium", 0.5), 0.5)
    w_low = _safe_float(weights.get("low", 0.2), 0.2)

    high_tokens = [x for x in (rules.get("high_risk_tokens") or []) if str(x).strip()]
    med_tokens = [x for x in (rules.get("medium_risk_tokens") or []) if str(x).strip()]
    low_tokens = [x for x in (rules.get("low_risk_tokens") or []) if str(x).strip()]
    manual_tokens = [x for x in (banned_tokens or []) if str(x).strip()]
    high_tokens.extend(manual_tokens)
    pubchem_enabled = bool(enable_pubchem_hazard and (rules.get("pubchem", {}) or {}).get("enabled", True))

    # 預先去重查詢 PubChem，避免每條路重複查同一分子
    pubchem_map: Dict[str, Dict[str, Any]] = {}
    if pubchem_enabled:
        candidates: List[str] = []
        for r in routes:
            if hazard_leaf_only:
                candidates.extend([x for x in (r.get("leaf_chemicals") or []) if x and ">>" not in str(x)])
            else:
                candidates.extend([x for x in (r.get("leaf_chemicals") or []) if x and ">>" not in str(x)])
                # reaction_nodes 為反應字串，不適合直接查 CID；先保留 leaf-only 為主
        uniq = []
        seen = set()
        for s in candidates:
            if s not in seen:
                uniq.append(s)
                seen.add(s)
        for smi in uniq[: max(0, int(max_unique_hazard_checks))]:
            pubchem_map[smi] = _pubchem_hazard_assess(smi, rules)

    scores: Dict[int, Dict[str, Any]] = {}
    rejects: Dict[int, List[str]] = {}
    meta: Dict[int, Dict[str, Any]] = {}
    for r in routes:
        rid = int(r["route_id"])
        text_blob = " ".join((r.get("reaction_nodes") or []) + (r.get("leaf_chemicals") or []))
        high_hits = [t for t in high_tokens if t in text_blob]
        med_hits = [t for t in med_tokens if t in text_blob]
        low_hits = [t for t in low_tokens if t in text_blob]
        pubchem_high: List[str] = []
        pubchem_med: List[str] = []
        pubchem_low: List[str] = []
        pubchem_unknown = 0
        pubchem_high_hcode_smiles: List[str] = []

        if pubchem_enabled:
            for smi in (r.get("leaf_chemicals") or []):
                if ">>" in str(smi):
                    continue
                info = pubchem_map.get(smi) or {"severity": "unknown", "reason": "not_checked", "hits": []}
                sev = str(info.get("severity", "unknown")).lower()
                if info.get("high_hcode_hits"):
                    pubchem_high_hcode_smiles.append(smi)
                if sev == "high":
                    pubchem_high.append(smi)
                elif sev == "medium":
                    pubchem_med.append(smi)
                elif sev == "low":
                    pubchem_low.append(smi)
                elif sev == "unknown":
                    pubchem_unknown += 1

        if high_hits or pubchem_high:
            reasons = [f"safety_reject(high-token): 命中 {h}" for h in high_hits[:3]]
            reasons.extend([f"safety_reject(pubchem-high): {s}" for s in pubchem_high[:3]])
            scores[rid] = {
                "score": 0.0,
                "reason": f"命中高危 token/pubchem（token={len(high_hits)}, pubchem={len(pubchem_high)}）",
            }
            rejects[rid] = reasons
            meta[rid] = {
                "high_token_hits": high_hits,
                "pubchem_high_smiles": pubchem_high,
                "pubchem_high_hcode_smiles": sorted(list(set(pubchem_high_hcode_smiles))),
                "pubchem_medium_smiles": pubchem_med,
                "pubchem_low_smiles": pubchem_low,
                "pubchem_unknown_count": pubchem_unknown,
            }
            continue

        # risk = 1 - Π(1 - w_i), safety_score = 1 - risk
        factors: List[float] = (
            [w_medium] * len(med_hits)
            + [w_low] * len(low_hits)
            + [w_medium] * len(pubchem_med)
            + [w_low] * len(pubchem_low)
            + [_safe_float(weights.get("unknown", 0.1), 0.1)] * pubchem_unknown
        )
        prod = 1.0
        for w in factors:
            w = max(0.0, min(1.0, _safe_float(w, 0.0)))
            prod *= (1.0 - w)
        risk = 1.0 - prod
        safety_score = 1.0 - risk
        reason = (
            f"high=0, medium={len(med_hits)+len(pubchem_med)}, low={len(low_hits)+len(pubchem_low)}, "
            f"unknown={pubchem_unknown}, risk={risk:.4f}"
        )
        scores[rid] = {"score": max(0.0, min(1.0, safety_score)), "reason": reason}
        meta[rid] = {
            "high_token_hits": high_hits,
            "pubchem_high_smiles": pubchem_high,
            "pubchem_high_hcode_smiles": sorted(list(set(pubchem_high_hcode_smiles))),
            "pubchem_medium_smiles": pubchem_med,
            "pubchem_low_smiles": pubchem_low,
            "pubchem_unknown_count": pubchem_unknown,
        }
    return scores, rejects, meta


def _objective_weights(objective: str) -> Dict[str, float]:
    key = (objective or "balanced").strip().lower()
    if key in {"cheapest", "cost", "最便宜"}:
        return {"cost": 0.65, "success": 0.15, "safety": 0.10, "supply": 0.10}
    if key in {"highest_success", "success", "成功率最高"}:
        return {"cost": 0.10, "success": 0.60, "safety": 0.20, "supply": 0.10}
    if key in {"safest", "安全"}:
        return {"cost": 0.10, "success": 0.20, "safety": 0.55, "supply": 0.15}
    return {"cost": 0.30, "success": 0.30, "safety": 0.20, "supply": 0.20}


def _parse_constraint_text(constraint_text: str) -> Dict[str, Any]:
    text = (constraint_text or "").strip()
    if not text:
        return {"hard": {}, "soft": {}}
    lower = text.lower()
    hard: Dict[str, Any] = {}
    soft: Dict[str, Any] = {}
    clauses = re.split(r"[，,。；;！!？?\n]+", text)

    zh_num_map = {
        "一": 1,
        "二": 2,
        "兩": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }

    def _zh_num_to_int(s: str) -> int:
        s = (s or "").strip()
        if not s:
            return 0
        if s.isdigit():
            return int(s)
        if s == "十":
            return 10
        if "十" in s:
            parts = s.split("十")
            left = zh_num_map.get(parts[0], 1 if parts[0] == "" else 0)
            right = zh_num_map.get(parts[1], 0) if len(parts) > 1 else 0
            return left * 10 + right
        return zh_num_map.get(s, 0)

    def _is_hard_clause(c: str) -> bool:
        cl = c.lower()
        return any(k in cl for k in ["不能", "禁止", "不可", "must", "不得", "一定要", "必須", "ban", "avoid all"])

    def _is_soft_clause(c: str) -> bool:
        cl = c.lower()
        return any(k in cl for k in ["盡量", "優先", "希望", "最好", "盡可能", "prefer", "ideally", "建議"])

    for c in clauses:
        c = c.strip()
        if not c:
            continue
        is_hard = _is_hard_clause(c)
        is_soft = _is_soft_clause(c) or not is_hard

        # depth / 步數
        m = re.search(
            r"(?:(?:最多|至多|不超過|低於|小於|不能超過)\s*([0-9]+|[一二兩三四五六七八九十]{1,3})\s*步)|"
            r"(?:([0-9]+|[一二兩三四五六七八九十]{1,3})\s*步內)|"
            r"(?:max[_ ]?depth\s*([0-9]+))|"
            r"(?:步數\s*(?:不超過|低於|小於|<=|<)\s*([0-9]+|[一二兩三四五六七八九十]{1,3}))",
            c,
            flags=re.I,
        )
        if m:
            raw_n = ""
            for idx in range(1, 6):
                part = m.group(idx)
                if part:
                    raw_n = str(part).strip()
                    break
            n = _zh_num_to_int(raw_n)
            if n > 0:
                if is_hard:
                    hard["max_depth"] = n
                elif is_soft:
                    soft["max_depth"] = n

        # 成本上限（支援「成本 50 以下 / 不要太貴 / 便宜優先」）
        m = re.search(
            r"(成本|cost|價格|花費)\s*(<=|<|不超過|低於|小於|以下)?\s*([0-9]+(?:\.[0-9]+)?)",
            c,
            flags=re.I,
        )
        if m:
            v = float(m.group(3))
            if is_hard:
                hard["max_precursor_cost"] = v
            else:
                soft["max_precursor_cost"] = v
        elif any(k in c for k in ["不要太貴", "別太貴", "便宜優先", "成本低", "省錢"]):
            soft.setdefault("prefer_low_cost", True)

        # leaf / 前驅物數
        m = re.search(
            r"(leaf|前驅物|終端前驅物)\s*(<=|<|不超過|低於|小於|以下)?\s*([0-9]+|[一二兩三四五六七八九十]{1,3})",
            c,
            flags=re.I,
        )
        if m:
            n = _zh_num_to_int(m.group(3))
            if n > 0:
                if is_hard:
                    hard["max_leaf_precursors"] = n
                else:
                    soft["max_leaf_precursors"] = n

    if any(k in lower for k in ["不能", "禁止", "avoid", "ban"]):
        segs = re.findall(r"(?:禁用|禁止|avoid|ban)\s*[:：]?\s*([^\n，,。；;]+)", text, flags=re.I)
        toks = []
        for seg in segs:
            parts = re.split(r"[、,，\s]+|與|和|以及|and", seg)
            for p in parts:
                p = p.strip()
                if re.fullmatch(r"[A-Za-z0-9@+\-=\[\]\(\)#\\/\.]{2,}", p):
                    toks.append(p)
        if toks:
            hard["banned_tokens"] = toks

    # 溶劑/安全語意映射（MVP）
    if any(k in lower for k in ["含氯溶劑", "鹵素溶劑", "halogen solvent", "chlorinated solvent"]):
        hard.setdefault("banned_tokens", [])
        hard["banned_tokens"].extend(["ClCCl", "ClCl", "CH2Cl2", "CCl4"])
    if any(k in lower for k in ["高危", "high hazard", "危險分子禁用", "不要高危"]):
        hard["forbid_high_hazard"] = True
    if any(k in lower for k in ["有毒盡量避免", "盡量不要有毒", "avoid toxic", "toxic avoid"]):
        soft["prefer_low_hazard"] = True
    if any(k in lower for k in ["供應穩定", "穩定供應", "supply stable"]):
        soft["prefer_supply_stability"] = True
    if any(k in lower for k in ["成功率高", "成功優先", "成功率優先", "highest success", "success first"]):
        soft["prefer_high_success"] = True
    if any(k in lower for k in ["步驟不要太多", "步數不要太多", "步驟少", "步數少"]):
        soft.setdefault("max_depth", 4)

    # 去重
    if isinstance(hard.get("banned_tokens"), list):
        seen = set()
        dedup = []
        for x in hard["banned_tokens"]:
            s = str(x).strip()
            if s and s not in seen:
                dedup.append(s)
                seen.add(s)
        hard["banned_tokens"] = dedup
    return {"hard": hard, "soft": soft}


def _merge_constraints(structured: Dict[str, Any], text_constraints: Dict[str, Any]) -> Dict[str, Any]:
    out = {"hard": {}, "soft": {}}
    for scope in ("hard", "soft"):
        src_a = structured.get(scope, {}) if isinstance(structured, dict) else {}
        src_b = text_constraints.get(scope, {}) if isinstance(text_constraints, dict) else {}
        if isinstance(src_a, dict):
            out[scope].update(src_a)
        if isinstance(src_b, dict):
            out[scope].update(src_b)
    return out


def _evaluate_constraints(
    *,
    route: Dict[str, Any],
    hard_constraints: Dict[str, Any],
    soft_constraints: Dict[str, Any],
) -> Dict[str, Any]:
    hard_violations: List[str] = []
    soft_violations: List[str] = []
    soft_penalty = 0.0
    checks = 0
    violated = 0

    depth = int(route.get("depth") or route.get("num_reactions") or 0)
    cost = _safe_float(route.get("precursor_cost"), 0.0)
    n_leaf = int(route.get("num_leaf_precursors") or 0)
    text_blob = " ".join((route.get("reaction_nodes") or []) + (route.get("leaf_chemicals") or []))

    if "max_depth" in hard_constraints:
        checks += 1
        lim = int(hard_constraints.get("max_depth", 0))
        if depth > lim:
            violated += 1
            hard_violations.append(f"depth>{lim} (actual={depth})")
    if "max_precursor_cost" in hard_constraints:
        checks += 1
        lim = _safe_float(hard_constraints.get("max_precursor_cost"), 1e18)
        if cost > lim:
            violated += 1
            hard_violations.append(f"precursor_cost>{lim:.4f} (actual={cost:.4f})")
    if "max_leaf_precursors" in hard_constraints:
        checks += 1
        lim = int(hard_constraints.get("max_leaf_precursors", 10**9))
        if n_leaf > lim:
            violated += 1
            hard_violations.append(f"num_leaf_precursors>{lim} (actual={n_leaf})")
    for t in (hard_constraints.get("banned_tokens") or []):
        checks += 1
        tok = str(t).strip()
        if tok and tok in text_blob:
            violated += 1
            hard_violations.append(f"banned_token={tok}")
    if bool(hard_constraints.get("forbid_high_hazard", False)) and bool(route.get("_has_pubchem_high_hcode", False)):
        checks += 1
        violated += 1
        hard_violations.append("forbid_high_hazard=true and route_has_pubchem_high_hcode=true")

    if "max_precursor_cost" in soft_constraints:
        checks += 1
        lim = _safe_float(soft_constraints.get("max_precursor_cost"), 1e18)
        if cost > lim:
            violated += 1
            ratio = (cost - lim) / max(1e-6, lim)
            p = min(0.30, 0.10 + 0.30 * ratio)
            soft_penalty += p
            soft_violations.append(f"soft_cost>{lim:.4f} (actual={cost:.4f}, penalty={p:.3f})")
    if "max_leaf_precursors" in soft_constraints:
        checks += 1
        lim = int(soft_constraints.get("max_leaf_precursors", 10**9))
        if n_leaf > lim:
            violated += 1
            p = min(0.25, 0.08 * (n_leaf - lim))
            soft_penalty += p
            soft_violations.append(f"soft_leaf>{lim} (actual={n_leaf}, penalty={p:.3f})")
    if "max_depth" in soft_constraints:
        checks += 1
        lim = int(soft_constraints.get("max_depth", 10**9))
        if depth > lim:
            violated += 1
            p = min(0.25, 0.08 * (depth - lim))
            soft_penalty += p
            soft_violations.append(f"soft_depth>{lim} (actual={depth}, penalty={p:.3f})")

    satisfaction_rate = 1.0 if checks <= 0 else max(0.0, min(1.0, 1.0 - (violated / checks)))
    return {
        "hard_violations": hard_violations,
        "soft_violations": soft_violations,
        "soft_penalty": max(0.0, min(0.7, soft_penalty)),
        "checks": checks,
        "violated": violated,
        "satisfaction_rate": satisfaction_rate,
    }


def _apply_soft_preferences(
    *,
    soft_constraints: Dict[str, Any],
    c: float,
    s: float,
    f: float,
    y: float,
) -> Tuple[float, List[str]]:
    delta = 0.0
    reasons: List[str] = []
    if bool(soft_constraints.get("prefer_low_cost", False)):
        add = 0.04 * c
        delta += add
        reasons.append(f"prefer_low_cost:+{add:.3f}")
    if bool(soft_constraints.get("prefer_low_hazard", False)):
        add = 0.05 * f
        delta += add
        reasons.append(f"prefer_low_hazard:+{add:.3f}")
    if bool(soft_constraints.get("prefer_supply_stability", False)):
        add = 0.04 * y
        delta += add
        reasons.append(f"prefer_supply_stability:+{add:.3f}")
    if bool(soft_constraints.get("prefer_high_success", False)):
        add = 0.05 * s
        delta += add
        reasons.append(f"prefer_high_success:+{add:.3f}")
    return delta, reasons


def _soft_penalty_for_banned_tokens(route: Dict[str, Any], banned_tokens: List[str]) -> Tuple[float, List[str]]:
    if not banned_tokens:
        return 0.0, []
    text_blob = " ".join((route.get("reaction_nodes") or []) + (route.get("leaf_chemicals") or []))
    hits = [t for t in banned_tokens if str(t).strip() and str(t) in text_blob]
    if not hits:
        return 0.0, []
    p = min(0.45, 0.12 * len(hits))
    return p, [f"exploration_soft_ban_hits={hits[:4]} penalty={p:.3f}"]


def _build_relaxation_suggestions(hard_reject_reasons: List[str], hard_constraints: Dict[str, Any]) -> List[str]:
    if not hard_reject_reasons:
        return []
    suggestions: List[str] = []
    if any("depth>" in x for x in hard_reject_reasons) and "max_depth" in hard_constraints:
        suggestions.append(
            f"可放寬 max_depth：{hard_constraints.get('max_depth')} -> {int(hard_constraints.get('max_depth', 0)) + 1}"
        )
    if any("precursor_cost>" in x for x in hard_reject_reasons) and "max_precursor_cost" in hard_constraints:
        v = _safe_float(hard_constraints.get("max_precursor_cost"), 0.0)
        suggestions.append(f"可放寬 max_precursor_cost：{v:.2f} -> {v * 1.15:.2f}")
    if any("num_leaf_precursors>" in x for x in hard_reject_reasons) and "max_leaf_precursors" in hard_constraints:
        suggestions.append(
            f"可放寬 max_leaf_precursors：{hard_constraints.get('max_leaf_precursors')} -> {int(hard_constraints.get('max_leaf_precursors', 0)) + 1}"
        )
    if any("banned_token=" in x for x in hard_reject_reasons):
        suggestions.append("檢視 banned_tokens 是否過嚴，可改為 soft 規則或只禁用 high-risk token。")
    return suggestions[:6]


def _relaxed_constraints(hard_constraints: Dict[str, Any], soft_constraints: Dict[str, Any]) -> Dict[str, Any]:
    hard = dict(hard_constraints or {})
    soft = dict(soft_constraints or {})
    if "max_depth" in hard:
        hard["max_depth"] = int(hard.get("max_depth", 0)) + 1
    if "max_precursor_cost" in hard:
        v = _safe_float(hard.get("max_precursor_cost"), 0.0)
        hard["max_precursor_cost"] = round(v * 1.15, 4)
    if "max_leaf_precursors" in hard:
        hard["max_leaf_precursors"] = int(hard.get("max_leaf_precursors", 0)) + 1
    # banned token 在放寬版本先保留；若全部失敗再由使用者決定是否降級為 soft
    if "max_depth" in soft:
        soft["max_depth"] = int(soft.get("max_depth", 0)) + 1
    if "max_precursor_cost" in soft:
        v = _safe_float(soft.get("max_precursor_cost"), 0.0)
        soft["max_precursor_cost"] = round(v * 1.10, 4)
    if "max_leaf_precursors" in soft:
        soft["max_leaf_precursors"] = int(soft.get("max_leaf_precursors", 0)) + 1
    return {"hard": hard, "soft": soft}


def _append_constraint_loop_log(record: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(CONSTRAINT_LOOP_LOG_PATH), exist_ok=True)
        with open(CONSTRAINT_LOOP_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _read_jsonl(path: str, limit: int = 2000) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        out.append(obj)
                except Exception:
                    continue
    except Exception:
        return []
    if limit > 0:
        return out[-limit:]
    return out


def _vote_label(score: float, rejected: bool, t_low: float, t_high: float) -> str:
    if rejected:
        return "reject"
    if score < t_low:
        return "reject"
    if score >= t_high:
        return "approve"
    return "neutral"


def _build_critic_votes(
    *,
    cost_score: float,
    success_score: float,
    safety_score: float,
    supply_score: float,
    rejected: bool,
    vote_low_threshold: float,
    vote_high_threshold: float,
) -> Dict[str, str]:
    return {
        "cost_agent": _vote_label(cost_score, False, vote_low_threshold, vote_high_threshold),
        "success_agent": _vote_label(success_score, False, vote_low_threshold, vote_high_threshold),
        "safety_agent": _vote_label(safety_score, rejected, vote_low_threshold, vote_high_threshold),
        "supply_agent": _vote_label(supply_score, False, vote_low_threshold, vote_high_threshold),
    }


def _disagreement_std(vals: List[float]) -> float:
    if not vals:
        return 0.0
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / len(vals)
    return math.sqrt(max(0.0, var))


def run_askcos_route_recommendation(
    target_smiles: str,
    objective: str = "balanced",
    backend: str = "mcts",
    max_depth: int = 5,
    max_paths: int = 120,
    expansion_time: int = 180,
    max_branching: int = 25,
    max_num_templates: int = 1000,
    threshold: float = 0.3,
    fast_filter_threshold: float = 0.001,
    top_n: int = 10,
    banned_tokens: List[str] = None,
    enable_pubchem_hazard: bool = True,
    hazard_leaf_only: bool = True,
    max_unique_hazard_checks: int = 120,
    vote_low_threshold: float = 0.33,
    vote_high_threshold: float = 0.67,
    contested_std_threshold: float = 0.25,
    constraints: Dict[str, Any] = None,
    constraint_text: str = "",
    constraint_parse_mode: str = "hybrid",
    auto_relax_if_infeasible: bool = True,
    strict_safety_mode: bool = True,
    exploration_mode: bool = False,
    use_cache: bool = True,
) -> str:
    if not target_smiles:
        return "錯誤：未提供目標分子 SMILES。"

    payload = _build_payload(
        target_smiles=target_smiles,
        backend=backend,
        max_depth=max_depth,
        max_paths=max_paths,
        expansion_time=expansion_time,
        max_branching=max_branching,
        max_num_templates=max_num_templates,
        threshold=threshold,
        fast_filter_threshold=fast_filter_threshold,
    )
    cache_key = cache.build_key(
        "askcos:route_reco:v1",
        payload=payload,
        objective=objective,
        top_n=top_n,
        banned_tokens=banned_tokens or [],
        enable_pubchem_hazard=enable_pubchem_hazard,
        hazard_leaf_only=hazard_leaf_only,
        max_unique_hazard_checks=max_unique_hazard_checks,
        vote_low_threshold=vote_low_threshold,
        vote_high_threshold=vote_high_threshold,
        contested_std_threshold=contested_std_threshold,
        constraints=constraints or {},
        constraint_text=constraint_text,
        constraint_parse_mode=constraint_parse_mode,
        auto_relax_if_infeasible=auto_relax_if_infeasible,
        strict_safety_mode=strict_safety_mode,
        exploration_mode=exploration_mode,
    )
    if use_cache:
        hit = cache.get(cache_key)
        if isinstance(hit, str) and hit.strip():
            return hit

    try:
        data = _call_tree_search(payload, expansion_time=expansion_time)
    except subprocess.TimeoutExpired:
        return f"路線推薦逾時：expansion_time={expansion_time}，請提高到 240~420 秒後再試。"
    except Exception as e:
        return f"路線推薦失敗：{e}"

    result = data.get("result", {}) if isinstance(data, dict) else {}
    stats = result.get("stats", {}) if isinstance(result, dict) else {}
    uds = result.get("uds", {}) if isinstance(result, dict) else {}
    routes_raw = parse_uds_paths(uds if isinstance(uds, dict) else {})
    if not routes_raw:
        return "未取得可分析路線（uds.pathways 為空）。"

    routes = [route_summary(r) for r in routes_raw]
    mode = (constraint_parse_mode or "hybrid").strip().lower()
    if mode == "llm_only":
        # LLM 解析結果應由外部傳入 constraints；此處不再做規則抽取
        text_constraints = {"hard": {}, "soft": {}}
    elif mode == "rule_only":
        text_constraints = _parse_constraint_text(constraint_text)
        constraints = {}
    else:
        # hybrid: 規則 + 外部（可為 LLM）融合
        text_constraints = _parse_constraint_text(constraint_text)
    merged_constraints = _merge_constraints(constraints or {}, text_constraints)
    hard_constraints = merged_constraints.get("hard", {}) if isinstance(merged_constraints.get("hard"), dict) else {}
    soft_constraints = merged_constraints.get("soft", {}) if isinstance(merged_constraints.get("soft"), dict) else {}
    constrained_banned = [x for x in (hard_constraints.get("banned_tokens") or []) if str(x).strip()]
    effective_banned_tokens = (banned_tokens or []) + constrained_banned

    cost = _cost_agent(routes)
    succ = _success_agent(routes)
    supply = _supply_agent(routes)
    safety, rejected, safety_meta = _safety_agent(
        routes,
        effective_banned_tokens,
        enable_pubchem_hazard=enable_pubchem_hazard,
        hazard_leaf_only=hazard_leaf_only,
        max_unique_hazard_checks=max_unique_hazard_checks,
    )
    weights = _objective_weights(objective)

    def _score_once(hc: Dict[str, Any], sc: Dict[str, Any]) -> Dict[str, Any]:
        scored_local: List[Dict[str, Any]] = []
        for r in routes:
            rid = int(r["route_id"])
            c = cost[rid]["score"]
            s = succ[rid]["score"]
            y = supply[rid]["score"]
            f = safety[rid]["score"]
            is_rejected = rid in rejected
            route_ext = dict(r)
            meta = safety_meta.get(rid, {})
            route_ext["_has_high_hazard"] = bool((meta.get("high_token_hits") or []) or (meta.get("pubchem_high_smiles") or []))
            route_ext["_has_pubchem_high_hcode"] = bool(meta.get("pubchem_high_hcode_smiles") or [])
            c_eval = _evaluate_constraints(route=route_ext, hard_constraints=hc, soft_constraints=sc)
            hard_violations = c_eval["hard_violations"]
            soft_violations = c_eval["soft_violations"]
            soft_penalty = _safe_float(c_eval["soft_penalty"], 0.0)
            if hard_violations:
                is_rejected = True
            if strict_safety_mode and route_ext.get("_has_pubchem_high_hcode", False):
                is_rejected = True
                hard_violations = list(hard_violations) + ["strict_safety_mode: pubchem_high_hcode"]
            votes = _build_critic_votes(
                cost_score=c,
                success_score=s,
                safety_score=f,
                supply_score=y,
                rejected=is_rejected,
                vote_low_threshold=vote_low_threshold,
                vote_high_threshold=vote_high_threshold,
            )
            approve_count = sum(1 for v in votes.values() if v == "approve")
            reject_count = sum(1 for v in votes.values() if v == "reject")
            neutral_count = sum(1 for v in votes.values() if v == "neutral")
            std_val = _disagreement_std([c, s, f, y])
            has_vote_conflict = approve_count > 0 and reject_count > 0
            contested = bool((std_val >= contested_std_threshold or has_vote_conflict) and not is_rejected)
            contested_reason = []
            if std_val >= contested_std_threshold:
                contested_reason.append(f"high_disagreement_std={std_val:.3f}")
            if has_vote_conflict:
                contested_reason.append(
                    f"vote_conflict(approve={approve_count}, reject={reject_count}, neutral={neutral_count})"
                )
            base_final = weights["cost"] * c + weights["success"] * s + weights["safety"] * f + weights["supply"] * y
            pref_bonus, pref_reasons = _apply_soft_preferences(soft_constraints=sc, c=c, s=s, f=f, y=y)
            final = max(0.0, min(1.0, base_final - soft_penalty + pref_bonus))
            if pref_reasons:
                soft_violations = list(soft_violations) + [f"soft_preference_bonus:{';'.join(pref_reasons)}"]
            scored_local.append(
                {
                    **route_ext,
                    "final_score": final,
                    "agent_scores": {
                        "cost_agent": cost[rid],
                        "success_agent": succ[rid],
                        "safety_agent": safety[rid],
                        "supply_agent": supply[rid],
                    },
                    "critic_votes": votes,
                    "vote_summary": {"approve": approve_count, "neutral": neutral_count, "reject": reject_count},
                    "disagreement_std": std_val,
                    "contested": contested,
                    "contested_reason": contested_reason,
                    "rejected": is_rejected,
                    "constraint_eval": c_eval,
                    "reject_reasons": (rejected.get(rid, []) + [f"constraint_reject: {x}" for x in hard_violations]),
                    "soft_penalty": soft_penalty,
                    "soft_violations": soft_violations,
                }
            )
        survivors_local = [x for x in scored_local if not x["rejected"]]
        survivors_local.sort(key=lambda x: x["final_score"], reverse=True)
        dropped_local = [x for x in scored_local if x["rejected"]]
        contested_local = [x for x in survivors_local if x.get("contested")]
        contested_local.sort(key=lambda x: x.get("disagreement_std", 0.0), reverse=True)
        hard_reason_pool_local: List[str] = []
        for x in dropped_local:
            hard_reason_pool_local.extend([r for r in (x.get("reject_reasons") or []) if "constraint_reject:" in r])
        return {
            "scored": scored_local,
            "survivors": survivors_local,
            "dropped": dropped_local,
            "contested": contested_local,
            "hard_reasons": hard_reason_pool_local,
        }

    pass1 = _score_once(hard_constraints, soft_constraints)
    active_hard = hard_constraints
    active_soft = soft_constraints
    loop_mode = "original"
    if auto_relax_if_infeasible and len(pass1["survivors"]) == 0 and (hard_constraints or soft_constraints):
        relaxed = _relaxed_constraints(hard_constraints, soft_constraints)
        pass2 = _score_once(relaxed.get("hard", {}), relaxed.get("soft", {}))
        if len(pass2["survivors"]) >= len(pass1["survivors"]):
            pass1 = pass2
            active_hard = relaxed.get("hard", {})
            active_soft = relaxed.get("soft", {})
            loop_mode = "relaxed"

    if exploration_mode and len(pass1["survivors"]) == 0 and active_hard.get("banned_tokens"):
        # 第二層放寬：保留 safety hard gate，但把 banned_tokens 改成 soft penalty
        hard2 = dict(active_hard)
        soft2 = dict(active_soft)
        soft2["soft_banned_tokens"] = list(hard2.get("banned_tokens") or [])
        hard2["banned_tokens"] = []
        pass3 = _score_once(hard2, soft2)
        # 對 pass3 套 soft ban penalty（不做硬淘汰）
        rescored = []
        for row in pass3["scored"]:
            p, reasons = _soft_penalty_for_banned_tokens(row, soft2.get("soft_banned_tokens") or [])
            if p > 0:
                row = dict(row)
                row["final_score"] = max(0.0, min(1.0, _safe_float(row.get("final_score"), 0.0) - p))
                sv = list(row.get("soft_violations") or [])
                sv.extend(reasons)
                row["soft_violations"] = sv
            rescored.append(row)
        pass3["scored"] = rescored
        pass3["survivors"] = [x for x in rescored if not x.get("rejected")]
        pass3["survivors"].sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        pass3["dropped"] = [x for x in rescored if x.get("rejected")]
        pass3["contested"] = [x for x in pass3["survivors"] if x.get("contested")]
        pass3["contested"].sort(key=lambda x: x.get("disagreement_std", 0.0), reverse=True)
        if len(pass3["survivors"]) >= len(pass1["survivors"]):
            pass1 = pass3
            active_hard = hard2
            active_soft = soft2
            loop_mode = "exploration_soft_ban"

    scored = pass1["scored"]
    survivors = pass1["survivors"]
    dropped = pass1["dropped"]
    contested_routes = pass1["contested"]
    hard_reject_reason_pool = pass1["hard_reasons"]
    relaxation_suggestions = _build_relaxation_suggestions(hard_reject_reason_pool, active_hard)

    _append_constraint_loop_log(
        {
            "target_smiles": target_smiles,
            "objective": objective,
            "mode": loop_mode,
            "constraint_parse_mode": mode,
            "hard_constraints": active_hard,
            "soft_constraints": active_soft,
            "n_survivors": len(survivors),
            "n_dropped": len(dropped),
            "n_contested": len(contested_routes),
            "relaxation_suggestions": relaxation_suggestions,
        }
    )

    lines = [
        "AskCOS 路線推薦（多 Agent 評估）",
        f"- 目標: {target_smiles}",
        f"- 後端: {backend}, objective: {objective}",
        f"- 搜尋統計: total_paths={stats.get('total_paths')}, total_chemicals={stats.get('total_chemicals')}, total_reactions={stats.get('total_reactions')}",
        f"- PubChem hazard: {'on' if enable_pubchem_hazard else 'off'} (max_unique_hazard_checks={max_unique_hazard_checks})",
        f"- Agent 權重: {weights}",
        f"- Vote thresholds: low={vote_low_threshold:.2f}, high={vote_high_threshold:.2f}, contested_std={contested_std_threshold:.2f}",
        f"- Constraints mode: parse={mode}, loop={loop_mode}, strict_safety_mode={strict_safety_mode}, exploration_mode={exploration_mode}",
        f"- Constraints(hard): {active_hard if active_hard else '{}'}",
        f"- Constraints(soft): {active_soft if active_soft else '{}'}",
        f"- 安全剔除: {len(dropped)} 條, 保留: {len(survivors)} 條",
        f"- 低共識(contested): {len(contested_routes)} 條",
        "",
        f"=== 推薦 Top {min(top_n, len(survivors))} ===",
    ]
    for i, r in enumerate(survivors[: max(1, top_n)], start=1):
        lines.extend(
            [
                f"[{i}] route_id={r.get('route_id')} final_score={r.get('final_score'):.4f}",
                f"  depth={r.get('depth')} precursor_cost={r.get('precursor_cost')} plausibility_product={r.get('plausibility_product')}",
                f"  cost_agent={r['agent_scores']['cost_agent']['score']:.4f} ({r['agent_scores']['cost_agent']['reason']})",
                f"  success_agent={r['agent_scores']['success_agent']['score']:.4f} ({r['agent_scores']['success_agent']['reason']})",
                f"  safety_agent={r['agent_scores']['safety_agent']['score']:.4f} ({r['agent_scores']['safety_agent']['reason']})",
                f"  supply_agent={r['agent_scores']['supply_agent']['score']:.4f} ({r['agent_scores']['supply_agent']['reason']})",
                f"  votes={r.get('critic_votes')} summary={r.get('vote_summary')} disagreement_std={r.get('disagreement_std',0.0):.3f} contested={r.get('contested')}",
                f"  constraint_satisfaction={r.get('constraint_eval',{}).get('satisfaction_rate',1.0):.3f} soft_penalty={r.get('soft_penalty',0.0):.3f}",
                f"  soft_violations={r.get('soft_violations')}",
                f"  leaves={', '.join((r.get('leaf_chemicals') or [])[:4])}",
            ]
        )

    if contested_routes:
        lines.append("")
        lines.append("=== 低共識路線（contested，前 10）===")
        for r in contested_routes[:10]:
            lines.append(
                f"- route_id={r.get('route_id')} disagreement_std={r.get('disagreement_std',0.0):.3f} "
                f"votes={r.get('vote_summary')} reasons={'; '.join(r.get('contested_reason') or ['n/a'])}"
            )

    if dropped:
        lines.append("")
        lines.append("=== 被剔除路線（前 10）===")
        for r in dropped[:10]:
            lines.append(
                f"- route_id={r.get('route_id')} reasons={'; '.join(r.get('reject_reasons') or ['unknown'])}"
            )

    if relaxation_suggestions:
        lines.append("")
        lines.append("=== Constraint 放寬建議（自動）===")
        for s in relaxation_suggestions:
            lines.append(f"- {s}")

    eval_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    lines.append("")
    lines.append(f"- eval_id: {eval_id}")

    text = "\n".join(lines)
    _append_jsonl(
        ROUTE_EVAL_LOG_PATH,
        {
            "ts": _utc_now(),
            "eval_id": eval_id,
            "target_smiles": target_smiles,
            "objective": objective,
            "backend": backend,
            "stats": {
                "total_paths": stats.get("total_paths"),
                "total_chemicals": stats.get("total_chemicals"),
                "total_reactions": stats.get("total_reactions"),
            },
            "constraints": {"hard": active_hard, "soft": active_soft},
            "constraint_loop_mode": loop_mode,
            "survivor_count": len(survivors),
            "dropped_count": len(dropped),
            "contested_count": len(contested_routes),
            "top_routes": [
                {
                    "route_id": r.get("route_id"),
                    "final_score": r.get("final_score"),
                    "rejected": r.get("rejected"),
                    "reject_reasons": r.get("reject_reasons"),
                    "vote_summary": r.get("vote_summary"),
                    "contested": r.get("contested"),
                    "contested_reason": r.get("contested_reason"),
                    "constraint_eval": r.get("constraint_eval"),
                }
                for r in (survivors[:top_n] + dropped[: min(5, len(dropped))])
            ],
            "relaxation_suggestions": relaxation_suggestions,
        },
    )
    if use_cache:
        cache.set(cache_key, text)
    return text


def run_askcos_route_recommendation_feedback(
    eval_id: str,
    route_id: int,
    decision: str,
    reason: str = "",
) -> str:
    if not eval_id:
        return "錯誤：未提供 eval_id。"
    d = (decision or "").strip().lower()
    if d not in {"accepted", "rejected", "needs_review"}:
        return "錯誤：decision 只接受 accepted/rejected/needs_review。"
    rid = int(route_id)
    eval_logs = _read_jsonl(ROUTE_EVAL_LOG_PATH, limit=5000)
    target_eval = None
    for item in reversed(eval_logs):
        if str(item.get("eval_id", "")) == eval_id:
            target_eval = item
            break
    if target_eval is None:
        return f"找不到 eval_id：{eval_id}"

    route_found = None
    for r in (target_eval.get("top_routes") or []):
        if int(r.get("route_id", -1)) == rid:
            route_found = r
            break

    rec = {
        "ts": _utc_now(),
        "eval_id": eval_id,
        "route_id": rid,
        "decision": d,
        "reason": reason or "",
        "target_smiles": target_eval.get("target_smiles", ""),
        "objective": target_eval.get("objective", ""),
        "route_snapshot": route_found or {},
    }
    _append_jsonl(ROUTE_FEEDBACK_LOG_PATH, rec)
    return f"已寫入回饋：eval_id={eval_id}, route_id={rid}, decision={d}"


def run_askcos_route_recommendation_recent_logs(limit: int = 5) -> str:
    logs = _read_jsonl(ROUTE_EVAL_LOG_PATH, limit=max(1, int(limit)))
    if not logs:
        return "目前沒有 route evaluation logs。"
    lines = ["最近 route evaluation logs："]
    for item in reversed(logs):
        lines.append(
            f"- eval_id={item.get('eval_id')} target={item.get('target_smiles')} "
            f"objective={item.get('objective')} survivors={item.get('survivor_count')} "
            f"dropped={item.get('dropped_count')} contested={item.get('contested_count')} "
            f"mode={item.get('constraint_loop_mode')}"
        )
    return "\n".join(lines)
