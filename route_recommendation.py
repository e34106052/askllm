import json
import os
import re
import subprocess
from typing import Any, Dict, List, Tuple

import cache_utils as cache
from askcos_tree_utils import parse_uds_paths, route_summary


TREE_SEARCH_CONTROLLER_URL = "http://127.0.0.1:9100/api/tree-search/controller/call-sync-without-token"
RISK_RULES_PATH = os.path.join(os.path.dirname(__file__), "risk_rules.json")


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


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

        high_hits = sorted(list(hcodes.intersection(high_h)))
        med_hits = sorted(list(hcodes.intersection(med_h)))
        low_hits = sorted(list(hcodes.intersection(low_h)))
        if not high_hits:
            high_hits.extend([kw for kw in high_kw if kw in merged][:3])
        if not med_hits:
            med_hits.extend([kw for kw in med_kw if kw in merged][:3])
        if not low_hits:
            low_hits.extend([kw for kw in low_kw if kw in merged][:3])

        if high_hits:
            out = {"severity": "high", "reason": "pubchem_high", "hits": high_hits}
        elif med_hits:
            out = {"severity": "medium", "reason": "pubchem_medium", "hits": med_hits}
        elif low_hits:
            out = {"severity": "low", "reason": "pubchem_low", "hits": low_hits}
        else:
            out = {"severity": "none", "reason": "pubchem_no_hazard_hit", "hits": []}
        cache.set(key, out)
        return out
    except Exception as e:
        out = {"severity": "unknown", "reason": f"pubchem_error:{str(e)[:120]}", "hits": []}
        cache.set(key, out)
        return out


def _safety_agent(
    routes: List[Dict[str, Any]],
    banned_tokens: List[str],
    enable_pubchem_hazard: bool = True,
    hazard_leaf_only: bool = True,
    max_unique_hazard_checks: int = 120,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, List[str]]]:
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

        if pubchem_enabled:
            for smi in (r.get("leaf_chemicals") or []):
                if ">>" in str(smi):
                    continue
                info = pubchem_map.get(smi) or {"severity": "unknown", "reason": "not_checked", "hits": []}
                sev = str(info.get("severity", "unknown")).lower()
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
    return scores, rejects


def _objective_weights(objective: str) -> Dict[str, float]:
    key = (objective or "balanced").strip().lower()
    if key in {"cheapest", "cost", "最便宜"}:
        return {"cost": 0.65, "success": 0.15, "safety": 0.10, "supply": 0.10}
    if key in {"highest_success", "success", "成功率最高"}:
        return {"cost": 0.10, "success": 0.60, "safety": 0.20, "supply": 0.10}
    if key in {"safest", "安全"}:
        return {"cost": 0.10, "success": 0.20, "safety": 0.55, "supply": 0.15}
    return {"cost": 0.30, "success": 0.30, "safety": 0.20, "supply": 0.20}


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
    cost = _cost_agent(routes)
    succ = _success_agent(routes)
    supply = _supply_agent(routes)
    safety, rejected = _safety_agent(
        routes,
        banned_tokens or [],
        enable_pubchem_hazard=enable_pubchem_hazard,
        hazard_leaf_only=hazard_leaf_only,
        max_unique_hazard_checks=max_unique_hazard_checks,
    )
    weights = _objective_weights(objective)

    scored: List[Dict[str, Any]] = []
    for r in routes:
        rid = int(r["route_id"])
        c = cost[rid]["score"]
        s = succ[rid]["score"]
        y = supply[rid]["score"]
        f = safety[rid]["score"]
        final = (
            weights["cost"] * c
            + weights["success"] * s
            + weights["safety"] * f
            + weights["supply"] * y
        )
        scored.append(
            {
                **r,
                "final_score": final,
                "agent_scores": {
                    "cost_agent": cost[rid],
                    "success_agent": succ[rid],
                    "safety_agent": safety[rid],
                    "supply_agent": supply[rid],
                },
                "rejected": rid in rejected,
                "reject_reasons": rejected.get(rid, []),
            }
        )

    survivors = [x for x in scored if not x["rejected"]]
    survivors.sort(key=lambda x: x["final_score"], reverse=True)
    dropped = [x for x in scored if x["rejected"]]

    lines = [
        "AskCOS 路線推薦（多 Agent 評估）",
        f"- 目標: {target_smiles}",
        f"- 後端: {backend}, objective: {objective}",
        f"- 搜尋統計: total_paths={stats.get('total_paths')}, total_chemicals={stats.get('total_chemicals')}, total_reactions={stats.get('total_reactions')}",
        f"- PubChem hazard: {'on' if enable_pubchem_hazard else 'off'} (max_unique_hazard_checks={max_unique_hazard_checks})",
        f"- Agent 權重: {weights}",
        f"- 安全剔除: {len(dropped)} 條, 保留: {len(survivors)} 條",
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
                f"  leaves={', '.join((r.get('leaf_chemicals') or [])[:4])}",
            ]
        )

    if dropped:
        lines.append("")
        lines.append("=== 被剔除路線（前 10）===")
        for r in dropped[:10]:
            lines.append(
                f"- route_id={r.get('route_id')} reasons={'; '.join(r.get('reject_reasons') or ['unknown'])}"
            )

    text = "\n".join(lines)
    if use_cache:
        cache.set(cache_key, text)
    return text
