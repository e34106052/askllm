import json
import subprocess
from typing import Any

import cache_utils as cache


ASKCOS_MULTISTEP_URL = "http://127.0.0.1:7000/get_buyable_paths"


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _format_pathway_item(pathway: Any, idx: int) -> str:
    if isinstance(pathway, dict):
        score = _safe_float(
            pathway.get("score", pathway.get("plausibility", pathway.get("path_score", 0.0)))
        )
        chemicals = pathway.get("chemicals", [])
        reactions = pathway.get("reactions", [])
        return (
            f"--- 路徑 {idx} (得分: {score:.4f}) ---\n"
            f"  - 化學節點數: {len(chemicals)}\n"
            f"  - 反應步數: {len(reactions)}\n"
            f"  - 關鍵資訊: {json.dumps(pathway, ensure_ascii=False)[:600]}"
        )
    return f"--- 路徑 {idx} ---\n  - 原始內容: {str(pathway)[:600]}"


def run_askcos_multistep_retrosynthesis(
    target_smiles: str,
    max_depth: int = 6,
    max_paths: int = 5,
    expansion_time: int = 45,
    max_branching: int = 25,
    retro_model_name: str = "reaxys",
    max_num_templates: int = 200,
    top_k: int = 20,
    threshold: float = 0.15,
    sorting_metric: str = "plausibility",
    use_cache: bool = True,
) -> str:
    if not target_smiles:
        return "錯誤：未提供目標分子的 SMILES。"

    payload = {
        "smiles": target_smiles,
        "build_tree_options": {
            "max_depth": max_depth,
            "expansion_time": expansion_time,
            "max_branching": max_branching,
            "retro_backend_options": {
                "retro_model_name": retro_model_name,
                "max_num_templates": max_num_templates,
                "top_k": top_k,
                "threshold": threshold,
            },
        },
        "enumerate_paths_options": {
            "max_paths": max_paths,
            "sorting_metric": sorting_metric,
        },
    }
    cache_key = cache.build_key("askcos:multistep:v1", url=ASKCOS_MULTISTEP_URL, payload=payload)
    if use_cache:
        cached = cache.get(cache_key)
        if isinstance(cached, str) and cached.strip():
            print("  -> 命中快取：AskCOS 多步逆合成")
            return cached

    try:
        result = subprocess.run(
            [
                "curl",
                ASKCOS_MULTISTEP_URL,
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
            timeout=max(60, expansion_time + 20),
        )
        data = json.loads(result.stdout)

        pathways = (
            data.get("pathways")
            or data.get("paths")
            or data.get("results", {}).get("pathways")
            or []
        )
        total_paths = int(data.get("total_paths", len(pathways) if isinstance(pathways, list) else 0))
        total_chemicals = int(data.get("total_chemicals", 0))
        total_reactions = int(data.get("total_reactions", 0))

        if not pathways:
            final_text = (
                "AskCOS 多步逆合成（MCTS）未找到可回推到可購買起始物的完整路徑。\n"
                f"搜尋統計：total_paths={total_paths}，total_chemicals={total_chemicals}，total_reactions={total_reactions}"
            )
            if use_cache:
                cache.set(cache_key, final_text)
            return final_text

        lines = [
            f"AskCOS 多步逆合成（MCTS）完成。共找到 {total_paths} 條路徑。",
            f"搜尋統計：total_chemicals={total_chemicals}，total_reactions={total_reactions}",
        ]
        for idx, pathway in enumerate(pathways[: max_paths], start=1):
            lines.append(_format_pathway_item(pathway, idx))

        final_text = "\n".join(lines)
        if use_cache:
            cache.set(cache_key, final_text)
        return final_text
    except subprocess.CalledProcessError as e:
        error_output = e.stderr if e.stderr else f"Curl 退出代碼: {e.returncode}"
        return f"調用 AskCOS 多步逆合成 API 失敗。錯誤詳情:\n{error_output}"
    except Exception as e:
        return f"多步逆合成發生未知錯誤或 JSON 解析失敗: {e}"
