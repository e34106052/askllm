import json
import os
import subprocess
from typing import Any, Dict, List

import cache_utils as cache


ASKCOS_RETRO_REAXYS_URL = os.environ.get(
    "ASKLLM_RETRO_REAXYS_URL",
    "http://127.0.0.1:9410/predictions/reaxys",
)
ASKCOS_RETRO_USPTO_FULL_URL = os.environ.get(
    "ASKLLM_RETRO_USPTO_FULL_URL",
    "http://127.0.0.1:9420/predictions/uspto_full",
)
ASKCOS_RETRO_PISTACHIO_URL = os.environ.get(
    "ASKLLM_RETRO_PISTACHIO_URL",
    "http://127.0.0.1:9420/predictions/pistachio_23Q3",
)
ASKCOS_RETRO_TEMPLATE_ENUM_URL = os.environ.get(
    "ASKLLM_RETRO_TEMPLATE_ENUM_URL",
    "http://127.0.0.1:9461/predictions/template_enumeration",
)


def _normalize_smiles_input(smiles_list: Any = None, target_smiles: str = "") -> List[str]:
    if isinstance(smiles_list, str) and smiles_list.strip():
        return [smiles_list.strip()]
    if isinstance(smiles_list, list):
        return [str(x).strip() for x in smiles_list if str(x).strip()]
    if target_smiles:
        return [target_smiles.strip()]
    return []


def _post_json(url: str, payload: Dict[str, Any], timeout_sec: int = 45) -> Dict[str, Any]:
    result = subprocess.run(
        [
            "curl",
            url,
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
        timeout=timeout_sec,
    )
    data = json.loads(result.stdout)
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            return data[0]
        return {"precursors": data}
    return {}


def _summarize_retro_results(engine_name: str, data: Dict[str, Any], max_routes: int) -> str:
    reactants = data.get("reactants") or data.get("precursors") or []
    scores = data.get("scores") or []
    templates = data.get("templates") or []

    if not reactants:
        return f"{engine_name} 逆合成調用成功，但未找到前體推薦。"

    limit = min(max_routes, len(reactants))
    summary_parts = []
    top_template_smarts = templates[0].get("reaction_smarts", "N/A") if templates else "N/A"

    for i in range(limit):
        precursor_smiles = reactants[i]
        score = float(scores[i]) if i < len(scores) else 0.0
        precursors_list = str(precursor_smiles).split(".")
        precursors_formatted = " / ".join(precursors_list)
        summary_parts.append(
            f"--- 第 {i+1} 名路徑 (得分: {score:.6f}) ---\n"
            f"  - 前體分子 ({len(precursors_list)} 個): {precursors_formatted}"
        )

    return (
        f"以下是 AskCOS 逆合成分析的結果。請以用戶可讀的中文總結以下路徑：\n"
        f"AskCOS 逆合成分析 ({engine_name}) 完成。共找到 {len(reactants)} 條路徑。\n"
        f"最高得分模板 (SMARTS): {top_template_smarts}\n"
        f"以下是您請求的前 {limit} 條路徑的詳細信息:\n"
        + "\n".join(summary_parts)
    )


def _run_retro_engine(
    *,
    engine_name: str,
    url: str,
    smiles_list: Any = None,
    target_smiles: str = "",
    max_routes: int = 3,
) -> str:
    normalized = _normalize_smiles_input(smiles_list=smiles_list, target_smiles=target_smiles)
    if not normalized:
        return "錯誤：未提供目標分子的 SMILES。"

    payload_data = {"smiles": normalized}
    cache_key = cache.build_key(
        "askcos:retro:v2",
        engine=engine_name,
        url=url,
        payload=payload_data,
        max_routes=max_routes,
    )
    cached = cache.get(cache_key)
    if isinstance(cached, str) and cached.strip():
        print(f"  -> 命中快取：AskCOS 逆合成 {engine_name}")
        return cached

    try:
        data = _post_json(url, payload_data, timeout_sec=60)
        if data.get("code") in [500, 503]:
            return f"AskCOS {engine_name} 服務端錯誤：{data.get('message', '預測失敗')}"
        final_summary = _summarize_retro_results(engine_name, data, max_routes=max_routes)
        cache.set(cache_key, final_summary)
        return final_summary
    except subprocess.CalledProcessError as e:
        error_output = e.stderr if e.stderr else f"Curl 退出代碼: {e.returncode}"
        return f"調用 AskCOS {engine_name} API 失敗，請檢查服務日誌。錯誤詳情:\n{error_output}"
    except Exception as e:
        return f"AskCOS {engine_name} 發生未知錯誤或 JSON 解析失敗: {e}"


def run_askcos_retrosynthesis(smiles_list: Any = None, target_smiles: str = "", max_routes: int = 3) -> str:
    return _run_retro_engine(
        engine_name="Reaxys",
        url=ASKCOS_RETRO_REAXYS_URL,
        smiles_list=smiles_list,
        target_smiles=target_smiles,
        max_routes=max_routes,
    )


def run_askcos_retrosynthesis_uspto_full(
    smiles_list: Any = None,
    target_smiles: str = "",
    max_routes: int = 3,
) -> str:
    return _run_retro_engine(
        engine_name="USPTO_FULL",
        url=ASKCOS_RETRO_USPTO_FULL_URL,
        smiles_list=smiles_list,
        target_smiles=target_smiles,
        max_routes=max_routes,
    )


def run_askcos_retrosynthesis_pistachio(
    smiles_list: Any = None,
    target_smiles: str = "",
    max_routes: int = 3,
) -> str:
    return _run_retro_engine(
        engine_name="PISTACHIO_23Q3",
        url=ASKCOS_RETRO_PISTACHIO_URL,
        smiles_list=smiles_list,
        target_smiles=target_smiles,
        max_routes=max_routes,
    )


def run_askcos_retrosynthesis_template_enum(
    smiles_list: Any = None,
    target_smiles: str = "",
    max_routes: int = 3,
) -> str:
    return _run_retro_engine(
        engine_name="TEMPLATE_ENUMERATION",
        url=ASKCOS_RETRO_TEMPLATE_ENUM_URL,
        smiles_list=smiles_list,
        target_smiles=target_smiles,
        max_routes=max_routes,
    )


def run_askcos_retrosynthesis_compare(
    smiles_list: Any = None,
    target_smiles: str = "",
    max_routes: int = 3,
) -> str:
    sections = [
        ("Reaxys", run_askcos_retrosynthesis(smiles_list=smiles_list, target_smiles=target_smiles, max_routes=max_routes)),
        ("USPTO_FULL", run_askcos_retrosynthesis_uspto_full(smiles_list=smiles_list, target_smiles=target_smiles, max_routes=max_routes)),
        ("PISTACHIO_23Q3", run_askcos_retrosynthesis_pistachio(smiles_list=smiles_list, target_smiles=target_smiles, max_routes=max_routes)),
        ("TEMPLATE_ENUMERATION", run_askcos_retrosynthesis_template_enum(smiles_list=smiles_list, target_smiles=target_smiles, max_routes=max_routes)),
    ]
    return "以下是多引擎逆合成比較結果，請比較各引擎的前體候選、得分與差異：\n\n" + "\n\n".join(
        [f"=== {name} ===\n{text}" for name, text in sections]
    )