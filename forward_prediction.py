import json
import os
import subprocess
from typing import Any, Dict, List

import cache_utils as cache


ASKCOS_FORWARD_WLDN5_URL = os.environ.get(
    "ASKLLM_FORWARD_WLDN5_URL",
    "http://127.0.0.1:9501/wldn5_predict",
)
ASKCOS_FORWARD_USPTO_STEREO_URL = os.environ.get(
    "ASKLLM_FORWARD_USPTO_STEREO_URL",
    "http://127.0.0.1:9510/predictions/uspto_stereo",
)
ASKCOS_FORWARD_GRAPH2SMILES_URL = os.environ.get(
    "ASKLLM_FORWARD_GRAPH2SMILES_URL",
    "http://127.0.0.1:9510/predictions/graph2smiles_pistachio",
)
ASKCOS_FORWARD_DEFAULT_MODEL = os.environ.get("ASKLLM_FORWARD_DEFAULT_MODEL", "pistachio")


def _normalize_reactants(reactants_smiles_list: Any) -> List[str]:
    if isinstance(reactants_smiles_list, str) and reactants_smiles_list.strip():
        return [reactants_smiles_list.strip()]
    if isinstance(reactants_smiles_list, list):
        return [str(x).strip() for x in reactants_smiles_list if str(x).strip()]
    return []


def _post_json(url: str, payload: Dict[str, Any], timeout_sec: int = 60) -> Any:
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
    return json.loads(result.stdout)


def _summarize_forward_results(engine_name: str, full_reactants_smiles: str, data: Any, top_k: int) -> str:
    results_obj = None
    if isinstance(data, list) and data and isinstance(data[0], dict):
        results_obj = data[0]
    elif isinstance(data, dict):
        results_obj = data

    products = []
    scores = []
    if results_obj:
        products = results_obj.get("products", []) or results_obj.get("predictions", [])
        scores = results_obj.get("scores", []) or results_obj.get("probabilities", [])

    if not products and isinstance(data, dict) and data.get("result"):
        result_list = data.get("result", [])
        products = [item.get("product", item.get("smiles", "N/A")) for item in result_list]
        scores = [float(item.get("score", 0.0) or 0.0) for item in result_list]

    if not products:
        return f"AskCOS 正向預測 ({engine_name}) 調用成功，但未找到產物候選。"

    limit = min(top_k, len(products))
    summary_parts = []
    for i in range(limit):
        product = products[i] if products[i] else "N/A"
        score = float(scores[i]) if i < len(scores) else 0.0
        summary_parts.append(
            f"--- 第 {i+1} 名預測 (得分: {score:.4f}) ---\n"
            f"  - 產物 SMILES: {product}\n"
            f"  - 完整反應 SMILES: {full_reactants_smiles}>>{product}"
        )

    return (
        f"AskCOS 正向預測 ({engine_name}) 完成。共找到 {len(products)} 種潛在產物。\n"
        f"以下是您請求的前 {limit} 名預測結果:\n"
        + "\n".join(summary_parts)
    )


def _run_forward_engine(
    *,
    engine_name: str,
    url: str,
    payload: Dict[str, Any],
    full_reactants_smiles: str,
    top_k: int,
) -> str:
    cache_key = cache.build_key(
        "askcos:forward:v2",
        engine=engine_name,
        url=url,
        payload=payload,
        top_k=top_k,
    )
    cached = cache.get(cache_key)
    if isinstance(cached, str) and cached.strip():
        print(f"  -> 命中快取：AskCOS 正向預測 {engine_name}")
        return cached

    try:
        data = _post_json(url, payload)
        final_summary = _summarize_forward_results(engine_name, full_reactants_smiles, data, top_k)
        cache.set(cache_key, final_summary)
        return final_summary
    except subprocess.CalledProcessError as e:
        return f"調用 AskCOS {engine_name} API 失敗，Curl 退出代碼: {e.returncode}。錯誤詳情:\n{e.stderr}"
    except subprocess.TimeoutExpired:
        return f"調用 AskCOS {engine_name} API 失敗，Curl 命令執行超時。"
    except FileNotFoundError:
        return "錯誤: 找不到 'curl' 命令。請確保它已安裝在系統 PATH 中。"
    except Exception as e:
        return f"AskCOS {engine_name} 發生未知錯誤: {e}"


def run_askcos_forward_prediction(reactants_smiles_list: Any, top_k: int = 3) -> str:
    normalized = _normalize_reactants(reactants_smiles_list)
    full_reactants_smiles = ".".join(normalized)
    if not full_reactants_smiles or full_reactants_smiles == ".":
        return "未提供有效的反應物 SMILES。"

    payload = {
        "reactants": full_reactants_smiles,
        "model_name": ASKCOS_FORWARD_DEFAULT_MODEL,
        "contexts": [],
    }
    return _run_forward_engine(
        engine_name="WLDN5_PISTACHIO",
        url=ASKCOS_FORWARD_WLDN5_URL,
        payload=payload,
        full_reactants_smiles=full_reactants_smiles,
        top_k=top_k,
    )


def run_askcos_forward_prediction_uspto_stereo(reactants_smiles_list: Any, top_k: int = 3) -> str:
    normalized = _normalize_reactants(reactants_smiles_list)
    full_reactants_smiles = ".".join(normalized)
    if not full_reactants_smiles or full_reactants_smiles == ".":
        return "未提供有效的反應物 SMILES。"

    payload = {"smiles": [full_reactants_smiles]}
    return _run_forward_engine(
        engine_name="USPTO_STEREO",
        url=ASKCOS_FORWARD_USPTO_STEREO_URL,
        payload=payload,
        full_reactants_smiles=full_reactants_smiles,
        top_k=top_k,
    )


def run_askcos_forward_prediction_graph2smiles(reactants_smiles_list: Any, top_k: int = 3) -> str:
    normalized = _normalize_reactants(reactants_smiles_list)
    full_reactants_smiles = ".".join(normalized)
    if not full_reactants_smiles or full_reactants_smiles == ".":
        return "未提供有效的反應物 SMILES。"

    payload = {"smiles": [full_reactants_smiles]}
    return _run_forward_engine(
        engine_name="GRAPH2SMILES_PISTACHIO",
        url=ASKCOS_FORWARD_GRAPH2SMILES_URL,
        payload=payload,
        full_reactants_smiles=full_reactants_smiles,
        top_k=top_k,
    )


def run_askcos_forward_prediction_wldn5(reactants_smiles_list: Any, top_k: int = 3) -> str:
    return run_askcos_forward_prediction(reactants_smiles_list=reactants_smiles_list, top_k=top_k)


def run_askcos_forward_prediction_compare(reactants_smiles_list: Any, top_k: int = 3) -> str:
    sections = [
        ("WLDN5_PISTACHIO", run_askcos_forward_prediction(reactants_smiles_list=reactants_smiles_list, top_k=top_k)),
        ("USPTO_STEREO", run_askcos_forward_prediction_uspto_stereo(reactants_smiles_list=reactants_smiles_list, top_k=top_k)),
        ("GRAPH2SMILES_PISTACHIO", run_askcos_forward_prediction_graph2smiles(reactants_smiles_list=reactants_smiles_list, top_k=top_k)),
    ]
    return "以下是多引擎正向預測比較結果，請比較產物候選、分數與差異：\n\n" + "\n\n".join(
        [f"=== {name} ===\n{text}" for name, text in sections]
    )