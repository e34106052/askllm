import json
import subprocess
from typing import List, Optional

import cache_utils as cache


ASKCOS_QUARC_URL = "http://127.0.0.1:9921/api/v2/condition/QUARC"


def run_askcos_quarc_prediction(
    reaction_smiles: str,
    reagents: Optional[List[str]] = None,
    n_conditions: int = 5,
) -> str:
    if not reaction_smiles or ">>" not in reaction_smiles:
        return "錯誤：未提供有效的反應 SMILES (格式應為 RCTS>>PRD)。"

    payload_data = {
        "smiles": reaction_smiles,
        "reagents": reagents if reagents is not None else [],
        "n_conditions": n_conditions,
    }
    payload_str = json.dumps(payload_data, ensure_ascii=False)
    cache_key = cache.build_key("askcos:quarc:v1", url=ASKCOS_QUARC_URL, payload=payload_data)
    cached = cache.get(cache_key)
    if isinstance(cached, str) and cached.strip():
        print("  -> 命中快取：AskCOS QUARC 條件預測")
        return cached

    command = [
        "curl",
        ASKCOS_QUARC_URL,
        "--header",
        "Content-Type: application/json",
        "--request",
        "POST",
        "--data",
        payload_str,
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=90,
        )
        data = json.loads(result.stdout)
        if not isinstance(data, list) or not data:
            return "AskCOS QUARC 調用成功，但未找到推薦條件。"

        lines = [
            f"AskCOS QUARC 條件預測完成。共找到 {len(data)} 種條件候選。",
        ]
        for idx, cond in enumerate(data[: max(1, min(n_conditions, len(data)))], start=1):
            score = cond.get("score", 0.0)
            temp_k = cond.get("temperature", 0.0)
            agents = cond.get("agents", [])
            agent_text = []
            for agent in agents:
                smi = agent.get("smi_or_name", "N/A")
                amt = agent.get("amt", 0.0)
                agent_text.append(f"{smi} ({amt:.2f})")
            lines.append(
                f"--- 第 {idx} 名條件 (得分: {score:.4f}) ---\n"
                f"  - 溫度: {temp_k:.1f} K ({temp_k - 273.15:.1f} °C)\n"
                f"  - 條件組成: {'; '.join(agent_text) if agent_text else '無'}"
            )

        final_text = "\n".join(lines)
        cache.set(cache_key, final_text)
        return final_text
    except subprocess.CalledProcessError as e:
        error_output = e.stderr if e.stderr else f"Curl 退出代碼: {e.returncode}"
        return f"調用 AskCOS QUARC API 失敗。錯誤詳情:\n{error_output}"
    except Exception as e:
        return f"AskCOS QUARC 發生未知錯誤或 JSON 解析失敗: {e}"
