"""
相容舊版 advanced two-stage 介面。

研究版後期已將此路線移出主流程；本檔保留為 shim，
避免舊匯入或 compare 邏輯直接崩潰。
"""


def run_advanced_condition_prediction(reaction_smiles: str, timeout_sec: int = 45) -> str:
    if not reaction_smiles:
        return "錯誤：未提供反應 SMILES。"
    return (
        "advanced two-stage 條件預測已從 AskLLM 主流程移除，"
        "目前不再作為 active tool 使用。"
    )
