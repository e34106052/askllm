import re
from typing import List


_SMILES_CHARSET = set("BCNOFPSIKHbrclonpsif@+-=#()[]\\/1234567890.%:,")


def is_tool_error(text: str) -> bool:
    value = (text or "").lower()
    error_markers = [
        "錯誤",
        "失敗",
        "exception",
        "traceback",
        "connection failed",
        "http 錯誤",
        "http error",
        "timeout",
        "超時",
        "找不到",
        "未知的工具",
        "未提供有效",
        "服務端錯誤",
    ]
    return any(marker in value for marker in error_markers)


def is_tool_empty(text: str) -> bool:
    value = (text or "").strip().lower()
    if not value:
        return True
    empty_markers = [
        "未找到",
        "未找到前體推薦",
        "未找到任何結果",
        "total_paths=0",
        "共找到 0",
        "no route",
        "empty",
    ]
    return any(marker in value for marker in empty_markers)


def extract_top_score(text: str) -> float:
    value = text or ""
    patterns = [
        r"得分[:：]\s*([0-9]*\.?[0-9]+)",
        r"score[:=]\s*([0-9]*\.?[0-9]+)",
        r"置信度分數[:：]\s*([0-9]*\.?[0-9]+)",
    ]
    best = 0.0
    for pattern in patterns:
        for match in re.finditer(pattern, value, flags=re.I):
            try:
                best = max(best, float(match.group(1)))
            except Exception:
                pass
    return best


def looks_like_smiles(text: str) -> bool:
    candidate = (text or "").strip()
    if len(candidate) < 2 or " " in candidate:
        return False
    if ">>" in candidate:
        left, _, right = candidate.partition(">>")
        return bool(left and right)
    if any(ch not in _SMILES_CHARSET for ch in candidate):
        return False
    letter_count = sum(ch.isalpha() for ch in candidate)
    return letter_count >= 1


def extract_smiles_candidate(text: str) -> str:
    value = (text or "").strip()
    if looks_like_smiles(value):
        return value

    for token in re.split(r"[\s,;，；]+", value):
        token = token.strip("()[]{}'\"")
        if looks_like_smiles(token):
            return token
    return ""


def extract_name_candidate(text: str) -> str:
    value = (text or "").strip()
    if not value:
        return ""

    patterns = [
        r"請(?:給我|幫我|做)?\s*(.+?)\s*(?:的|之)?(?:逆合成|正向預測|雜質|條件|結構圖|圖片)",
        r"(?:compound|molecule|name)[:：]\s*([^\n]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, value, flags=re.I)
        if match:
            return match.group(1).strip(" 。,，")

    if not looks_like_smiles(value) and len(value) <= 120:
        return value
    return ""


def recent_effective_evidence(tool_outputs: List[str], max_items: int = 3) -> List[str]:
    useful = []
    for item in reversed(tool_outputs):
        if not item:
            continue
        if is_tool_error(item):
            continue
        useful.append(item[:800])
        if len(useful) >= max_items:
            break
    return list(reversed(useful))
