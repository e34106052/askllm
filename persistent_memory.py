"""
跨執行緒的輕量長期記憶（cmem）：摘要 + 最近若干輪對話文字 + 主題/反思。

環境變數：
  ASKLLM_MEMORY_DIR            記憶檔目錄（預設：<repo>/.askllm_memory）
  ASKLLM_MEMORY_DISABLE=1      關閉讀寫
  ASKLLM_MEMORY_MAX_TURNS      最多保留幾則訊息（預設 40）
  ASKLLM_MEMORY_AI_SUMMARY=1   啟用 AI 摘要壓縮（由上層決定何時呼叫）
  ASKLLM_MEMORY_RAW_KEEP       壓縮時保留最近幾則 raw 訊息（預設 12）
  ASKLLM_MEMORY_SUMMARY_CHUNK  每輪最多壓縮幾則舊訊息（預設 16）
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


MEMORY_DIR = os.environ.get(
    "ASKLLM_MEMORY_DIR",
    os.path.join(os.path.dirname(__file__), ".askllm_memory"),
)
STATE_FILE = "memory_state.json"
DISABLE = os.environ.get("ASKLLM_MEMORY_DISABLE", "0") == "1"
MAX_TURNS = int(os.environ.get("ASKLLM_MEMORY_MAX_TURNS", "40"))
SUMMARY_MODEL = os.environ.get("ASKLLM_MEMORY_SUMMARY_MODEL", "gemini-2.5-flash-lite")


def _state_path() -> str:
    os.makedirs(MEMORY_DIR, exist_ok=True)
    return os.path.join(MEMORY_DIR, STATE_FILE)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def env_ai_summary_default() -> bool:
    return os.environ.get("ASKLLM_MEMORY_AI_SUMMARY", "0") == "1"


def ai_summary_enabled() -> bool:
    return env_ai_summary_default()


def _raw_keep_count() -> int:
    return int(os.environ.get("ASKLLM_MEMORY_RAW_KEEP", "12"))


def _summary_chunk_count() -> int:
    return int(os.environ.get("ASKLLM_MEMORY_SUMMARY_CHUNK", "16"))


def default_state() -> Dict[str, Any]:
    return {
        "version": 1,
        "summary_zh": "",
        "turns": [],
        "ai_summary": env_ai_summary_default(),
        "current_topic": "",
        "topics": {},
        "meta_reflections": [],
    }


def _coerce_state(data: Dict[str, Any]) -> Dict[str, Any]:
    state = default_state()
    state.update(data or {})
    if not isinstance(state.get("turns"), list):
        state["turns"] = []
    if not isinstance(state.get("topics"), dict):
        state["topics"] = {}
    if not isinstance(state.get("meta_reflections"), list):
        state["meta_reflections"] = []
    if not isinstance(state.get("summary_zh"), str):
        state["summary_zh"] = ""
    if not isinstance(state.get("current_topic"), str):
        state["current_topic"] = ""
    return state


def load_state() -> Dict[str, Any]:
    if DISABLE:
        return default_state()

    path = _state_path()
    if not os.path.exists(path):
        return default_state()

    try:
        with open(path, "r", encoding="utf-8") as f:
            return _coerce_state(json.load(f))
    except Exception:
        return default_state()


def save_state(state: Dict[str, Any]) -> None:
    if DISABLE:
        return

    normalized = _coerce_state(state)
    with open(_state_path(), "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)


def append_turn(state: Dict[str, Any], role: str, text: str) -> Dict[str, Any]:
    if DISABLE:
        return state

    normalized = _coerce_state(state)
    normalized["turns"].append(
        {
            "role": role,
            "text": (text or "").strip(),
            "ts": _utc_now_iso(),
        }
    )
    if len(normalized["turns"]) > MAX_TURNS:
        normalized["turns"] = normalized["turns"][-MAX_TURNS:]
    return normalized


def set_summary(state: Dict[str, Any], summary_zh: str) -> Dict[str, Any]:
    normalized = _coerce_state(state)
    normalized["summary_zh"] = (summary_zh or "").strip()
    return normalized


def set_topic(state: Dict[str, Any], topic_name: str) -> Dict[str, Any]:
    normalized = _coerce_state(state)
    normalized["current_topic"] = (topic_name or "").strip()
    if normalized["current_topic"] and normalized["current_topic"] not in normalized["topics"]:
        normalized["topics"][normalized["current_topic"]] = ""
    return normalized


def set_topic_summary(state: Dict[str, Any], topic_name: str, summary_zh: str) -> Dict[str, Any]:
    normalized = _coerce_state(state)
    topic = (topic_name or "").strip()
    if topic:
        normalized["topics"][topic] = (summary_zh or "").strip()
        normalized["current_topic"] = topic
    return normalized


def get_current_topic_summary(state: Dict[str, Any]) -> str:
    normalized = _coerce_state(state)
    topic = normalized.get("current_topic", "").strip()
    if not topic:
        return ""
    return str(normalized.get("topics", {}).get(topic, "")).strip()


def add_reflection(state: Dict[str, Any], text: str, topic: str = "") -> Dict[str, Any]:
    normalized = _coerce_state(state)
    value = (text or "").strip()
    if not value:
        return normalized
    normalized["meta_reflections"].append(
        {
            "topic": (topic or normalized.get("current_topic", "")).strip(),
            "text": value,
            "ts": _utc_now_iso(),
        }
    )
    normalized["meta_reflections"] = normalized["meta_reflections"][-50:]
    return normalized


def clear_state() -> Dict[str, Any]:
    state = default_state()
    save_state(state)
    return state


def clear_turns_only(state: Dict[str, Any]) -> Dict[str, Any]:
    normalized = _coerce_state(state)
    normalized["turns"] = []
    return normalized


def clear_current_topic(state: Dict[str, Any]) -> Dict[str, Any]:
    normalized = _coerce_state(state)
    topic = normalized.get("current_topic", "").strip()
    if topic and topic in normalized["topics"]:
        normalized["topics"].pop(topic, None)
    normalized["current_topic"] = ""
    return normalized


def state_to_gemini_history(
    state: Dict[str, Any],
    max_messages: Optional[int] = None,
) -> List[Any]:
    from google.genai import types

    turns: List[Dict[str, Any]] = _coerce_state(state).get("turns") or []
    if max_messages is not None and max_messages > 0:
        turns = turns[-max_messages:]

    history = []
    for item in turns:
        role = item.get("role", "user")
        text = (item.get("text") or "").strip()
        if not text:
            continue
        history.append(types.Content(role=role, parts=[types.Part(text=text)]))
    return history


def format_summary_for_system(summary_zh: str) -> str:
    summary = (summary_zh or "").strip()
    if not summary:
        return ""
    return f"\n\n[持久記憶摘要（cmem）]\n{summary}\n"


def format_topic_summary_for_system(state: Dict[str, Any]) -> str:
    topic = (_coerce_state(state).get("current_topic") or "").strip()
    summary = get_current_topic_summary(state)
    if not topic and not summary:
        return ""
    return f"\n\n[主題記憶]\n目前主題: {topic or '未設定'}\n主題摘要: {summary or '（尚無）'}\n"


def compact_old_turns_for_summary(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    normalized = _coerce_state(state)
    turns = normalized.get("turns", [])
    raw_keep = _raw_keep_count()
    if len(turns) <= raw_keep:
        return []
    chunk = _summary_chunk_count()
    old_turns = turns[:-raw_keep]
    return old_turns[:chunk]


def apply_summary_compression(
    state: Dict[str, Any],
    new_summary_zh: str,
    consumed_count: int,
) -> Dict[str, Any]:
    normalized = _coerce_state(state)
    normalized["summary_zh"] = (new_summary_zh or normalized["summary_zh"]).strip()
    if consumed_count > 0:
        normalized["turns"] = normalized["turns"][consumed_count:]
    if len(normalized["turns"]) > MAX_TURNS:
        normalized["turns"] = normalized["turns"][-MAX_TURNS:]
    return normalized
