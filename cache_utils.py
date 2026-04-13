"""
簡單磁碟快取（避免重複打 PubChem / AskCOS / 工具摘要）。

環境變數：
  ASKLLM_CACHE_DIR          快取目錄（預設：<repo>/.askllm_cache）
  ASKLLM_CACHE_DISABLE=1    關閉快取
  ASKLLM_CACHE_TTL_SEC      預設 TTL（秒，預設 86400）
"""

import hashlib
import json
import os
import time
from typing import Any, Optional


CACHE_DIR = os.environ.get(
    "ASKLLM_CACHE_DIR",
    os.path.join(os.path.dirname(__file__), ".askllm_cache"),
)
DISABLE = os.environ.get("ASKLLM_CACHE_DISABLE", "0") == "1"
DEFAULT_TTL_SEC = int(os.environ.get("ASKLLM_CACHE_TTL_SEC", "86400"))


def _ensure_dir() -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return CACHE_DIR


def _normalize_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> str:
    return os.path.join(_ensure_dir(), f"{_normalize_key(key)}.json")


def _utc_now() -> int:
    return int(time.time())


def get(key: str, ttl_sec: Optional[int] = None) -> Any:
    if DISABLE:
        return None

    ttl = DEFAULT_TTL_SEC if ttl_sec is None else int(ttl_sec)
    path = _cache_path(key)
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None

    created_at = int(payload.get("created_at", 0))
    if ttl > 0 and created_at and (_utc_now() - created_at) > ttl:
        try:
            os.remove(path)
        except OSError:
            pass
        return None

    return payload.get("value")


def set(key: str, value: Any, ttl_sec: Optional[int] = None) -> bool:
    if DISABLE:
        return False

    path = _cache_path(key)
    payload = {
        "created_at": _utc_now(),
        "ttl_sec": DEFAULT_TTL_SEC if ttl_sec is None else int(ttl_sec),
        "value": value,
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def delete(key: str) -> bool:
    if DISABLE:
        return False
    path = _cache_path(key)
    if not os.path.exists(path):
        return False
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def clear() -> int:
    if DISABLE:
        return 0
    count = 0
    cache_dir = _ensure_dir()
    for filename in os.listdir(cache_dir):
        if not filename.endswith(".json"):
            continue
        try:
            os.remove(os.path.join(cache_dir, filename))
            count += 1
        except OSError:
            pass
    return count


def build_key(namespace: str, **kwargs: Any) -> str:
    normalized = json.dumps(kwargs, ensure_ascii=False, sort_keys=True, default=str)
    return f"{namespace}:{normalized}"
