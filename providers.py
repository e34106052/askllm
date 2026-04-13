import json
import os
import subprocess
import urllib.error
import urllib.request
from typing import Any, Dict, List


class QuotaLimitError(RuntimeError):
    """API 配額或速率限制錯誤，應直接拋出給上層處理。"""


def format_quota_help_message(provider: str, model: str, detail: str = "") -> str:
    extra = f"\n原始錯誤：{detail}" if detail else ""
    return (
        f"{provider} 模型 `{model}` 目前遇到配額或速率限制，請稍後重試、切換模型，"
        f"或檢查 API key / 帳號配額設定。{extra}"
    )


def _read_response_body(resp: Any) -> Dict[str, Any]:
    body = resp.read().decode("utf-8")
    return json.loads(body)


def _chat_with_groq_via_curl(
    *,
    url: str,
    api_key: str,
    payload: Dict[str, Any],
    timeout_sec: int,
) -> Dict[str, Any]:
    cmd = [
        "curl",
        "-sS",
        "--max-time",
        str(timeout_sec),
        url,
        "-H",
        f"Authorization: Bearer {api_key}",
        "-H",
        "Content-Type: application/json",
        "-d",
        json.dumps(payload, ensure_ascii=False),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout_sec + 2,
    )
    return json.loads(result.stdout)


def chat_with_groq(
    *,
    messages: List[Dict[str, str]],
    model: str,
    timeout_sec: int = 60,
    temperature: float = 0.2,
    max_tokens: int = 1200,
) -> str:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("未設定 GROQ_API_KEY。")

    url = os.environ.get("ASKLLM_GROQ_URL", "https://api.groq.com/openai/v1/chat/completions")
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "curl/8.0",
        },
    )

    data = None
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = _read_response_body(resp)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        if e.code == 429:
            raise QuotaLimitError(format_quota_help_message("Groq", model, detail))
        # 某些環境下 urllib 會被風控規則擋下（例如 403/1010），curl 可正常通過。
        if e.code == 403 and "1010" in detail:
            try:
                data = _chat_with_groq_via_curl(
                    url=url,
                    api_key=api_key,
                    payload=payload,
                    timeout_sec=timeout_sec,
                )
            except subprocess.CalledProcessError as ce:
                raise RuntimeError(
                    f"Groq HTTP Error 403: {detail}; curl fallback failed: {ce.stderr or ce}"
                )
            except Exception as ce:
                raise RuntimeError(f"Groq HTTP Error 403: {detail}; curl fallback failed: {ce}")
        else:
            raise RuntimeError(f"Groq HTTP Error {e.code}: {detail}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Groq 連線失敗：{e}")

    if data is None:
        raise RuntimeError("Groq 返回空資料。")
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("Groq 返回空 choices。")
    return choices[0].get("message", {}).get("content", "") or ""
