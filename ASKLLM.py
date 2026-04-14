import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Union

from google import genai
from google.genai import types

import cache_utils as cache
import persistent_memory as pmem
from condition_prediction import (
    run_askcos_condition_prediction,
    run_askcos_condition_prediction_compare,
)
from context_quarc import run_askcos_quarc_prediction
from forward_prediction import (
    run_askcos_forward_prediction,
    run_askcos_forward_prediction_compare,
    run_askcos_forward_prediction_graph2smiles,
    run_askcos_forward_prediction_uspto_stereo,
    run_askcos_forward_prediction_wldn5,
)
from impurity_prediction import run_askcos_impurity_prediction
from multistep_retrosynthesis import (
    run_askcos_multistep_retrosynthesis,
    run_askcos_multistep_retrosynthesis_compare,
    run_askcos_multistep_retrosynthesis_retro_star,
)
from orchestrator import run_groq_turn
from policies import (
    extract_name_candidate,
    extract_smiles_candidate,
    extract_top_score,
    is_tool_empty,
    is_tool_error,
    looks_like_smiles,
    recent_effective_evidence,
)
from providers import QuotaLimitError, chat_with_groq, format_quota_help_message
from route_recommendation import run_askcos_route_recommendation
from retrosynthesis import (
    run_askcos_retrosynthesis,
    run_askcos_retrosynthesis_compare,
    run_askcos_retrosynthesis_pistachio,
    run_askcos_retrosynthesis_template_enum,
    run_askcos_retrosynthesis_uspto_full,
)
from smiles_resolver import resolve_smiles_from_name

try:
    from config import GEMINI_API_KEY
except ImportError:
    GEMINI_API_KEY = ""


PRIMARY_MODEL = os.environ.get("ASKLLM_PRIMARY_MODEL", "gemini-2.5-flash")
BACKUP_MODEL = os.environ.get("ASKLLM_BACKUP_MODEL", "gemini-2.5-flash-lite")
AUX_MODEL = os.environ.get("ASKLLM_AUX_MODEL", "gemini-2.5-flash-lite")
PLANNER_MODEL = os.environ.get("ASKLLM_PLANNER_MODEL", "gemini-2.5-flash")
CRITIC_MODEL = os.environ.get("ASKLLM_CRITIC_MODEL", "gemini-2.5-flash-lite")
SKILL_ROUTER_MODEL = os.environ.get("ASKLLM_SKILL_ROUTER_MODEL", "gemini-2.5-flash-lite")

GROQ_DECISION_MODEL = os.environ.get("ASKLLM_GROQ_DECISION_MODEL", "llama-3.3-70b-versatile")
GROQ_AUX_MODEL = os.environ.get("ASKLLM_GROQ_AUX_MODEL", "llama-3.1-8b-instant")
GROQ_PLANNER_MODEL = os.environ.get("ASKLLM_GROQ_PLANNER_MODEL", "llama-3.1-8b-instant")
GROQ_CRITIC_MODEL = os.environ.get("ASKLLM_GROQ_CRITIC_MODEL", "llama-3.1-8b-instant")
GROQ_SKILL_ROUTER_MODEL = os.environ.get("ASKLLM_GROQ_SKILL_ROUTER_MODEL", "llama-3.1-8b-instant")
GROQ_SIMPLE_MODEL = os.environ.get("ASKLLM_GROQ_SIMPLE_MODEL", "llama-3.1-8b-instant")
# 複雜任務預設改走 Gemini（避免 70b rate limit），可用環境變數覆蓋。
GROQ_COMPLEX_MODEL = os.environ.get("ASKLLM_GROQ_COMPLEX_MODEL", PRIMARY_MODEL)

DECISION_PROVIDER = os.environ.get("ASKLLM_DECISION_PROVIDER", "gemini").lower()
PLANNER_PROVIDER = os.environ.get("ASKLLM_PLANNER_PROVIDER", DECISION_PROVIDER).lower()
AUX_PROVIDER = os.environ.get("ASKLLM_AUX_PROVIDER", DECISION_PROVIDER).lower()
CRITIC_PROVIDER = os.environ.get("ASKLLM_CRITIC_PROVIDER", AUX_PROVIDER).lower()
SKILL_ROUTER_PROVIDER = os.environ.get("ASKLLM_SKILL_ROUTER_PROVIDER", AUX_PROVIDER).lower()

PLANNER_TIMEOUT_SEC = int(os.environ.get("ASKLLM_PLANNER_TIMEOUT_SEC", "60"))
TOOL_RESULT_MAX_CHARS = int(os.environ.get("ASKLLM_TOOL_RESULT_MAX_CHARS", "2200"))

ENABLE_AI_SKILL_ROUTER = os.environ.get("ASKLLM_ENABLE_AI_SKILL_ROUTER", "1") == "1"
ENABLE_TOOL_OUTPUT_SUMMARY = os.environ.get("ASKLLM_ENABLE_TOOL_OUTPUT_SUMMARY", "1") == "1"
ENABLE_CRITIC = os.environ.get("ASKLLM_ENABLE_CRITIC", "1") == "1"
ENABLE_META_REFLECTION = os.environ.get("ASKLLM_ENABLE_META_REFLECTION", "1") == "1"
ENABLE_ADAPTIVE_POLICY = os.environ.get("ASKLLM_ENABLE_ADAPTIVE_POLICY", "1") == "1"

SKILLS_DIR = os.path.join(os.path.dirname(__file__), "skills")
SKILL_FALLBACK_FILE = os.path.join(SKILLS_DIR, "00-core.md")
BASE_SKILL_FILES = [
    "00-core.md",
    "01-language.md",
    "08-cmem.md",
    "07-completion.md",
]
CONDITIONAL_SKILL_RULES = [
    ("02-ambiguity.md", ["co", "no", " p ", "smiles", "分子式", "結構式", "歧義", "縮寫"]),
    ("03-name-resolution.md", ["名稱", "name", "翻譯", "smiles", "化合物", "compound", "分子"]),
    ("04-condition-priority.md", ["條件", "condition", "yield", "產率", "buchwald", "grignard", "chan-lam"]),
    ("05-task-routing.md", ["逆合成", "retrosynthesis", "forward", "正向", "雜質", "impurity", "路徑", "mcts"]),
    ("06-formatting.md", ["輸出", "格式", "temperature", "probability", "score", "top 5", "攝氏", "表格"]),
]

MEMORY_DIR = pmem.MEMORY_DIR
EVIDENCE_LOG_PATH = os.path.join(MEMORY_DIR, "evidence_logs.jsonl")
TOOL_TRACE_PATH = os.path.join(MEMORY_DIR, "tool_trace_current_session.json")

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_memory_dir() -> None:
    os.makedirs(MEMORY_DIR, exist_ok=True)


def _read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def _read_json_file(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _write_json_file(path: str, data: Any) -> None:
    _ensure_memory_dir()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_tool_trace(record: Dict[str, Any]) -> None:
    _ensure_memory_dir()
    data = _read_json_file(
        TOOL_TRACE_PATH,
        {"session_started_at": utc_now_iso(), "items": []},
    )
    items = data.get("items", [])
    items.append(record)
    data["items"] = items
    _write_json_file(TOOL_TRACE_PATH, data)


def write_evidence_log(record: Dict[str, Any]) -> None:
    _ensure_memory_dir()
    with open(EVIDENCE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _extract_json_block(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def _is_simple_task(user_prompt: str) -> bool:
    q = (user_prompt or "").strip()
    ql = q.lower()
    if not q:
        return True
    complex_markers = [
        "比較",
        "compare",
        "對比",
        "多工具",
        "multi tool",
        "mcts",
        "多步",
        "路徑",
        "策略",
        "a/b/c",
        "replan",
        "評估",
    ]
    if any(token in ql for token in complex_markers):
        return False
    return len(q) <= 80


def _pick_groq_model_for_task(
    *,
    user_prompt: str,
    task_type: str,
    adaptive_plan: Dict[str, Any] = None,
) -> str:
    is_simple = _is_simple_task(user_prompt)
    if task_type == "decision":
        if adaptive_plan:
            compare_allowed = adaptive_plan.get("compare_allowed")
            multistep_requested = adaptive_plan.get("multistep_requested")
            if compare_allowed or multistep_requested:
                return GROQ_COMPLEX_MODEL
        return GROQ_SIMPLE_MODEL if is_simple else GROQ_COMPLEX_MODEL
    if task_type == "planner":
        return GROQ_SIMPLE_MODEL if is_simple else GROQ_PLANNER_MODEL
    if task_type == "critic":
        return GROQ_SIMPLE_MODEL if is_simple else GROQ_CRITIC_MODEL
    return GROQ_SIMPLE_MODEL


def _chat_with_gemini_text(prompt: str, model: str, system_instruction: str = "", timeout_sec: int = 60) -> str:
    if client is None:
        raise RuntimeError("未設定 GEMINI_API_KEY，無法使用 Gemini。")
    response = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
        config=types.GenerateContentConfig(system_instruction=system_instruction),
    )
    return response.text or ""


def _generate_text(
    *,
    prompt: str,
    model: str,
    provider: str,
    system_instruction: str = "",
    timeout_sec: int = 60,
) -> str:
    original_provider = (provider or "gemini").lower()
    provider = original_provider
    # 允許在 groq 路徑下動態切換到 Gemini 模型（例如複雜任務）。
    if provider == "groq" and str(model).startswith("gemini"):
        provider = "gemini"
    if provider == "groq":
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        return chat_with_groq(messages=messages, model=model, timeout_sec=timeout_sec)

    try:
        gemini_text = _chat_with_gemini_text(
            prompt=prompt,
            model=model,
            system_instruction=system_instruction,
            timeout_sec=timeout_sec,
        )
        return gemini_text
    except Exception as primary_error:
        if model != BACKUP_MODEL:
            try:
                return _chat_with_gemini_text(
                    prompt=prompt,
                    model=BACKUP_MODEL,
                    system_instruction=system_instruction,
                    timeout_sec=timeout_sec,
                )
            except Exception as backup_error:
                gemini_error_text = (
                    "主備模型均調用失敗，請檢查 API Key 或配額。\n"
                    f"主模型錯誤: {primary_error}\n"
                    f"備用模型錯誤: {backup_error}"
                )
                # 若原本是 Groq 路徑但複雜任務暫切 Gemini，Gemini 失敗時自動降回 Groq 8b。
                if original_provider == "groq":
                    try:
                        messages = []
                        if system_instruction:
                            messages.append({"role": "system", "content": system_instruction})
                        messages.append({"role": "user", "content": prompt})
                        return chat_with_groq(
                            messages=messages,
                            model=GROQ_SIMPLE_MODEL,
                            timeout_sec=timeout_sec,
                        )
                    except Exception:
                        return gemini_error_text
                return gemini_error_text
        raise


def _route_skills_with_ai(user_prompt: str) -> List[str]:
    if not ENABLE_AI_SKILL_ROUTER:
        return []
    available_files = ["02-ambiguity.md", "03-name-resolution.md", "04-condition-priority.md", "05-task-routing.md", "06-formatting.md"]
    provider = SKILL_ROUTER_PROVIDER
    model = (
        _pick_groq_model_for_task(user_prompt=user_prompt, task_type="planner")
        if provider == "groq"
        else SKILL_ROUTER_MODEL
    )
    prompt = (
        "請根據使用者問題，從以下 skill 檔案中挑選最相關者，"
        "只輸出 JSON：{\"files\":[...]}。\n"
        f"available={available_files}\n"
        f"user_prompt={user_prompt}"
    )
    try:
        text = _generate_text(prompt=prompt, model=model, provider=provider, timeout_sec=30)
        data = _extract_json_block(text)
        files = data.get("files", [])
        return [x for x in files if x in available_files]
    except Exception:
        return []


def _select_skill_files(user_prompt: str) -> List[str]:
    normalized = f" {str(user_prompt).lower()} "
    selected = list(BASE_SKILL_FILES)
    selected.extend(_route_skills_with_ai(user_prompt))

    for filename, keywords in CONDITIONAL_SKILL_RULES:
        if any(keyword in normalized for keyword in keywords):
            selected.append(filename)

    deduped = []
    seen = set()
    for filename in selected:
        if filename not in seen:
            deduped.append(filename)
            seen.add(filename)
    return deduped


def load_system_instruction_from_skill(user_prompt: str, state: Dict[str, Any]) -> str:
    selected_files = _select_skill_files(user_prompt)
    fragments = []
    for filename in selected_files:
        content = _read_text_file(os.path.join(SKILLS_DIR, filename))
        if content:
            fragments.append(f"[SKILL:{filename}]\n{content}")

    if not fragments:
        fallback = _read_text_file(SKILL_FALLBACK_FILE)
        if fallback:
            fragments.append(fallback)

    if not fragments:
        fragments.append(
            "You are a professional chemistry assistant using the provided tools. "
            "Always respond in Traditional Chinese (繁體中文)."
        )

    fragments.append(pmem.format_summary_for_system(state.get("summary_zh", "")))
    fragments.append(pmem.format_topic_summary_for_system(state))
    return "\n\n".join([x for x in fragments if x.strip()])


def _explicit_compare_request(user_prompt: str) -> bool:
    lower = (user_prompt or "").lower()
    return any(token in lower for token in ["比較", "compare", "對比", "multi tool", "多工具", "ensemble", "多引擎"])


def _explicit_multistep_request(user_prompt: str) -> bool:
    lower = (user_prompt or "").lower()
    return any(token in lower for token in ["多步", "mcts", "路徑規劃", "buyable", "起始物"])


def _build_heuristic_plan(user_prompt: str, available_tool_names: List[str]) -> Dict[str, Any]:
    lower = (user_prompt or "").lower()
    compare_allowed = _explicit_compare_request(user_prompt)
    multistep = _explicit_multistep_request(user_prompt)

    candidates = ["resolve_smiles_from_name"]
    if any(k in lower for k in ["逆合成", "retrosynthesis", "合成路徑"]):
        candidates.extend(["run_askcos_retrosynthesis"])
        if multistep:
            candidates.extend(
                [
                    "run_askcos_multistep_retrosynthesis",
                    "run_askcos_multistep_retrosynthesis_retro_star",
                ]
            )
        if compare_allowed:
            candidates.append("run_askcos_retrosynthesis_compare")
            if multistep:
                candidates.append("run_askcos_multistep_retrosynthesis_compare")
    if any(k in lower for k in ["正向", "forward", "產物", "product"]):
        candidates.append("run_askcos_forward_prediction")
        if compare_allowed:
            candidates.append("run_askcos_forward_prediction_compare")
    if any(k in lower for k in ["條件", "condition", "yield", "產率"]):
        candidates.extend(["run_askcos_condition_prediction", "run_askcos_quarc_prediction"])
        if compare_allowed:
            candidates.append("run_askcos_condition_prediction_compare")
    if any(k in lower for k in ["雜質", "impurity", "副產物"]):
        candidates.append("run_askcos_impurity_prediction")
    if any(k in lower for k in ["推薦路線", "路線推薦", "最便宜", "成功率最高", "highest_success", "cheapest"]):
        candidates.append("run_askcos_route_recommendation")
    if len(candidates) == 1:
        candidates.extend(
            [
                "run_askcos_retrosynthesis",
                "run_askcos_forward_prediction",
                "run_askcos_condition_prediction",
                "run_askcos_impurity_prediction",
            ]
        )

    filtered = []
    for name in candidates:
        if name not in available_tool_names:
            continue
        if not compare_allowed and name.endswith("_compare"):
            continue
        if not multistep and name.startswith("run_askcos_multistep_retrosynthesis"):
            continue
        filtered.append(name)

    deduped = []
    seen = set()
    for name in filtered:
        if name not in seen:
            deduped.append(name)
            seen.add(name)

    return {
        "intent": "heuristic",
        "tool_candidates": deduped,
        "compare_allowed": compare_allowed,
        "multistep_requested": multistep,
        "max_tool_calls": 3 if compare_allowed or multistep else 2,
        "reasoning": "heuristic fallback",
    }


def _build_adaptive_plan(user_prompt: str, tools_to_use: list) -> Dict[str, Any]:
    available_tool_names = [getattr(t, "__name__", str(t)) for t in tools_to_use]
    heuristic = _build_heuristic_plan(user_prompt, available_tool_names)
    if not ENABLE_ADAPTIVE_POLICY:
        heuristic["planner_enabled"] = False
        return heuristic

    planner_provider = PLANNER_PROVIDER
    planner_model = (
        _pick_groq_model_for_task(user_prompt=user_prompt, task_type="planner")
        if planner_provider == "groq"
        else PLANNER_MODEL
    )
    prompt = (
        "你是 AskLLM 的 adaptive planner。請輸出 JSON，欄位包含："
        "intent, tool_candidates, compare_allowed, max_tool_calls, reasoning。"
        "要求：預設不要啟用 compare，除非使用者明確要求比較；"
        "若提到多步/MCTS/路徑規劃，優先保留多步逆合成工具。\n"
        f"available_tools={available_tool_names}\n"
        f"user_prompt={user_prompt}\n"
        f"heuristic_baseline={json.dumps(heuristic, ensure_ascii=False)}"
    )
    try:
        text = _generate_text(
            prompt=prompt,
            model=planner_model,
            provider=planner_provider,
            timeout_sec=PLANNER_TIMEOUT_SEC,
        )
        data = _extract_json_block(text)
        if not data:
            raise RuntimeError("planner 未返回合法 JSON")
        tool_candidates = [x for x in data.get("tool_candidates", []) if x in available_tool_names]
        if not data.get("compare_allowed"):
            tool_candidates = [x for x in tool_candidates if not x.endswith("_compare")]
        if not data.get("tool_candidates"):
            data["tool_candidates"] = heuristic["tool_candidates"]
        else:
            data["tool_candidates"] = tool_candidates or heuristic["tool_candidates"]
        data["compare_allowed"] = bool(data.get("compare_allowed", heuristic["compare_allowed"]))
        data["max_tool_calls"] = int(data.get("max_tool_calls", heuristic["max_tool_calls"]))
        data["planner_enabled"] = True
        return data
    except Exception:
        heuristic["planner_enabled"] = True
        return heuristic


def _filter_tools_for_turn(user_prompt: str, tools_to_use: list, adaptive_plan: Dict[str, Any]) -> list:
    requested = set(adaptive_plan.get("tool_candidates", []))
    compare_allowed = bool(adaptive_plan.get("compare_allowed"))
    multistep_requested = bool(adaptive_plan.get("multistep_requested") or _explicit_multistep_request(user_prompt))

    selected = []
    for tool in tools_to_use:
        name = getattr(tool, "__name__", str(tool))
        if requested and name not in requested:
            continue
        if not compare_allowed and name.endswith("_compare"):
            continue
        if not multistep_requested and name == "run_askcos_multistep_retrosynthesis":
            continue
        selected.append(tool)

    if not selected:
        selected = [
            tool for tool in tools_to_use
            if getattr(tool, "__name__", str(tool)) in adaptive_plan.get("tool_candidates", [])
        ] or list(tools_to_use)

    if not compare_allowed:
        selected = [tool for tool in selected if not getattr(tool, "__name__", str(tool)).endswith("_compare")]
    if not multistep_requested:
        selected = [
            tool
            for tool in selected
            if not getattr(tool, "__name__", str(tool)).startswith("run_askcos_multistep_retrosynthesis")
        ]
    return selected


def _extract_smiles_from_resolver_output(text: str) -> str:
    match = re.search(r"SMILES 字符串是:\s*\*?\*?([^\n*]+)", text or "")
    if match:
        candidate = match.group(1).strip()
        if looks_like_smiles(candidate):
            return candidate
    return extract_smiles_candidate(text)


def _default_args_for_tool(tool_name: str, user_prompt: str, resolved_smiles: str = "") -> Dict[str, Any]:
    reaction_smiles = resolved_smiles if ">>" in resolved_smiles else extract_smiles_candidate(user_prompt)
    name_candidate = extract_name_candidate(user_prompt)
    molecule_smiles = resolved_smiles if resolved_smiles and ">>" not in resolved_smiles else extract_smiles_candidate(user_prompt)
    if tool_name == "resolve_smiles_from_name":
        return {"compound_name": name_candidate or user_prompt}
    if tool_name.startswith("run_askcos_retrosynthesis"):
        return {"smiles_list": [molecule_smiles] if molecule_smiles else [], "max_routes": 3}
    if tool_name.startswith("run_askcos_multistep_retrosynthesis"):
        return {"target_smiles": molecule_smiles, "max_depth": 6, "max_paths": 5, "expansion_time": 45}
    if tool_name == "run_askcos_route_recommendation":
        objective = "balanced"
        lower = (user_prompt or "").lower()
        if "最便宜" in user_prompt or "cheapest" in lower:
            objective = "cheapest"
        elif "成功率最高" in user_prompt or "highest_success" in lower:
            objective = "highest_success"
        elif "最安全" in user_prompt or "safest" in lower:
            objective = "safest"
        return {
            "target_smiles": molecule_smiles,
            "objective": objective,
            "backend": "mcts",
            "max_depth": 5,
            "max_paths": 120,
            "expansion_time": 180,
            "top_n": 10,
            "enable_pubchem_hazard": True,
            "max_unique_hazard_checks": 120,
        }
    if tool_name.startswith("run_askcos_forward_prediction"):
        reactants = reaction_smiles.split(".") if reaction_smiles and ">>" not in reaction_smiles else ([molecule_smiles] if molecule_smiles else [])
        return {"reactants_smiles_list": reactants, "top_k": 3}
    if tool_name in {"run_askcos_condition_prediction", "run_askcos_condition_prediction_compare", "run_askcos_quarc_prediction"}:
        return {"reaction_smiles": reaction_smiles, "n_conditions": 5}
    if tool_name == "run_askcos_impurity_prediction":
        return {"reactants_smiles": reaction_smiles or molecule_smiles}
    return {}


def _sanitize_tool_args(tool_name: str, args: dict) -> dict:
    cleaned = dict(args or {})
    for junk_key in ("path1", "path2", "path3", "engine", "provider"):
        cleaned.pop(junk_key, None)

    if tool_name.startswith("run_askcos_retrosynthesis"):
        if "smiles" in cleaned and "smiles_list" not in cleaned:
            cleaned["smiles_list"] = cleaned["smiles"]
        if "target_smiles" in cleaned and "smiles_list" not in cleaned:
            cleaned["smiles_list"] = cleaned["target_smiles"]
        cleaned.pop("smiles", None)
        cleaned.pop("target_smiles", None)
        if "num_paths" in cleaned and "max_routes" not in cleaned:
            cleaned["max_routes"] = cleaned.pop("num_paths")
        if "max_paths" in cleaned and "max_routes" not in cleaned:
            cleaned["max_routes"] = cleaned.pop("max_paths")
        if isinstance(cleaned.get("smiles_list"), str):
            cleaned["smiles_list"] = [cleaned["smiles_list"]]

    if tool_name.startswith("run_askcos_forward_prediction"):
        if "reactants" in cleaned and "reactants_smiles_list" not in cleaned:
            reactants = cleaned.pop("reactants")
            cleaned["reactants_smiles_list"] = reactants.split(".") if isinstance(reactants, str) else reactants
        if "smiles" in cleaned and "reactants_smiles_list" not in cleaned:
            reactants = cleaned.pop("smiles")
            cleaned["reactants_smiles_list"] = reactants.split(".") if isinstance(reactants, str) else reactants
        if "top_n" in cleaned and "top_k" not in cleaned:
            cleaned["top_k"] = cleaned.pop("top_n")

    if tool_name in {"run_askcos_condition_prediction", "run_askcos_condition_prediction_compare", "run_askcos_quarc_prediction"}:
        if "smiles" in cleaned and "reaction_smiles" not in cleaned:
            cleaned["reaction_smiles"] = cleaned.pop("smiles")
        if "top_k" in cleaned and "n_conditions" not in cleaned:
            cleaned["n_conditions"] = cleaned.pop("top_k")

    if tool_name.startswith("run_askcos_multistep_retrosynthesis"):
        if "num_paths" in cleaned and "max_paths" not in cleaned:
            cleaned["max_paths"] = cleaned.pop("num_paths")
        if "max_steps" in cleaned and "max_depth" not in cleaned:
            cleaned["max_depth"] = cleaned.pop("max_steps")
        if "smiles" in cleaned and "target_smiles" not in cleaned:
            cleaned["target_smiles"] = cleaned["smiles"]
        if "target" in cleaned and "target_smiles" not in cleaned:
            cleaned["target_smiles"] = cleaned["target"]
        cleaned.pop("smiles", None)
        cleaned.pop("target", None)

    if tool_name == "run_askcos_route_recommendation":
        if "smiles" in cleaned and "target_smiles" not in cleaned:
            cleaned["target_smiles"] = cleaned["smiles"]
        if "target" in cleaned and "target_smiles" not in cleaned:
            cleaned["target_smiles"] = cleaned["target"]
        cleaned.pop("smiles", None)
        cleaned.pop("target", None)
        cleaned.setdefault("objective", "balanced")
        cleaned.setdefault("backend", "mcts")
        cleaned.setdefault("max_depth", 5)
        cleaned.setdefault("max_paths", 120)
        cleaned.setdefault("expansion_time", 180)
        cleaned.setdefault("top_n", 10)
        cleaned.setdefault("enable_pubchem_hazard", True)
        cleaned.setdefault("hazard_leaf_only", True)
        cleaned.setdefault("max_unique_hazard_checks", 120)
        cleaned.setdefault("use_cache", True)

    if tool_name == "run_askcos_impurity_prediction":
        if "reaction_smiles" in cleaned and "reactants_smiles" not in cleaned:
            cleaned["reactants_smiles"] = cleaned.pop("reaction_smiles")
        if "smiles" in cleaned and "reactants_smiles" not in cleaned:
            cleaned["reactants_smiles"] = cleaned.pop("smiles")

    return cleaned


def _tool_requires_smiles(tool_name: str) -> bool:
    return tool_name != "resolve_smiles_from_name"


def _tool_prefers_reaction_smiles(tool_name: str) -> bool:
    return tool_name in {
        "run_askcos_condition_prediction",
        "run_askcos_condition_prediction_compare",
        "run_askcos_quarc_prediction",
        "run_askcos_impurity_prediction",
    } or tool_name.startswith("run_askcos_forward_prediction")


def _summarize_tool_output(tool_name: str, raw_output: str) -> str:
    text = str(raw_output or "")
    if len(text) <= TOOL_RESULT_MAX_CHARS or not ENABLE_TOOL_OUTPUT_SUMMARY:
        return text[:TOOL_RESULT_MAX_CHARS]

    cache_key = cache.build_key("tool_summary:v1", tool_name=tool_name, raw_output=text[:8000])
    cached = cache.get(cache_key)
    if isinstance(cached, str) and cached.strip():
        return cached

    provider = AUX_PROVIDER
    model = GROQ_AUX_MODEL if provider == "groq" else AUX_MODEL
    summary = _generate_text(
        prompt=(
            "請將以下工具輸出摘要成 8 行內，保留最重要分數、候選、限制與錯誤。"
            f"\nTool={tool_name}\n{text[:8000]}"
        ),
        model=model,
        provider=provider,
        timeout_sec=45,
    )
    summary = (summary or text[:TOOL_RESULT_MAX_CHARS]).strip()
    cache.set(cache_key, summary)
    return summary[:TOOL_RESULT_MAX_CHARS]


def _pick_groq_decision_model(adaptive_plan: Dict[str, Any], user_prompt: str) -> str:
    return _pick_groq_model_for_task(
        user_prompt=user_prompt,
        task_type="decision",
        adaptive_plan=adaptive_plan or {},
    )


def _execute_tool(function_name: str, function_args: Dict[str, Any]) -> str:
    tool_fn = TOOLS_BY_NAME.get(function_name)
    if tool_fn is None:
        return f"未知的工具 {function_name}"
    try:
        return str(tool_fn(**function_args))
    except TypeError as e:
        return f"工具參數錯誤：{e}"
    except Exception as e:
        return f"工具執行失敗：{e}"


def _run_critic(user_prompt: str, final_answer: str, used_tools: List[str], evidence: List[str]) -> str:
    if not ENABLE_CRITIC:
        return ""
    provider = CRITIC_PROVIDER
    model = (
        _pick_groq_model_for_task(user_prompt=user_prompt, task_type="critic")
        if provider == "groq"
        else CRITIC_MODEL
    )
    try:
        return _generate_text(
            prompt=(
                "你是 AskLLM critic。請檢查回答是否與工具證據一致、是否漏掉關鍵限制、"
                "是否過度自信。請輸出精簡評論。\n"
                f"user_prompt={user_prompt}\nused_tools={used_tools}\n"
                f"evidence={json.dumps(evidence[-3:], ensure_ascii=False)}\n"
                f"final_answer={final_answer}"
            ),
            model=model,
            provider=provider,
            timeout_sec=45,
        )
    except Exception:
        return ""


def _conservative_low_confidence(user_prompt: str, critic_text: str, tool_outputs: List[str]) -> Dict[str, Any]:
    reasons = []
    if any(is_tool_error(x) for x in tool_outputs):
        reasons.append("tool_error")
    if not recent_effective_evidence(tool_outputs):
        reasons.append("no_effective_evidence")
    if "不確定" in critic_text or "證據不足" in critic_text or "inconsistent" in critic_text.lower():
        reasons.append("critic_warning")
    return {"low_confidence": bool(reasons), "reasons": reasons}


def _maybe_update_memory_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    if not state.get("ai_summary"):
        return state
    old_turns = pmem.compact_old_turns_for_summary(state)
    if not old_turns:
        return state

    prompt = (
        "請將以下較早對話壓縮成繁體中文長期摘要，保留使用者偏好、已知事實、"
        "當前任務上下文與未完成事項。避免逐字抄錄。\n"
        f"existing_summary={state.get('summary_zh', '')}\n"
        f"old_turns={json.dumps(old_turns, ensure_ascii=False)}"
    )
    provider = AUX_PROVIDER
    model = GROQ_AUX_MODEL if provider == "groq" else pmem.SUMMARY_MODEL
    try:
        new_summary = _generate_text(
            prompt=prompt,
            model=model,
            provider=provider,
            timeout_sec=45,
        ).strip()
        if new_summary:
            state = pmem.apply_summary_compression(state, new_summary, consumed_count=len(old_turns))
    except Exception:
        pass
    return state


def _maybe_add_reflection(state: Dict[str, Any], user_prompt: str, final_response: str, tool_outputs: List[str]) -> Dict[str, Any]:
    if not ENABLE_META_REFLECTION:
        return state
    if not tool_outputs:
        return state
    reflection = (
        f"查詢摘要：{user_prompt[:80]}；"
        f"工具數={len(tool_outputs)}；"
        f"錯誤工具={sum(1 for x in tool_outputs if is_tool_error(x))}；"
        f"最高分={max([extract_top_score(x) for x in tool_outputs] + [0.0]):.4f}"
    )
    return pmem.add_reflection(state, reflection)


def _call_model_with_tools(
    model_name: str,
    contents: List[Union[str, types.Content, types.Part]],
    tools: list,
    system_instruction: str,
):
    if client is None:
        raise RuntimeError("未設定 GEMINI_API_KEY，無法使用 Gemini function calling。")
    return client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=tools,
        ),
    )


def _run_gemini_turn(
    *,
    user_prompt: str,
    history: List[types.Content],
    tools_for_turn: list,
    system_instruction: str,
    adaptive_plan: Dict[str, Any],
) -> Tuple[str, List[str], str]:
    current_user_content = types.Content(role="user", parts=[types.Part(text=str(user_prompt))])
    contents = history + [current_user_content]
    current_model = PRIMARY_MODEL
    last_tool_output_result = None
    raw_tool_outputs: List[str] = []
    used_tool_names: List[str] = []

    try:
        response = _call_model_with_tools(current_model, contents, tools_for_turn, system_instruction)
    except Exception as primary_error:
        try:
            current_model = BACKUP_MODEL
            response = _call_model_with_tools(current_model, contents, tools_for_turn, system_instruction)
        except Exception as backup_error:
            return (
                "主備模型均調用失敗，請檢查 API Key 或配額。\n"
                f"主模型錯誤: {primary_error}\n備用模型錯誤: {backup_error}",
                [],
                current_model,
            )

    max_tool_calls = max(1, int(adaptive_plan.get("max_tool_calls", 2)))
    tool_call_count = 0

    while getattr(response, "function_calls", None) and tool_call_count < max_tool_calls:
        model_request_content = response.candidates[0].content if response.candidates else None
        tool_outputs = []
        for tool_call in response.function_calls:
            function_name = tool_call.name
            function_args = _sanitize_tool_args(function_name, dict(tool_call.args))
            if not function_args:
                function_args = _default_args_for_tool(function_name, user_prompt=user_prompt)

            if _tool_requires_smiles(function_name):
                if _tool_prefers_reaction_smiles(function_name):
                    reaction_candidate = function_args.get("reaction_smiles") or extract_smiles_candidate(user_prompt)
                    if not reaction_candidate or ">>" not in str(reaction_candidate):
                        name_candidate = extract_name_candidate(user_prompt)
                        if name_candidate and function_name.startswith("run_askcos_retrosynthesis"):
                            resolved_text = resolve_smiles_from_name(compound_name=name_candidate)
                            resolved_smiles = _extract_smiles_from_resolver_output(str(resolved_text))
                            if resolved_smiles:
                                function_args.update(_default_args_for_tool(function_name, user_prompt, resolved_smiles))
                else:
                    smiles_candidate = (
                        function_args.get("target_smiles")
                        or function_args.get("smiles")
                        or extract_smiles_candidate(user_prompt)
                    )
                    if not smiles_candidate or not looks_like_smiles(str(smiles_candidate)):
                        name_candidate = extract_name_candidate(user_prompt)
                        if name_candidate:
                            resolved_text = resolve_smiles_from_name(compound_name=name_candidate)
                            resolved_smiles = _extract_smiles_from_resolver_output(str(resolved_text))
                            if resolved_smiles:
                                function_args.update(_default_args_for_tool(function_name, user_prompt, resolved_smiles))

            tool_output = _execute_tool(function_name, function_args)
            last_tool_output_result = tool_output
            raw_tool_outputs.append(tool_output)
            used_tool_names.append(function_name)
            append_tool_trace(
                {
                    "ts": utc_now_iso(),
                    "query": user_prompt,
                    "tool_name": function_name,
                    "tool_args": function_args,
                    "raw_output": tool_output,
                    "output_for_model": _summarize_tool_output(function_name, tool_output),
                }
            )
            tool_outputs.append(
                types.Part.from_function_response(
                    name=function_name,
                    response={"tool_result": tool_output},
                )
            )
            tool_call_count += 1
            if tool_call_count >= max_tool_calls:
                break

        tool_outputs_content = types.Content(role="tool", parts=tool_outputs)
        contents_feedback = contents + ([model_request_content] if model_request_content else []) + [tool_outputs_content]
        contents = contents_feedback
        try:
            response = _call_model_with_tools(current_model, contents_feedback, tools_for_turn, system_instruction)
        except Exception as e:
            return f"模型 {current_model} 在工具調用反饋階段失敗: {e}", raw_tool_outputs, current_model

    final_response_text = response.text if response else ""
    if not final_response_text or not final_response_text.strip():
        if last_tool_output_result is not None:
            final_response_text = (
                "Agent 成功執行工具，但模型未能生成完整總結。以下是最後一次工具的結果：\n"
                f"{last_tool_output_result}"
            )
        else:
            final_response_text = "Agent 完成計算，但模型返回了空響應。"

    return final_response_text, raw_tool_outputs, current_model


def run_interactive_agent(
    user_prompt: str,
    history: List[types.Content],
    tools_to_use: list,
    long_term_summary_zh: str = "",
) -> str:
    state = pmem.load_state()
    if long_term_summary_zh and not state.get("summary_zh"):
        state["summary_zh"] = long_term_summary_zh

    if not history:
        history.extend(pmem.state_to_gemini_history(state))

    system_instruction = load_system_instruction_from_skill(user_prompt, state)
    adaptive_plan = _build_adaptive_plan(user_prompt, tools_to_use)
    tools_for_turn = _filter_tools_for_turn(user_prompt, tools_to_use, adaptive_plan)

    current_user_content = types.Content(role="user", parts=[types.Part(text=str(user_prompt))])
    tool_names = [getattr(t, "__name__", str(t)) for t in tools_for_turn]
    groq_decision_model_for_turn = _pick_groq_decision_model(adaptive_plan, user_prompt)

    resolved_smiles = extract_smiles_candidate(user_prompt)
    raw_tool_outputs: List[str] = []

    try:
        if DECISION_PROVIDER == "groq":
            final_response_text = run_groq_turn(
                user_prompt=user_prompt,
                history=history,
                current_user_content=current_user_content,
                types_module=types,
                tools_for_turn=tools_for_turn,
                adaptive_plan=adaptive_plan,
                compare_allowed=bool(adaptive_plan.get("compare_allowed")),
                tool_budget=int(adaptive_plan.get("max_tool_calls", 2)),
                enable_adaptive_policy=ENABLE_ADAPTIVE_POLICY,
                groq_decision_model_for_turn=groq_decision_model_for_turn,
                q=user_prompt,
                ql=user_prompt.lower(),
                smiles=resolved_smiles,
                resolve_smiles_from_name_fn=resolve_smiles_from_name,
                looks_like_smiles_fn=looks_like_smiles,
                extract_name_candidate_fn=extract_name_candidate,
                write_evidence_log_fn=write_evidence_log,
                append_tool_trace_fn=append_tool_trace,
                tool_output_for_model_fn=_summarize_tool_output,
                utc_now_iso_fn=utc_now_iso,
                generate_text_fn=_generate_text,
                extract_json_block_fn=_extract_json_block,
                extract_smiles_candidate_fn=_extract_smiles_from_resolver_output,
                default_args_for_tool_fn=lambda tool_name, latest_smiles="": _default_args_for_tool(
                    tool_name,
                    user_prompt,
                    latest_smiles or resolved_smiles,
                ),
                sanitize_tool_args_fn=_sanitize_tool_args,
                execute_tool_fn=_execute_tool,
                conservative_low_confidence_fn=_conservative_low_confidence,
                run_critic_fn=_run_critic,
                is_tool_error_fn=is_tool_error,
                extract_top_score_fn=extract_top_score,
                aux_model=AUX_MODEL,
                groq_aux_model=GROQ_AUX_MODEL,
                primary_model=groq_decision_model_for_turn,
                planner_timeout_sec=PLANNER_TIMEOUT_SEC,
            )
        else:
            final_response_text, raw_tool_outputs, current_model = _run_gemini_turn(
                user_prompt=user_prompt,
                history=history,
                tools_for_turn=tools_for_turn,
                system_instruction=system_instruction,
                adaptive_plan=adaptive_plan,
            )
            critic_text = _run_critic(user_prompt, final_response_text, tool_names, raw_tool_outputs)
            write_evidence_log(
                {
                    "ts": utc_now_iso(),
                    "query": user_prompt,
                    "decision_provider": "gemini",
                    "decision_model_for_turn": current_model,
                    "planner_on": ENABLE_ADAPTIVE_POLICY,
                    "planner_output": adaptive_plan,
                    "compare_allowed": bool(adaptive_plan.get("compare_allowed")),
                    "tool_call_count": len(raw_tool_outputs),
                    "tool_names": tool_names,
                    "critic": critic_text[:500],
                    "tool_outputs_preview": [_summarize_tool_output("tool", x) for x in raw_tool_outputs[-3:]],
                }
            )
    except QuotaLimitError:
        raise
    except Exception as e:
        final_response_text = f"AskLLM 執行失敗：{e}"

    history.append(current_user_content)
    history.append(types.Content(role="model", parts=[types.Part(text=final_response_text)]))

    state = pmem.append_turn(state, "user", user_prompt)
    state = pmem.append_turn(state, "model", final_response_text)
    state = _maybe_update_memory_summary(state)
    state = _maybe_add_reflection(state, user_prompt, final_response_text, raw_tool_outputs)
    pmem.save_state(state)

    return final_response_text


def _memory_command(user_input: str, state: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    cmd = (user_input or "").strip()
    if not cmd.startswith("/memory"):
        return False, "", state

    if cmd == "/memory show":
        return True, json.dumps(state, ensure_ascii=False, indent=2), state
    if cmd == "/memory clear":
        state = pmem.clear_state()
        return True, "已清除全部持久記憶。", state
    if cmd == "/memory clear turns":
        state = pmem.clear_turns_only(state)
        pmem.save_state(state)
        return True, "已清除最近 raw turns。", state
    if cmd == "/memory clear topic":
        state = pmem.clear_current_topic(state)
        pmem.save_state(state)
        return True, "已清除目前主題記憶。", state
    if cmd.startswith("/memory summary "):
        state = pmem.set_summary(state, cmd[len("/memory summary "):].strip())
        pmem.save_state(state)
        return True, "已更新長期摘要。", state
    if cmd == "/memory ai on":
        state["ai_summary"] = True
        pmem.save_state(state)
        return True, "已啟用 AI 記憶壓縮。", state
    if cmd == "/memory ai off":
        state["ai_summary"] = False
        pmem.save_state(state)
        return True, "已關閉 AI 記憶壓縮。", state
    if cmd == "/memory ai status":
        return True, f"AI 記憶壓縮目前為: {'on' if state.get('ai_summary') else 'off'}", state
    return True, "可用指令：/memory show | /memory clear | /memory clear turns | /memory clear topic | /memory summary ... | /memory ai on|off|status", state


def _topic_command(user_input: str, state: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    cmd = (user_input or "").strip()
    if not cmd.startswith("/topic"):
        return False, "", state
    if cmd.startswith("/topic set "):
        state = pmem.set_topic(state, cmd[len("/topic set "):].strip())
        pmem.save_state(state)
        return True, f"目前主題已設為：{state.get('current_topic', '')}", state
    if cmd == "/topic show":
        topic = state.get("current_topic", "") or "未設定"
        summary = pmem.get_current_topic_summary(state) or "（尚無）"
        return True, f"目前主題：{topic}\n主題摘要：{summary}", state
    if cmd == "/topic list":
        topics = state.get("topics", {})
        return True, json.dumps(topics, ensure_ascii=False, indent=2), state
    return True, "可用指令：/topic set <name> | /topic show | /topic list", state


def _planner_command(user_input: str) -> Tuple[bool, str]:
    global ENABLE_ADAPTIVE_POLICY
    cmd = (user_input or "").strip()
    if not cmd.startswith("/planner"):
        return False, ""
    if cmd == "/planner on":
        ENABLE_ADAPTIVE_POLICY = True
        return True, "已啟用 adaptive planner。"
    if cmd == "/planner off":
        ENABLE_ADAPTIVE_POLICY = False
        return True, "已關閉 adaptive planner。"
    if cmd == "/planner status":
        return True, f"adaptive planner 目前為: {'on' if ENABLE_ADAPTIVE_POLICY else 'off'}"
    return True, "可用指令：/planner on | /planner off | /planner status"


askcos_tools = [
    run_askcos_forward_prediction,
    run_askcos_forward_prediction_uspto_stereo,
    run_askcos_forward_prediction_graph2smiles,
    run_askcos_forward_prediction_wldn5,
    run_askcos_forward_prediction_compare,
    run_askcos_retrosynthesis,
    run_askcos_retrosynthesis_uspto_full,
    run_askcos_retrosynthesis_pistachio,
    run_askcos_retrosynthesis_template_enum,
    run_askcos_retrosynthesis_compare,
    run_askcos_multistep_retrosynthesis,
    run_askcos_multistep_retrosynthesis_retro_star,
    run_askcos_multistep_retrosynthesis_compare,
    run_askcos_route_recommendation,
    run_askcos_impurity_prediction,
    resolve_smiles_from_name,
    run_askcos_condition_prediction,
    run_askcos_condition_prediction_compare,
    run_askcos_quarc_prediction,
]
TOOLS_BY_NAME = {tool.__name__: tool for tool in askcos_tools}


def main_loop():
    print("--- 歡迎使用 AskLLM 研究版化學 Agent ---")
    print("可用指令：/memory ... | /topic ... | /planner ... | exit")

    state = pmem.load_state()
    chat_history: List[types.Content] = pmem.state_to_gemini_history(state)

    while True:
        try:
            user_input = input("\n[ 用戶] 請輸入您的查詢: ")
            if user_input.lower() in ["exit", "quit"]:
                print("\n[ Agent] 謝謝使用，再見！")
                break
            if not user_input.strip():
                continue

            handled, text, state = _memory_command(user_input, state)
            if handled:
                print(f"\n[ Agent] {text}")
                continue

            handled, text, state = _topic_command(user_input, state)
            if handled:
                print(f"\n[ Agent] {text}")
                continue

            handled, text = _planner_command(user_input)
            if handled:
                print(f"\n[ Agent] {text}")
                continue

            try:
                final_response = run_interactive_agent(
                    user_prompt=user_input,
                    history=chat_history,
                    tools_to_use=askcos_tools,
                    long_term_summary_zh=state.get("summary_zh", ""),
                )
                state = pmem.load_state()
                print(f"\n[ Agent] 最終回覆:\n{final_response}")
            except QuotaLimitError as e:
                print(f"\n[ Agent] {e}")
            except Exception as e:
                print(f"\n[ 錯誤] 發生意外錯誤: {e}")
                print("請重試或檢查 AskCOS / Gemini / Groq 服務是否運行正常。")
        except KeyboardInterrupt:
            print("\n[ Agent] 已中止，謝謝使用。")
            break


if __name__ == "__main__":
    main_loop()