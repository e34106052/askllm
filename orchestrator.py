import json
import re
from typing import Any, Callable, Dict, List


SWITCH_REASON_ENUM = {
    "tool_error",
    "low_score",
    "no_evidence_gain",
    "quota_limit",
    "duplicate_step",
    "invalid_plan",
    "plan_success",
}


def build_route_candidates(user_prompt: str, compare_allowed: bool) -> List[Dict[str, Any]]:
    lower = (user_prompt or "").lower()
    routes = [
        {
            "plan_id": "A",
            "label": "primary",
            "strategy": "single_best_tool",
            "success_hint": "用最符合問題意圖的單一工具先取得高品質證據。",
        },
        {
            "plan_id": "B",
            "label": "robust",
            "strategy": "cross_check_or_fallback",
            "success_hint": "若 A 無法提供穩定證據，改用替代引擎或 compare 交叉驗證。",
        },
        {
            "plan_id": "C",
            "label": "fallback",
            "strategy": "minimal_answer_or_clarify",
            "success_hint": "若工具不可用或輸入不足，改做名稱解析、澄清、或基於現有證據收斂回答。",
        },
    ]
    if not compare_allowed and "compare" not in lower and "比較" not in lower:
        routes[1]["success_hint"] = "若 A 失敗，優先改用另一個單工具，不主動 compare。"
    return routes


def compact_abandoned_routes(plan_switch_logs: List[Dict[str, Any]], step_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact = []
    for item in plan_switch_logs[-6:]:
        from_plan = item.get("from", "")
        related = [x for x in step_records if x.get("plan_id") == from_plan][-2:]
        compact.append(
            {
                "from": from_plan,
                "to": item.get("to", ""),
                "switch_reason": item.get("switch_reason", ""),
                "recent_steps": [
                    {
                        "tool_name": x.get("tool_name", ""),
                        "score": x.get("score", 0.0),
                        "error": x.get("error", False),
                        "evidence_gain": x.get("evidence_gain", 0),
                    }
                    for x in related
                ],
            }
        )
    return compact


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


def run_groq_turn(
    *,
    user_prompt: str,
    history: list,
    current_user_content: Any,
    types_module: Any,
    tools_for_turn: list,
    adaptive_plan: dict,
    compare_allowed: bool,
    tool_budget: int,
    enable_adaptive_policy: bool,
    groq_decision_model_for_turn: str,
    q: str,
    ql: str,
    smiles: str,
    resolve_smiles_from_name_fn: Callable[..., Any],
    looks_like_smiles_fn: Callable[[str], bool],
    extract_name_candidate_fn: Callable[[str], str],
    extract_smiles_candidate_fn: Callable[[str], str],
    write_evidence_log_fn: Callable[[dict], None],
    append_tool_trace_fn: Callable[[dict], None],
    tool_output_for_model_fn: Callable[[str, str], str],
    utc_now_iso_fn: Callable[[], str],
    generate_text_fn: Callable[..., str],
    extract_json_block_fn: Callable[[str], dict],
    default_args_for_tool_fn: Callable[[str], dict],
    sanitize_tool_args_fn: Callable[[str, dict], dict],
    execute_tool_fn: Callable[[str, dict], Any],
    conservative_low_confidence_fn: Callable[[str, str, List[str]], Any],
    run_critic_fn: Callable[[str, str, List[str], List[str]], str],
    is_tool_error_fn: Callable[[str], bool],
    extract_top_score_fn: Callable[[str], float],
    aux_model: str,
    groq_aux_model: str,
    primary_model: str,
    planner_timeout_sec: int,
) -> str:
    tool_names = [getattr(t, "__name__", str(t)) for t in tools_for_turn]
    route_candidates = build_route_candidates(user_prompt, compare_allowed)
    plan_contracts = {
        "A": {"success": "獲得非空且非錯誤的主要工具證據", "failure": "工具錯誤、低分或無證據增益"},
        "B": {"success": "交叉驗證或替代引擎提供更穩定證據", "failure": "仍無法改善證據品質"},
        "C": {"success": "給出澄清請求或保守答案", "failure": "仍無法形成可交付輸出"},
    }

    current_plan_id = "A"
    plan_switch_logs: List[Dict[str, Any]] = []
    step_summaries: List[str] = []
    raw_tool_outputs: List[str] = []
    compact_evidence: List[str] = []
    step_records: List[Dict[str, Any]] = []
    tool_call_count = 0
    used_tool_names: List[str] = []
    used_steps = set()
    resolved_smiles = smiles
    non_adaptive_executed = False

    if not resolved_smiles and not looks_like_smiles_fn(user_prompt):
        name_candidate = extract_name_candidate_fn(user_prompt)
        if name_candidate:
            resolved_text = str(resolve_smiles_from_name_fn(compound_name=name_candidate))
            maybe_smiles = extract_smiles_candidate_fn(resolved_text)
            if maybe_smiles:
                resolved_smiles = maybe_smiles
            append_tool_trace_fn(
                {
                    "ts": utc_now_iso_fn(),
                    "query": user_prompt,
                    "tool_name": "resolve_smiles_from_name",
                    "tool_args": {"compound_name": name_candidate},
                    "raw_output": resolved_text,
                    "output_for_model": tool_output_for_model_fn("resolve_smiles_from_name", resolved_text),
                }
            )
            raw_tool_outputs.append(resolved_text)
            compact_evidence.append(tool_output_for_model_fn("resolve_smiles_from_name", resolved_text))

    while tool_call_count < max(1, tool_budget):
        if enable_adaptive_policy:
            planner_prompt = {
                "user_prompt": user_prompt,
                "adaptive_plan": adaptive_plan,
                "current_plan_id": current_plan_id,
                "route_candidates": route_candidates,
                "plan_contracts": plan_contracts,
                "used_tools": used_tool_names,
                "available_tools": tool_names,
                "compact_evidence": compact_evidence[-3:],
                "step_summaries": step_summaries[-3:],
            }
            decision_text = generate_text_fn(
                prompt=(
                    "你是 AskLLM 的決策規劃器。請輸出 JSON，欄位包含："
                    "tool_name, args, expected_gain, stop, final_answer, switch_plan, switch_reason。"
                    f"\n{json.dumps(planner_prompt, ensure_ascii=False)}"
                ),
                model=groq_decision_model_for_turn,
                provider="groq",
                timeout_sec=planner_timeout_sec,
            )
            decision = extract_json_block_fn(decision_text) or _extract_json_block(decision_text)
        else:
            if non_adaptive_executed:
                break
            fallback_tool = ""
            for name in adaptive_plan.get("tool_candidates", []):
                if name in tool_names:
                    fallback_tool = name
                    break
            if not fallback_tool and tool_names:
                fallback_tool = tool_names[0]
            decision = {
                "tool_name": fallback_tool,
                "args": {},
                "expected_gain": "baseline",
                "stop": False,
            }
            non_adaptive_executed = True

        if decision.get("switch_plan") and decision.get("switch_plan") != current_plan_id:
            switch_reason = decision.get("switch_reason", "invalid_plan")
            if switch_reason not in SWITCH_REASON_ENUM:
                switch_reason = "invalid_plan"
            plan_switch_logs.append(
                {
                    "ts": utc_now_iso_fn(),
                    "from": current_plan_id,
                    "to": decision.get("switch_plan"),
                    "switch_reason": switch_reason,
                }
            )
            current_plan_id = decision.get("switch_plan")

        if decision.get("stop"):
            final_answer = str(decision.get("final_answer") or "").strip()
            if final_answer:
                write_evidence_log_fn(
                    {
                        "ts": utc_now_iso_fn(),
                        "query": user_prompt,
                        "decision_provider": "groq",
                        "decision_model_for_turn": groq_decision_model_for_turn,
                        "tool_call_count": tool_call_count,
                        "used_tool_count": len(set(used_tool_names)),
                        "tool_names": used_tool_names,
                        "planner_output": adaptive_plan,
                        "plan_switch_logs": plan_switch_logs,
                        "plan_contracts": plan_contracts,
                        "final_plan_id": current_plan_id,
                        "tool_outputs_preview": compact_evidence[-3:],
                    }
                )
                return final_answer
            break

        tool_name = str(decision.get("tool_name") or "").strip()
        if not tool_name:
            break
        if tool_name not in tool_names:
            plan_switch_logs.append(
                {
                    "ts": utc_now_iso_fn(),
                    "from": current_plan_id,
                    "to": "B" if current_plan_id == "A" else "C",
                    "switch_reason": "invalid_plan",
                }
            )
            current_plan_id = "B" if current_plan_id == "A" else "C"
            continue

        args = default_args_for_tool_fn(tool_name, resolved_smiles)
        args.update(decision.get("args") or {})
        args = sanitize_tool_args_fn(tool_name, args)
        step_key = json.dumps({"tool_name": tool_name, "args": args}, ensure_ascii=False, sort_keys=True)
        if step_key in used_steps:
            next_plan = "B" if current_plan_id == "A" else "C"
            plan_switch_logs.append(
                {
                    "ts": utc_now_iso_fn(),
                    "from": current_plan_id,
                    "to": next_plan,
                    "switch_reason": "duplicate_step",
                }
            )
            current_plan_id = next_plan
            continue
        used_steps.add(step_key)

        tool_output = str(execute_tool_fn(tool_name, args))
        tool_call_count += 1
        used_tool_names.append(tool_name)
        raw_tool_outputs.append(tool_output)
        summarized = tool_output_for_model_fn(tool_name, tool_output)
        compact_evidence.append(summarized)
        append_tool_trace_fn(
            {
                "ts": utc_now_iso_fn(),
                "query": user_prompt,
                "tool_name": tool_name,
                "tool_args": args,
                "raw_output": tool_output,
                "output_for_model": summarized,
            }
        )

        score = extract_top_score_fn(tool_output)
        error_flag = is_tool_error_fn(tool_output)
        evidence_gain = 1 if summarized not in compact_evidence[:-1] else 0
        step_records.append(
            {
                "plan_id": current_plan_id,
                "tool_name": tool_name,
                "args": args,
                "score": score,
                "error": error_flag,
                "evidence_gain": evidence_gain,
            }
        )
        step_summaries.append(
            f"plan={current_plan_id}; tool={tool_name}; error={error_flag}; "
            f"score={score:.4f}; evidence_gain={evidence_gain}"
        )

        if error_flag or evidence_gain == 0:
            next_plan = "B" if current_plan_id == "A" else "C"
            switch_reason = "tool_error" if error_flag else "no_evidence_gain"
            plan_switch_logs.append(
                {
                    "ts": utc_now_iso_fn(),
                    "from": current_plan_id,
                    "to": next_plan,
                    "switch_reason": switch_reason,
                }
            )
            current_plan_id = next_plan
        elif score <= 0.15:
            next_plan = "B" if current_plan_id == "A" else "C"
            plan_switch_logs.append(
                {
                    "ts": utc_now_iso_fn(),
                    "from": current_plan_id,
                    "to": next_plan,
                    "switch_reason": "low_score",
                }
            )
            current_plan_id = next_plan

        if score > 0.5 and not error_flag:
            plan_switch_logs.append(
                {
                    "ts": utc_now_iso_fn(),
                    "from": current_plan_id,
                    "to": current_plan_id,
                    "switch_reason": "plan_success",
                }
            )
            break

    critic_text = ""
    try:
        critic_text = run_critic_fn(user_prompt, "\n\n".join(raw_tool_outputs), used_tool_names, compact_evidence[-3:])
    except Exception:
        critic_text = ""

    low_conf = conservative_low_confidence_fn(
        user_prompt,
        critic_text,
        raw_tool_outputs,
    )

    final_prompt = {
        "user_prompt": user_prompt,
        "tool_outputs": compact_evidence[-4:],
        "critic": critic_text,
        "low_confidence": low_conf,
        "current_plan_id": current_plan_id,
        "plan_switch_logs": plan_switch_logs,
    }
    final_answer = generate_text_fn(
        prompt=(
            "你是 AskLLM 的最終回答器。請根據以下工具證據，以繁體中文給出完整答案；"
            "若證據不足，要誠實指出限制並給出下一步建議。\n"
            f"{json.dumps(final_prompt, ensure_ascii=False)}"
        ),
        model=primary_model,
        provider="groq",
        timeout_sec=planner_timeout_sec,
    )

    write_evidence_log_fn(
        {
            "ts": utc_now_iso_fn(),
            "query": user_prompt,
            "decision_provider": "groq",
            "decision_model_for_turn": groq_decision_model_for_turn,
            "tool_call_count": tool_call_count,
            "used_tool_count": len(set(used_tool_names)),
            "tool_names": used_tool_names,
            "planner_output": adaptive_plan,
            "plan_switch_logs": plan_switch_logs,
            "abandoned_routes_compact": compact_abandoned_routes(plan_switch_logs, step_records),
            "plan_contracts": plan_contracts,
            "final_plan_id": current_plan_id,
            "tool_outputs_preview": compact_evidence[-3:],
            "critic": critic_text[:500],
            "step_records": step_records[-6:],
        }
    )

    return final_answer
