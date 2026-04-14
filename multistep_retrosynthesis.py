import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

from askcos_tree_utils import parse_uds_paths, route_summary
import cache_utils as cache


ASKCOS_MULTISTEP_URL = "http://127.0.0.1:7000/get_buyable_paths"
# 與本機 docker host 網路中的 retro_star_* 容器埠一致（依實際部署調整）
ASKCOS_RETROSTAR_PORTS = [9322, 9323, 9324, 9325, 9326, 9327, 9328, 9329, 9330, 9331]
ASYNC_JOBS_DIR = os.path.join(os.path.dirname(__file__), "runtime_jobs", "multistep")


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_async_jobs_dir() -> None:
    os.makedirs(ASYNC_JOBS_DIR, exist_ok=True)


def _job_file_path(job_id: str) -> str:
    return os.path.join(ASYNC_JOBS_DIR, f"{job_id}.json")


def _result_file_path(job_id: str) -> str:
    return os.path.join(ASYNC_JOBS_DIR, f"{job_id}.result.txt")


def _read_job(job_id: str) -> dict:
    path = _job_file_path(job_id)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_job(job_id: str, payload: dict) -> None:
    _ensure_async_jobs_dir()
    with open(_job_file_path(job_id), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _list_jobs() -> list:
    _ensure_async_jobs_dir()
    out = []
    for name in os.listdir(ASYNC_JOBS_DIR):
        if not name.endswith(".json"):
            continue
        job_id = name[:-5]
        data = _read_job(job_id)
        if data:
            out.append(data)
    out.sort(key=lambda x: str(x.get("submitted_at", "")), reverse=True)
    return out


def _format_pathway_item(pathway: Any, idx: int) -> str:
    if isinstance(pathway, dict):
        score = _safe_float(
            pathway.get("score", pathway.get("plausibility", pathway.get("path_score", 0.0)))
        )
        chemicals = pathway.get("chemicals", [])
        reactions = pathway.get("reactions", [])
        return (
            f"--- 路徑 {idx} (得分: {score:.4f}) ---\n"
            f"  - 化學節點數: {len(chemicals)}\n"
            f"  - 反應步數: {len(reactions)}\n"
            f"  - 關鍵資訊: {json.dumps(pathway, ensure_ascii=False)[:600]}"
        )
    return f"--- 路徑 {idx} ---\n  - 原始內容: {str(pathway)[:600]}"


def _format_route_summary(item: dict) -> str:
    idx = item.get("route_id", "?")
    lines = [f"--- 路徑 {idx} ---"]
    lines.append(f"  - 反應步數(depth): {item.get('depth') if item.get('depth') is not None else item.get('num_reactions')}")
    lines.append(f"  - 估計前驅物成本(precursor_cost): {item.get('precursor_cost')}")
    lines.append(f"  - 路徑分數(score): {item.get('score')}")
    lines.append(f"  - 叢集(cluster_id): {item.get('cluster_id')}")
    lines.append(f"  - 路徑整體plausibility乘積: {item.get('plausibility_product')}")
    lines.append(f"  - 反應節點數: {item.get('num_reactions')}")
    lines.append(f"  - 終端前驅物數: {item.get('num_leaf_precursors')}")
    for rxn in (item.get("reaction_nodes") or [])[:8]:
        if rxn == (item.get("reaction_nodes") or [])[0]:
            lines.append("  - 反應序列(前幾步):")
        lines.append(f"    * {rxn}")
    for smi in (item.get("leaf_chemicals") or [])[:12]:
        if smi == (item.get("leaf_chemicals") or [])[0]:
            lines.append("  - 終端前驅物(前幾個):")
        lines.append(f"    * {smi}")
    return "\n".join(lines)


def _detect_retro_star_url() -> str:
    for port in ASKCOS_RETROSTAR_PORTS:
        try:
            probe = subprocess.run(
                ["curl", "-sS", f"http://127.0.0.1:{port}/openapi.json"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            api = json.loads(probe.stdout)
            if "/get_buyable_paths" in (api.get("paths") or {}):
                return f"http://127.0.0.1:{port}/get_buyable_paths"
        except Exception:
            continue
    return "http://127.0.0.1:9322/get_buyable_paths"


def _curl_wall_timeout_seconds(expansion_time: int, backend_label: str) -> int:
    """
    Retro* 在相同 gateway 下常比 MCTS 慢（實測單次可 >300s）。
    必須大於 build_tree_options.expansion_time，並留足 HTTP 緩衝，否則會誤判為「服務掛掉」。
    """
    base = max(180, expansion_time + 180)
    if backend_label.strip().lower().startswith("retro"):
        return max(base, expansion_time + 420)
    return base


def _run_backend(
    *,
    backend_label: str,
    endpoint_url: str,
    payload: dict,
    max_paths: int,
    expansion_time: int,
    use_cache: bool,
) -> str:
    cache_key = cache.build_key(f"askcos:multistep:{backend_label}:v1", url=endpoint_url, payload=payload)
    if use_cache:
        cached = cache.get(cache_key)
        if isinstance(cached, str) and cached.strip():
            print(f"  -> 命中快取：AskCOS 多步逆合成（{backend_label}）")
            return cached

    try:
        result = subprocess.run(
            [
                "curl",
                endpoint_url,
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
            timeout=_curl_wall_timeout_seconds(expansion_time, backend_label),
        )
        data = json.loads(result.stdout)
        results_obj = data.get("results", {}) if isinstance(data, dict) else {}
        stats_obj = results_obj.get("stats", {}) if isinstance(results_obj, dict) else {}
        uds_obj = results_obj.get("uds", {}) if isinstance(results_obj, dict) else {}
        parsed_routes = parse_uds_paths(uds_obj if isinstance(uds_obj, dict) else {})

        pathways = (
            results_obj.get("paths")
            or results_obj.get("pathways")
            or data.get("paths")
            or data.get("pathways")
            or []
        )
        total_paths = int(
            stats_obj.get("total_paths", data.get("total_paths", len(pathways) if isinstance(pathways, list) else 0))
        )
        total_chemicals = int(stats_obj.get("total_chemicals", data.get("total_chemicals", 0)))
        total_reactions = int(stats_obj.get("total_reactions", data.get("total_reactions", 0)))

        if not parsed_routes and not pathways and total_paths <= 0:
            text = (
                f"AskCOS 多步逆合成（{backend_label}）未找到可回推到可購買起始物的完整路徑。\n"
                f"搜尋統計：total_paths={total_paths}，total_chemicals={total_chemicals}，total_reactions={total_reactions}"
            )
            if use_cache:
                cache.set(cache_key, text)
            return text

        lines = [
            f"AskCOS 多步逆合成（{backend_label}）完成。共找到 {total_paths} 條路徑。",
            f"搜尋統計：total_chemicals={total_chemicals}，total_reactions={total_reactions}",
        ]
        if parsed_routes:
            for rt in parsed_routes[:max_paths]:
                lines.append(_format_route_summary(route_summary(rt)))
        else:
            for idx, pathway in enumerate(pathways[:max_paths], start=1):
                lines.append(_format_pathway_item(pathway, idx))
        text = "\n".join(lines)
        if use_cache:
            cache.set(cache_key, text)
        return text
    except subprocess.CalledProcessError as e:
        detail = e.stderr if e.stderr else f"Curl 退出代碼: {e.returncode}"
        return f"調用 AskCOS 多步逆合成 API 失敗（{backend_label}）。錯誤詳情:\n{detail}"
    except subprocess.TimeoutExpired:
        return (
            f"多步逆合成逾時（{backend_label}）：expansion_time={expansion_time}，"
            f"HTTP 等待上限約 {_curl_wall_timeout_seconds(expansion_time, backend_label)} 秒。"
            f"若為 Retro*，可再提高 expansion_time 或檢查 docker log（單次常需數分鐘）。"
        )
    except Exception as e:
        return f"多步逆合成發生未知錯誤（{backend_label}）: {e}"


def run_askcos_multistep_retrosynthesis(
    target_smiles: str,
    max_depth: int = 6,
    max_paths: int = 5,
    expansion_time: int = 45,
    max_branching: int = 25,
    retro_model_name: str = "reaxys",
    max_num_templates: int = 200,
    top_k: int = 20,
    threshold: float = 0.15,
    sorting_metric: str = "plausibility",
    use_cache: bool = True,
) -> str:
    if not target_smiles:
        return "錯誤：未提供目標分子的 SMILES。"

    payload = {
        "smiles": target_smiles,
        "expand_one_options": {
            "retro_backend_options": [
                {
                    "retro_backend": "template_relevance",
                    "retro_model_name": retro_model_name,
                    "max_num_templates": max_num_templates,
                    "top_k": top_k,
                    "threshold": threshold,
                }
            ]
        },
        "build_tree_options": {
            "max_depth": max_depth,
            "expansion_time": expansion_time,
            "max_branching": max_branching,
            "termination_logic": {"and": ["buyable"]},
        },
        "enumerate_paths_options": {
            "max_paths": max_paths,
            "sorting_metric": sorting_metric,
            "validate_paths": True,
        },
    }
    return _run_backend(
        backend_label="MCTS",
        endpoint_url=ASKCOS_MULTISTEP_URL,
        payload=payload,
        max_paths=max_paths,
        expansion_time=expansion_time,
        use_cache=use_cache,
    )


def run_askcos_multistep_retrosynthesis_retro_star(
    target_smiles: str,
    max_depth: int = 6,
    max_paths: int = 5,
    expansion_time: int = 45,
    max_branching: int = 25,
    retro_model_name: str = "reaxys",
    max_num_templates: int = 200,
    top_k: int = 20,
    threshold: float = 0.15,
    sorting_metric: str = "plausibility",
    use_cache: bool = True,
) -> str:
    if not target_smiles:
        return "錯誤：未提供目標分子的 SMILES。"
    payload = {
        "smiles": target_smiles,
        "expand_one_options": {
            "retro_backend_options": [
                {
                    "retro_backend": "template_relevance",
                    "retro_model_name": retro_model_name,
                    "max_num_templates": max_num_templates,
                    "top_k": top_k,
                    "threshold": threshold,
                }
            ]
        },
        "build_tree_options": {
            "max_depth": max_depth,
            "expansion_time": expansion_time,
            "max_branching": max_branching,
            "termination_logic": {"and": ["buyable"]},
        },
        "enumerate_paths_options": {
            "max_paths": max_paths,
            "sorting_metric": sorting_metric,
            "validate_paths": True,
        },
    }
    return _run_backend(
        backend_label="Retro*",
        endpoint_url=_detect_retro_star_url(),
        payload=payload,
        max_paths=max_paths,
        expansion_time=expansion_time,
        use_cache=use_cache,
    )


def run_askcos_multistep_retrosynthesis_compare(
    target_smiles: str,
    max_depth: int = 6,
    max_paths: int = 5,
    expansion_time: int = 45,
    retro_model_name: str = "reaxys",
    use_cache: bool = True,
) -> str:
    mcts_text = run_askcos_multistep_retrosynthesis(
        target_smiles=target_smiles,
        max_depth=max_depth,
        max_paths=max_paths,
        expansion_time=expansion_time,
        retro_model_name=retro_model_name,
        use_cache=use_cache,
    )
    retro_text = run_askcos_multistep_retrosynthesis_retro_star(
        target_smiles=target_smiles,
        max_depth=max_depth,
        max_paths=max_paths,
        expansion_time=expansion_time,
        retro_model_name=retro_model_name,
        use_cache=use_cache,
    )
    return "AskCOS 多步逆合成比較（MCTS vs Retro*）\n\n=== MCTS ===\n" + mcts_text + "\n\n=== Retro* ===\n" + retro_text


def run_askcos_multistep_retrosynthesis_async_submit(
    target_smiles: str,
    backend: str = "mcts",
    max_depth: int = 6,
    max_paths: int = 5,
    expansion_time: int = 180,
    max_branching: int = 25,
    retro_model_name: str = "reaxys",
    max_num_templates: int = 200,
    top_k: int = 20,
    threshold: float = 0.15,
    sorting_metric: str = "plausibility",
    use_cache: bool = True,
) -> str:
    if not target_smiles:
        return "錯誤：未提供目標分子的 SMILES。"

    _ensure_async_jobs_dir()
    key = (backend or "mcts").strip().lower()
    if key in {"retrostar", "retro*", "retro_star"}:
        key = "retro_star"
    elif key in {"compare", "both"}:
        key = "compare"
    else:
        key = "mcts"

    job_id = f"ms_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    payload = {
        "job_id": job_id,
        "status": "queued",
        "submitted_at": _utc_now(),
        "backend": key,
        "params": {
            "target_smiles": target_smiles,
            "max_depth": int(max_depth),
            "max_paths": int(max_paths),
            "expansion_time": int(expansion_time),
            "max_branching": int(max_branching),
            "retro_model_name": retro_model_name,
            "max_num_templates": int(max_num_templates),
            "top_k": int(top_k),
            "threshold": float(threshold),
            "sorting_metric": sorting_metric,
            "use_cache": bool(use_cache),
        },
        "result_file": _result_file_path(job_id),
    }
    _write_job(job_id, payload)

    runner_path = os.path.join(os.path.dirname(__file__), "multistep_async_runner.py")
    proc = subprocess.Popen(
        [sys.executable, runner_path, "--job-id", job_id],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    payload["pid"] = int(proc.pid)
    payload["status"] = "running"
    payload["started_at"] = _utc_now()
    _write_job(job_id, payload)
    return (
        "已提交多步逆合成背景任務。\n"
        f"- job_id: {job_id}\n"
        f"- backend: {key}\n"
        "可用 run_askcos_multistep_retrosynthesis_async_status(job_id=...) 查詢進度，"
        "完成後用 run_askcos_multistep_retrosynthesis_async_result(job_id=...) 取結果。"
    )


def run_askcos_multistep_retrosynthesis_async_status(job_id: str) -> str:
    if not job_id:
        return "錯誤：未提供 job_id。"
    job = _read_job(job_id)
    if not job:
        return f"找不到背景任務：{job_id}"
    lines = [
        "多步逆合成背景任務狀態",
        f"- job_id: {job_id}",
        f"- status: {job.get('status', 'unknown')}",
        f"- backend: {job.get('backend', 'unknown')}",
        f"- submitted_at: {job.get('submitted_at', '')}",
    ]
    if job.get("started_at"):
        lines.append(f"- started_at: {job.get('started_at')}")
    if job.get("ended_at"):
        lines.append(f"- ended_at: {job.get('ended_at')}")
    if job.get("error"):
        lines.append(f"- error: {job.get('error')}")
    return "\n".join(lines)


def run_askcos_multistep_retrosynthesis_async_result(job_id: str) -> str:
    if not job_id:
        return "錯誤：未提供 job_id。"
    job = _read_job(job_id)
    if not job:
        return f"找不到背景任務：{job_id}"
    status = str(job.get("status", "unknown")).lower()
    if status not in {"done", "failed"}:
        return (
            f"任務尚未完成（status={status}）。"
            f"請稍後再查 run_askcos_multistep_retrosynthesis_async_status(job_id='{job_id}')。"
        )
    if status == "failed":
        return f"背景任務失敗（job_id={job_id}）：{job.get('error', 'unknown')}"
    result_file = str(job.get("result_file") or _result_file_path(job_id))
    if not os.path.exists(result_file):
        return f"任務已完成但找不到結果檔：{result_file}"
    try:
        with open(result_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            return f"任務已完成（job_id={job_id}），但結果為空。"
        return text
    except Exception as e:
        return f"讀取背景任務結果失敗（job_id={job_id}）：{e}"


def run_askcos_multistep_retrosynthesis_async_list_jobs(limit: int = 10) -> str:
    jobs = _list_jobs()
    n = max(1, int(limit or 10))
    if not jobs:
        return "目前沒有多步逆合成背景任務紀錄。"
    lines = ["最近多步逆合成背景任務："]
    for j in jobs[:n]:
        p = j.get("params", {}) if isinstance(j.get("params"), dict) else {}
        lines.append(
            f"- job_id={j.get('job_id','')} status={j.get('status','unknown')} "
            f"backend={j.get('backend','')} target={str(p.get('target_smiles',''))[:60]} "
            f"submitted_at={j.get('submitted_at','')}"
        )
    return "\n".join(lines)


def run_askcos_multistep_retrosynthesis_async_find(query: str = "", auto_result: bool = True) -> str:
    jobs = _list_jobs()
    if not jobs:
        return "目前沒有多步逆合成背景任務紀錄。"
    q = (query or "").strip()
    chosen = None
    if q:
        ql = q.lower()
        for j in jobs:
            p = j.get("params", {}) if isinstance(j.get("params"), dict) else {}
            text = " ".join(
                [
                    str(j.get("job_id", "")),
                    str(j.get("status", "")),
                    str(j.get("backend", "")),
                    str(p.get("target_smiles", "")),
                ]
            ).lower()
            if ql in text:
                chosen = j
                break
    if chosen is None:
        chosen = jobs[0]

    job_id = str(chosen.get("job_id", ""))
    status = str(chosen.get("status", "unknown"))
    if status == "done" and auto_result:
        return run_askcos_multistep_retrosynthesis_async_result(job_id=job_id)
    p = chosen.get("params", {}) if isinstance(chosen.get("params"), dict) else {}
    return (
        "找到最相關的背景任務：\n"
        f"- job_id: {job_id}\n"
        f"- status: {status}\n"
        f"- backend: {chosen.get('backend','')}\n"
        f"- target_smiles: {p.get('target_smiles','')}\n"
        "若要完整結果，可呼叫 run_askcos_multistep_retrosynthesis_async_result(job_id=...)。"
    )
