import argparse
import json
import os
from datetime import datetime, timezone

from multistep_retrosynthesis import (
    run_askcos_multistep_retrosynthesis,
    run_askcos_multistep_retrosynthesis_compare,
    run_askcos_multistep_retrosynthesis_retro_star,
)
from route_recommendation import run_askcos_route_recommendation


ASYNC_JOBS_DIR = os.path.join(os.path.dirname(__file__), "runtime_jobs", "multistep")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_file_path(job_id: str) -> str:
    return os.path.join(ASYNC_JOBS_DIR, f"{job_id}.json")


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
    os.makedirs(ASYNC_JOBS_DIR, exist_ok=True)
    with open(_job_file_path(job_id), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()

    job_id = args.job_id
    job = _read_job(job_id)
    if not job:
        return

    params = job.get("params", {}) if isinstance(job.get("params"), dict) else {}
    backend = str(job.get("backend", "mcts")).lower()

    job["status"] = "running"
    job["started_at"] = job.get("started_at") or _utc_now()
    _write_job(job_id, job)

    tool_kwargs = {
        "target_smiles": str(params.get("target_smiles", "")),
        "max_depth": int(params.get("max_depth", 8)),
        "max_paths": int(params.get("max_paths", 200)),
        "expansion_time": int(params.get("expansion_time", 300)),
        "max_branching": int(params.get("max_branching", 25)),
        "retro_model_name": str(params.get("retro_model_name", "reaxys")),
        "max_num_templates": int(params.get("max_num_templates", 200)),
        "top_k": int(params.get("top_k", 20)),
        "threshold": float(params.get("threshold", 0.15)),
        "sorting_metric": str(params.get("sorting_metric", "plausibility")),
        "use_cache": bool(params.get("use_cache", True)),
    }

    try:
        if backend == "retro_star":
            text = run_askcos_multistep_retrosynthesis_retro_star(**tool_kwargs)
        elif backend == "compare":
            # compare 不支援部分參數；先抽出相容欄位
            text = run_askcos_multistep_retrosynthesis_compare(
                target_smiles=tool_kwargs["target_smiles"],
                max_depth=tool_kwargs["max_depth"],
                max_paths=tool_kwargs["max_paths"],
                expansion_time=tool_kwargs["expansion_time"],
                retro_model_name=tool_kwargs["retro_model_name"],
                use_cache=tool_kwargs["use_cache"],
            )
        else:
            text = run_askcos_multistep_retrosynthesis(**tool_kwargs)

        result_file = str(job.get("result_file", ""))
        if result_file:
            os.makedirs(os.path.dirname(result_file), exist_ok=True)
            with open(result_file, "w", encoding="utf-8") as f:
                f.write(str(text))

        auto_analyze = bool(params.get("auto_analyze", True))
        if auto_analyze:
            job["analysis_status"] = "running"
            _write_job(job_id, job)
            try:
                analyze_text = run_askcos_route_recommendation(
                    target_smiles=str(params.get("target_smiles", "")),
                    objective=str(params.get("analyze_objective", "balanced")),
                    backend="mcts",
                    max_depth=int(params.get("max_depth", 8)),
                    max_paths=int(params.get("max_paths", 200)),
                    expansion_time=int(params.get("expansion_time", 300)),
                    top_n=int(params.get("analyze_top_n", 10)),
                    enable_pubchem_hazard=True,
                    use_cache=bool(params.get("use_cache", True)),
                )
                analysis_file = str(job.get("analysis_file", ""))
                if analysis_file:
                    os.makedirs(os.path.dirname(analysis_file), exist_ok=True)
                    with open(analysis_file, "w", encoding="utf-8") as af:
                        af.write(str(analyze_text))
                job["analysis_status"] = "done"
                job["analysis_error"] = ""
            except Exception as ae:
                job["analysis_status"] = "failed"
                job["analysis_error"] = str(ae)[:500]
        else:
            job["analysis_status"] = "skipped"

        job["status"] = "done"
        job["ended_at"] = _utc_now()
        job["error"] = ""
        _write_job(job_id, job)
    except Exception as e:
        job["status"] = "failed"
        job["ended_at"] = _utc_now()
        job["error"] = str(e)[:500]
        _write_job(job_id, job)


if __name__ == "__main__":
    main()
