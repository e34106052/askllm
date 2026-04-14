import argparse
import json
import os
from datetime import datetime, timezone

from multistep_retrosynthesis import (
    run_askcos_multistep_retrosynthesis,
    run_askcos_multistep_retrosynthesis_compare,
    run_askcos_multistep_retrosynthesis_retro_star,
)


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

    try:
        if backend == "retro_star":
            text = run_askcos_multistep_retrosynthesis_retro_star(**params)
        elif backend == "compare":
            # compare 不支援部分參數；先抽出相容欄位
            text = run_askcos_multistep_retrosynthesis_compare(
                target_smiles=params.get("target_smiles", ""),
                max_depth=int(params.get("max_depth", 6)),
                max_paths=int(params.get("max_paths", 5)),
                expansion_time=int(params.get("expansion_time", 180)),
                retro_model_name=str(params.get("retro_model_name", "reaxys")),
                use_cache=bool(params.get("use_cache", True)),
            )
        else:
            text = run_askcos_multistep_retrosynthesis(**params)

        result_file = str(job.get("result_file", ""))
        if result_file:
            os.makedirs(os.path.dirname(result_file), exist_ok=True)
            with open(result_file, "w", encoding="utf-8") as f:
                f.write(str(text))

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
