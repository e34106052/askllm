# AskLLM 化學 Agent（研究版）

`askllm` 是一個面向化學任務的 Agent 系統，整合 LLM 決策、AskCOS 工具鏈、持久記憶與可追蹤研究日誌。  
目前重點支援：逆合成（含多步）、正向預測、條件/雜質預測、以及多 critic 路線推薦與回饋閉環。

## 專案定位

- **決策層**：Gemini / Groq 雙路徑，支援 planner、critic、工具摘要。
- **工具層**：AskCOS API wrappers（retro/forward/condition/impurity/multistep）。
- **規則層**：skills 指令片段、constraint-aware 路線評估、風險規則（含 PubChem）。
- **資料層**：cache、memory、evidence/tool trace、route eval/feedback logs。

## 目錄與模組職責

- `ASKLLM.py`：主入口、模型路由、adaptive planner、tool dispatch、CLI 指令。
- `orchestrator.py`：Groq 路徑下的 Plan-Act-Observe-Replan（A/B/C 切換）。
- `policies.py`：SMILES/名稱抽取、錯誤判斷、分數提取、有效證據判斷。
- `providers.py`：Groq client、quota 例外抽象。
- `askcos_api.py`：Flask API 入口（`POST /askllm`）。
- 工具模組：
  - `retrosynthesis.py`（單步逆合成）
  - `multistep_retrosynthesis.py`（多步 + async job）
  - `multistep_async_runner.py`（背景任務 worker）
  - `forward_prediction.py`
  - `condition_prediction.py`
  - `context_quarc.py`
  - `impurity_prediction.py`
  - `route_recommendation.py`（多 critic 評估、constraint loop、feedback）
- 基礎設施：
  - `cache_utils.py`
  - `persistent_memory.py`
  - `askcos_tree_utils.py`
- 規則/提示：
  - `skills/*.md`
  - `risk_rules.json`

## 核心執行流程

1. `run_interactive_agent()` 載入記憶與 skills，建立系統指令。
2. `adaptive plan` 決定候選工具與 `max_tool_calls`。
3. 依 `DECISION_PROVIDER` 分流：
   - **Groq**：`orchestrator.run_groq_turn()`（A/B/C + replan）
   - **Gemini**：function-calling 多輪工具執行
4. 寫入 `evidence_logs.jsonl` 與 `tool_trace_current_session.json`。
5. 更新 `memory_state.json`（turns/topic/summary/reflection）。

## 工具能力總覽

### 逆合成

- `run_askcos_retrosynthesis`
- `run_askcos_retrosynthesis_uspto_full`
- `run_askcos_retrosynthesis_pistachio`
- `run_askcos_retrosynthesis_template_enum`
- `run_askcos_retrosynthesis_compare`
- `run_askcos_multistep_retrosynthesis`
- `run_askcos_multistep_retrosynthesis_retro_star`
- `run_askcos_multistep_retrosynthesis_compare`

### 多步背景任務（async）

- `run_askcos_multistep_retrosynthesis_async_submit`
- `run_askcos_multistep_retrosynthesis_async_status`
- `run_askcos_multistep_retrosynthesis_async_result`
- `run_askcos_multistep_retrosynthesis_async_list_jobs`
- `run_askcos_multistep_retrosynthesis_async_find`

> `async_submit` 支援 `auto_analyze`。主任務完成後可自動跑四 critic 路線分析。

### 正向 / 條件 / 雜質

- `run_askcos_forward_prediction`
- `run_askcos_forward_prediction_uspto_stereo`
- `run_askcos_forward_prediction_graph2smiles`
- `run_askcos_forward_prediction_wldn5`
- `run_askcos_forward_prediction_compare`
- `run_askcos_condition_prediction`
- `run_askcos_quarc_prediction`
- `run_askcos_condition_prediction_compare`
- `run_askcos_impurity_prediction`

### 路線推薦與回饋

- `run_askcos_route_recommendation`
- `run_askcos_route_recommendation_recent_logs`
- `run_askcos_route_recommendation_feedback`

## Route Recommendation（重點）

`run_askcos_route_recommendation` 提供多 agent 評估與可解釋輸出：

- **四 critic 分數**：`cost/success/safety/supply`
- **投票與低共識標記**：`critic_votes`、`disagreement_std`、`contested`
- **constraint-aware**：
  - `hard/soft` 限制
  - 違規檢查（hard reject / soft penalty）
  - 放寬建議（relaxation suggestions）
- **安全規則**：
  - token 規則（`risk_rules.json`）
  - PubChem hazard（含高危 H-code 依據）

### 重要模式參數

- `constraint_parse_mode`: `hybrid | rule_only | llm_only`
- `strict_safety_mode`: 嚴格安全 gate（預設 `True`）
- `exploration_mode`: 無解時允許第二層放寬（token hard -> soft penalty）
- `auto_relax_if_infeasible`: constraints 無解時自動放寬重跑

## 多步 Async + 自動分析

- `async_submit` 會建立 `job_id` 與狀態檔。
- `status` 可看 `status` 與 `analysis_status`。
- `result` 會回：
  - 主任務多步結果
  - 若 `auto_analyze=True`，附上 `=== 自動四 critic 分析 ===`
- `find` 可用自然語言找最近/最相關 job（不一定要手打 `job_id`）。

## 執行產物與日誌

- Cache
  - `.askllm_cache/`
- 記憶/證據
  - `.askllm_memory/memory_state.json`
  - `.askllm_memory/evidence_logs.jsonl`
  - `.askllm_memory/tool_trace_current_session.json`
- Route 推薦閉環
  - `runtime_jobs/route_eval_logs.jsonl`
  - `runtime_jobs/route_feedback_logs.jsonl`
  - `runtime_jobs/constraint_loop_logs.jsonl`
- 多步 async
  - `runtime_jobs/multistep/<job_id>.json`
  - `runtime_jobs/multistep/<job_id>.result.txt`
  - `runtime_jobs/multistep/<job_id>.analysis.txt`

## CLI 指令

- `/memory show|clear|clear turns|clear topic|summary ...|ai on|off|status`
- `/topic set|show|list`
- `/planner on|off|status`

## API 入口

- `askcos_api.py`
- `POST /askllm`
  - request: `{"query": "...", "session_id": "optional"}`
  - response: `{"session_id": "...", "answer": "..."}`
  - quota 限制時回 `429`

## 主要環境變數

- provider / model
  - `ASKLLM_DECISION_PROVIDER` (`gemini` / `groq`)
  - `ASKLLM_PRIMARY_MODEL`, `ASKLLM_BACKUP_MODEL`
  - `ASKLLM_GROQ_DECISION_MODEL`, `ASKLLM_GROQ_PLANNER_MODEL`
  - `ASKLLM_GROQ_AUX_MODEL`, `ASKLLM_GROQ_CRITIC_MODEL`
- planner / behavior
  - `ASKLLM_ENABLE_ADAPTIVE_POLICY`
  - `ASKLLM_PLANNER_TIMEOUT_SEC`
  - `ASKLLM_ENABLE_TOOL_OUTPUT_SUMMARY`
  - `ASKLLM_ENABLE_CRITIC`
- memory / cache
  - `ASKLLM_MEMORY_DIR`, `ASKLLM_MEMORY_DISABLE`
  - `ASKLLM_MEMORY_AI_SUMMARY`, `ASKLLM_MEMORY_MAX_TURNS`
  - `ASKLLM_CACHE_DIR`, `ASKLLM_CACHE_DISABLE`, `ASKLLM_CACHE_TTL_SEC`

## 已知限制

- 多數工具透過 `curl + subprocess`，統一 retry/backoff 還可加強。
- 本機端點與埠號依部署環境而異，需留意設定一致性。
- JSON/JSONL 寫檔目前未做 file lock，高併發下可能有競態風險。
- 安全評估屬工程啟發式，非正式法規合規判定工具。

## 快速驗證建議

1. `python -m py_compile ASKLLM.py route_recommendation.py multistep_retrosynthesis.py`
2. 跑一筆 `run_askcos_route_recommendation`，確認輸出有 `eval_id`
3. 查 `run_askcos_route_recommendation_recent_logs(limit=3)`
4. 寫回 `run_askcos_route_recommendation_feedback(...)`
5. 提交一筆 `async_submit` 並用 `status/result/find` 驗證背景流程
