# AskLLM 研究版重建說明

## 專案定位

`askllm` 是以化學任務為核心的 Agent 系統，整合：

- LLM 決策（Gemini / Groq）
- AskCOS 工具適配層（retro / forward / condition / impurity / multistep）
- 技能化系統指令（`skills/*.md`）
- 持久化記憶（`.askllm_memory/memory_state.json`）
- 研究追蹤資料（`evidence_logs.jsonl`、`tool_trace_current_session.json`）

## 目前架構（對應重建目標）

- `ASKLLM.py`
  - entrypoint / wiring
  - adaptive planner（tool whitelist、compare gating、`max_tool_calls`）
  - planner on/off CLI 指令
  - evidence + tool trace 寫入
- `providers.py`
  - Groq chat client
  - `QuotaLimitError`
  - quota 訊息格式化
- `policies.py`
  - `is_tool_error`, `is_tool_empty`, `extract_top_score`
  - `looks_like_smiles`, SMILES/name candidate 提取
- `orchestrator.py`
  - `build_route_candidates`
  - `run_groq_turn`（Plan -> Act -> Observe -> Replan + A/B/C switch）

## 工具層能力

- 逆合成
  - `run_askcos_retrosynthesis`
  - `run_askcos_retrosynthesis_uspto_full`
  - `run_askcos_retrosynthesis_pistachio`
  - `run_askcos_retrosynthesis_template_enum`
  - `run_askcos_retrosynthesis_compare`
  - `run_askcos_multistep_retrosynthesis`（MCTS）
- 正向預測
  - `run_askcos_forward_prediction`
  - `run_askcos_forward_prediction_uspto_stereo`
  - `run_askcos_forward_prediction_graph2smiles`
  - `run_askcos_forward_prediction_wldn5`
  - `run_askcos_forward_prediction_compare`
- 條件預測
  - `run_askcos_condition_prediction`（GRAPH）
  - `run_askcos_quarc_prediction`（QUARC）
  - `run_askcos_condition_prediction_compare`
- 其他
  - `run_askcos_impurity_prediction`
  - `resolve_smiles_from_name`
  - `generate_molecule_image`

> `run_advanced_condition_prediction` 已保留 shim（`reaction_yield.py`）但不在 active tools。

## Planner 與研究欄位

- 預設策略：單工具優先；只有明確 compare 請求才放行 compare tool
- evidence log 主要欄位
  - `decision_provider`
  - `decision_model_for_turn`
  - `planner_output`
  - `plan_switch_logs`
  - `tool_call_count`
  - `tool_outputs_preview`
- tool trace 主要欄位
  - `tool_name`
  - `tool_args`
  - `raw_output`
  - `output_for_model`

## 記憶與指令

- 持久記憶檔：`.askllm_memory/memory_state.json`
- 支援欄位：`summary_zh`、`turns`、`current_topic`、`topics`、`meta_reflections`
- CLI 指令
  - `/memory show|clear|clear turns|clear topic|summary ...|ai on|off|status`
  - `/topic set|show|list`
  - `/planner on|off|status`

## API 入口

- `askcos_api.py` 保留作為 n8n/外部系統入口
- `POST /askllm`
  - request: `{"query": "...", "session_id": "optional"}`
  - response: `{"session_id": "...", "answer": "..."}`
  - quota 限制時回傳 `429` + `action` 指引

## 主要環境變數

- provider / model
  - `ASKLLM_DECISION_PROVIDER` (`gemini` or `groq`)
  - `ASKLLM_PRIMARY_MODEL`, `ASKLLM_BACKUP_MODEL`
  - `ASKLLM_GROQ_DECISION_MODEL`, `ASKLLM_GROQ_PLANNER_MODEL`, `ASKLLM_GROQ_AUX_MODEL`, `ASKLLM_GROQ_CRITIC_MODEL`
- planner / behavior
  - `ASKLLM_ENABLE_ADAPTIVE_POLICY`
  - `ASKLLM_PLANNER_TIMEOUT_SEC`
  - `ASKLLM_ENABLE_TOOL_OUTPUT_SUMMARY`
  - `ASKLLM_ENABLE_CRITIC`
- memory / cache
  - `ASKLLM_MEMORY_DIR`, `ASKLLM_MEMORY_DISABLE`
  - `ASKLLM_MEMORY_AI_SUMMARY`, `ASKLLM_MEMORY_MAX_TURNS`
  - `ASKLLM_CACHE_DIR`, `ASKLLM_CACHE_DISABLE`, `ASKLLM_CACHE_TTL_SEC`

## 驗證流程（重建版）

1. import/syntax
   - 在 `askcos_py311` 下可 import 核心模組
2. structure smoke test
   - 不依賴外部服務，驗證 planner/evidence/tool trace 骨架
3. integration smoke test
   - 在可用 Gemini/Groq/AskCOS 服務下跑最小 query
