# AskLLM Recovered Progress

本文件根據對話紀錄與目前工作目錄重建 `askllm` 的開發進度，目標是回答三件事：

1. 你已經做過哪些功能
2. 哪些功能目前在 repo 中仍存在
3. 哪些功能曾存在，但 source 已遺失，只剩對話紀錄或 `pyc` 痕跡

## 一、最早期已正式進 git 的功能

以下檔案在 repo 初始提交時已存在：

- `ASKLLM.py`
- `condition_prediction.py`
- `forward_prediction.py`
- `impurity_prediction.py`
- `retrosynthesis.py`
- `smiles_resolver.py`

這一階段的 `askllm` 是較早期版本，主要特徵：

- Gemini function calling
- 基本多步工具鏈
- 名稱轉 SMILES
- AskCOS 工具調用
- 簡單的 CLI 互動與 Flask API 包裝

## 二、後續已做過的功能演進

以下內容可從對話紀錄重建，代表它們曾經被實作過：

### 1. Skill 化與模組化 system instruction

曾建立：

- `skills/chemistry_assistant_skill.md`
- `skills/00-core.md`
- `skills/01-language.md`
- `skills/02-ambiguity.md`
- `skills/03-name-resolution.md`
- `skills/04-condition-priority.md`
- `skills/05-task-routing.md`
- `skills/06-formatting.md`
- `skills/07-completion.md`
- `skills/08-cmem.md`

曾修改 `ASKLLM.py` 以支援：

- `SKILLS_DIR`
- `SKILL_FALLBACK_FILE`
- `BASE_SKILL_FILES`
- `CONDITIONAL_SKILL_RULES`
- `load_system_instruction_from_skill(user_prompt)`
- 後續還打算做 Hybrid skill routing

### 2. 持久化記憶（cmem / persistent memory）

曾建立：

- `persistent_memory.py`

曾做到：

- `.askllm_memory/memory_state.json`
- `summary_zh`
- `turns`
- 啟動時還原記憶到 `chat_history`
- 每輪寫回磁碟
- `/memory show`
- `/memory clear`
- `/memory summary ...`

之後還往更進一步方向做：

- topic/task slots
- `current_topic`
- `topics`
- `meta_reflections`
- `/topic` 類指令
- 反思記錄與自我評估

### 3. 多步逆合成與多工具比較

對話紀錄顯示後續 `askllm` 不再只是單步 one-step retrosynthesis，而是逐步加入：

- 多引擎 retrosynthesis compare
- 多引擎 forward compare
- 條件 compare
- `run_askcos_multistep_retrosynthesis`
- 預設單工具，只有明確要求比較才 compare

### 4. Adaptive Planner / Phase 2 MVP

後來已進入你定義的第 2 階段 MVP，並曾在 `ASKLLM.py` 內實作：

- Planner JSON
- Adaptive Tool Loop
- 工具白名單與每輪工具子集
- `max_tool_calls`
- `.askllm_memory/evidence_logs.jsonl`
- `.askllm_memory/tool_trace_current_session.json`

已做過的能力包含：

- 每輪先規劃再執行
- 有 tool budget
- 有 evidence log
- planner on/off 測試
- compare 不是預設常開

### 5. 進一步研究版 Planner

後續又從「單檔 adaptive planner」演進到較完整研究版，根據對話紀錄曾出現：

- `providers.py`
- `policies.py`
- `orchestrator.py`
- `QuotaLimitError`
- `run_groq_turn(...)`
- Groq decision path
- Plan -> Act -> Observe -> Replan
- A/B/C strategy switching
- `plan_contracts`
- `switch_reason_enum`
- `evidence_gain`
- `compact_abandoned_routes`

這些是較晚期的研究版 agent 能力，不屬於最早期 git 版本。

### 6. 多 provider / Groq 整合

後續曾做過：

- Groq OpenAI-compatible API 接入
- Gemini / Groq 分工
- quota limit 類型錯誤拋出
- `429` 以 raise error 回傳給使用者
- decision model / planner model / critic model 分開

### 7. 條件預測路線調整

後續曾做過這些調整：

- 加回 `run_askcos_condition_prediction` 作為 fallback
- 曾把 `run_advanced_condition_prediction` 放進優先路線
- 之後又決定將 advanced two-stage 從 active tools 移除

### 8. 專案整理與搬移

後續曾做過：

- 將非 runtime 需要的檔案移到 `/home/ryan/storage`
- 保留 `askcos_api.py` 作為之後接 `n8n` 的 API 入口
- 更新 `README.md` 反映 planner、API、memory、研究功能

## 三、目前工作目錄還能直接看到的內容

目前 repo 內可見且可直接使用的 source：

- `ASKLLM.py`
- `askcos_api.py`
- `condition_prediction.py`
- `forward_prediction.py`
- `image_tool.py`
- `impurity_prediction.py`
- `retrosynthesis.py`
- `smiles_resolver.py`
- `config.py`
- `skills/*.md`（本次依對話紀錄重建）

目前已驗證：

- `ASKLLM` 可在 `askcos_py311` 環境 import
- `askcos_api.py` 可在 `askcos_py311` 環境 import

## 四、目前已遺失但幾乎確定存在過的 source

以下功能現在 repo 看不到原始碼，但從對話紀錄與 `__pycache__` 幾乎可確定它們曾存在過：

- `persistent_memory.py`
- `orchestrator.py`
- `providers.py`
- `policies.py`
- 可能還有：
  - `cache_utils.py`
  - `context_quarc.py`
  - `multistep_retrosynthesis.py`
  - `reaction_yield.py`

判斷依據：

- 對話紀錄曾明確描述建立或修改
- `__pycache__` 中仍有對應 `.cpython-311.pyc`

## 五、為什麼會突然不見

目前最合理推論：

- 早期基礎檔案是 git tracked，所以可從 `HEAD` 還原
- 後續研究版功能很多是在本地做開發，但沒有正式 commit
- 這些 source 後來從工作目錄消失
- 因為 git 沒追蹤，所以無法直接從版本庫救回
- `pyc` 只能證明它們曾存在，不能直接視為可靠 source

目前沒有找到明確的：

- `git reset --hard`
- branch 切換
- rebase
- 明確的 `rm` / `mv` 終端機指令

所以更像是本地 source 被移除或覆蓋，而不是 git 歷史主動刪除。

## 六、目前最接近的真實狀態

若要描述你的 `askllm` 真正進度，最接近的說法是：

- 你不是只有一個簡單 Gemini tool-calling demo
- 你其實已經做過：
  - skill-driven prompt system
  - persistent memory / cmem
  - topic/reflection memory extension
  - multi-tool compare
  - multistep retrosynthesis
  - adaptive planner MVP
  - evidence logging
  - Groq integration
  - research-grade replan / A-B-C strategy architecture

但目前 source tree 只保留了較早期的一部分，需要依對話紀錄逐步補回。

## 七、建議恢復順序

若要把專案恢復到你最後研究版的狀態，建議順序：

1. 先恢復 runtime 必需骨架
   - `ASKLLM.py`
   - 工具模組
   - `askcos_api.py`
   - `skills/`

2. 再恢復記憶層
   - `persistent_memory.py`
   - `.askllm_memory` 讀寫邏輯
   - `/memory`、`/topic`

3. 再恢復 planner MVP
   - planner JSON
   - tool budget
   - `evidence_logs.jsonl`
   - `tool_trace_current_session.json`

4. 最後恢復研究版 orchestrator
   - `providers.py`
   - `policies.py`
   - `orchestrator.py`
   - Groq / Replan / A-B-C

## 八、這份文件的用途

這份文件不是「完整 source 替代品」，而是之後重建 source 時的路標。

如果要繼續補回完整 askllm，下一步應該依這份文件逐項恢復缺失模組，而不是再從零猜現在專案應該長怎樣。
