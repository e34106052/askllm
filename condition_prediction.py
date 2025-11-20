import json
import subprocess
from typing import List, Optional

# AskCOS 反應條件預測服務的 URL
ASKCOS_CONDITION_URL = "http://0.0.0.0:9901/api/v2/condition/GRAPH" 

def run_askcos_condition_prediction(
    reaction_smiles: str, 
    reagents: Optional[List[str]] = None, 
    n_conditions: int = 5
) -> str:
    """
    使用 subprocess 調用 curl，執行 AskCOS 反應條件預測，並解析實際 JSON 結構。
    
    Args:
        reaction_smiles: 包含反應物和產物 SMILES 的字符串 (格式: RCTS>>PRD)。
        reagents: 反應中已知的試劑 SMILES 列表 (可選)。
        n_conditions: 最多返回的預測條件數量 (默認為 5)。
                          
    Returns:
        AskCOS API 返回的預測條件摘要。
    """
    
    if not reaction_smiles or ">>" not in reaction_smiles:
        return "錯誤：未提供有效的反應 SMILES (格式應為 RCTS>>PRD)。"

    print(f" 正在請求 AskCOS 條件預測服務分析反應: {reaction_smiles[:60]}")
    
    # 構造 Curl 命令的 payload (保持不變)
    payload_data = {
        "smiles": reaction_smiles,
        "reagents": reagents if reagents is not None else [],
        "n_conditions": n_conditions
    }
    payload_str = json.dumps(payload_data)
    
    command = [
        "curl", ASKCOS_CONDITION_URL,
        "--header", "Content-Type: application/json",
        "--request", "POST",
        "--data", payload_str 
    ]
    
    try:
        print(f"--- 檢查點 1: Curl 命令執行 ---")
        
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,           
            check=True,          
            timeout=90          
        )
        
        print(f"  Curl 狀態: 成功 (stdout 長度: {len(result.stdout)})")
        
        # 2. 解析 JSON 字符串 (結果是一個列表)
        data = json.loads(result.stdout)
        
        # 3. 提取核心數據
        # 實際返回的 JSON 根是一個列表
        conditions = data
        
        if not conditions or not isinstance(conditions, list):
            return "AskCOS 條件預測調用成功，但未找到推薦反應條件列表或格式不正確。"
             
        # 4. 提取前 N 條路徑 (摘要邏輯)
        limit = min(n_conditions, len(conditions))
        summary_parts = []
        
        for i in range(limit):
            cond = conditions[i]
            score = cond.get('score', 0.0)
            temp_k = cond.get('temperature', 0.0)
            
            # 提取詳細條件 (Agents 列表)
            agents_list = cond.get('agents', [])
            
            reagents_display = []
            solvents_display = []
            other_agents_display = []
            
            # 解析 agents 列表，區分 REACTANT, SOLVENT, REAGENT
            for agent in agents_list:
                smi = agent.get('smi_or_name', 'N/A')
                role = agent.get('role')
                amt = agent.get('amt', 0.0)
                
                # 假設未被標記為 REACTANT 的 SMILES/Name 是溶劑、催化劑或試劑
                if role == "REACTANT":
                    # 反應物跳過，因為用戶已經知道
                    continue
                elif smi == reaction_smiles.split('>>')[0].split('.')[0] or smi == reaction_smiles.split('>>')[0].split('.')[-1]:
                    # 確保跳過反應物
                    continue
                
                # 根據 SMILES 規則判斷常見的溶劑 (簡化處理)
                if len(smi) > 1 and smi[0].islower() and 'O' in smi or 'C' in smi and '=' not in smi:
                    solvents_display.append(f"{smi} ({amt:.2f})")
                else:
                    reagents_display.append(f"{smi} ({amt:.2f})")
            
            # 格式化輸出
            summary_parts.append(
                f"--- 第 {i+1} 名條件 (得分: {score:.4f}) ---\n"
                f"  - **溫度** (Temperature): {temp_k:.1f} K ({temp_k - 273.15:.1f} °C)\n"
                f"  - **溶劑** (Solvents): {'; '.join(solvents_display) if solvents_display else '無特定溶劑'}\n"
                f"  - **試劑/催化劑** (Reagents/Catalyst): {'; '.join(reagents_display) if reagents_display else '無額外試劑'}"
            )
        
        # 5. 構造最終摘要
        final_summary = (
            #  增加指令前綴，明確告訴 Gemini 總結結果
            f"以下是 AskCOS 反應條件預測的結果。請以用戶可讀的中文總結以下條件，並建議最優條件：\n" 
            f"AskCOS 反應條件預測 (GRAPH 模型) 完成。共找到 {len(conditions)} 種可行條件。\n"
            f"以下是您請求的前 {limit} 名條件的詳細信息:\n"
        )
        final_summary += "\n".join(summary_parts)
        return final_summary
        
    except subprocess.CalledProcessError as e:
        error_output = e.stderr if e.stderr else f"Curl 退出代碼: {e.returncode}"
        return f"調用 AskCOS API 失敗，請檢查服務是否運行在 9901 端口。錯誤詳情:\n{error_output}"
    except Exception as e:
        return f"發生未知錯誤或 JSON 解析失敗: {e}"