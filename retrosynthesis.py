import json
import subprocess
from typing import List
import sys

#  逆合成 URL 
ASKCOS_RETRO_URL = "http://0.0.0.0:9410/predictions/reaxys" 

def run_askcos_retrosynthesis(smiles_list: List[str], max_routes: int = 3) -> str:
    """
    使用 subprocess 調用 curl，執行 AskCOS reaxys 逆合成預測。
    此版本修正了覆雜的 JSON 列表解析問題，並匹配了最新的輸出結構。
    
    Args:
        smiles_list: 包含目標分子和（可選）上下文分子的 SMILES 字符串列表。
        max_routes: 最多返回的逆合成路徑數量 (默認為 3)。
                          
    Returns:
        AskCOS API 返回的預測路徑摘要。
    """
    target_molecule = smiles_list[0] if smiles_list else "N/A"
    print(f" 正在請求 AskCOS 逆合成服務分析: {target_molecule} (上下文分子數量: {len(smiles_list) - 1})")
    
    payload_data = {"smiles": smiles_list}
    payload_str = json.dumps(payload_data)
    
    command = [
        "curl", ASKCOS_RETRO_URL,
        "--header", "Content-Type: application/json",
        "--request", "POST",
        "--data", payload_str 
    ]
    
    try:
        print(f"--- 檢查點 1: Curl 命令執行 ---")
        print(f"  目標 URL: {ASKCOS_RETRO_URL}")
        
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,           
            check=True,          
            timeout=60          
        )
        
        print(f"  Curl 狀態: 成功 (stdout 長度: {len(result.stdout)})")
        
        # 2. 解析 JSON 字符串
        data = json.loads(result.stdout)
        
        # 3. 核心修正：獲取列表中的第一個數據對象
        if isinstance(data, list) and len(data) > 0:
            results_obj = data[0]
        else:
            return f"AskCOS 服務返回意外結構，無法解析預測結果。"
             
        # 檢查 AskCOS 是否返回了服務端錯誤 (例如 500 或 503 錯誤信息)
        if results_obj.get('code') in [500, 503]:
             return f"AskCOS 服務端錯誤：{results_obj.get('message', '預測失敗')}"
             
        # 4. 提取核心數據
        reactants = results_obj.get('reactants', [])
        scores = results_obj.get('scores', [])
        templates = results_obj.get('templates', []) 
        
        # 5. 檢查結果數量
        if not reactants or not scores or len(reactants) != len(scores):
            return f"AskCOS 逆合成調用成功，但前體/得分數據缺失或不匹配 (找到 {len(reactants)} 個前體，{len(scores)} 個得分)。"
            
        # 6. 提取前 N 條路徑 (摘要邏輯)
        limit = min(max_routes, len(reactants))
        summary_parts = []
        
        # 提取排名第一的模板作為參考
        top_template_smarts = templates[0].get('reaction_smarts', 'N/A') if templates else 'N/A'
        
        for i in range(limit):
            precursor_smiles = reactants[i]
            score = scores[i]
            
            precursors_list = precursor_smiles.split('.')
            precursors_formatted = ' / '.join(precursors_list)

            summary_parts.append(
                f"--- 第 {i+1} 名路徑 (得分: {score:.6f}) ---\n"
                f"  - **前體分子 ({len(precursors_list)} 個)**: {precursors_formatted}"
            )
        
        # 7. 構造最終摘要
        final_summary = (
            f"以下是 AskCOS 逆合成分析的結果。請以用戶可讀的中文總結以下路徑：\n"
            f"AskCOS 逆合成分析 (Reaxys 模型) 完成。共找到 {len(reactants)} 條路徑。\n"
            f"最高得分模板 (SMARTS): {top_template_smarts}\n"
            f"以下是您請求的前 {limit} 條路徑的詳細信息:\n"
        )
        final_summary += "\n".join(summary_parts)
        return final_summary
        
    except subprocess.CalledProcessError as e:
        error_output = e.stderr if e.stderr else f"Curl 退出代碼: {e.returncode}"
        return f"調用 AskCOS API 失敗，請檢查服務日志。錯誤詳情:\n{error_output}"
    except Exception as e:
        return f"發生未知錯誤或 JSON 解析失敗: {e}"