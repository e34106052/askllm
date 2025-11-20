import requests
import json
import subprocess
from typing import List

#  AskCOS 正向預測服務的確切 URL (使用 0.0.0.0:9510，匹配 Curl 成功的環境)
ASKCOS_FORWARD_URL = "http://0.0.0.0:9510/predictions/pistachio_23Q3" 

def run_askcos_forward_prediction(reactants_smiles_list: List[str], top_k: int = 3) -> str:
    """
    使用 subprocess 模塊調用 curl 命令，執行 AskCOS 正向預測模型。
    
    Args:
        reactants_smiles_list: 包含反應物 SMILES 字符串的列表。
        top_k: 最多返回的預測產物數量 (默認為 3)。
                          
    Returns:
        AskCOS API 返回的預測產物摘要。
    """
    full_reactants_smiles = ".".join(reactants_smiles_list)
    
    if not full_reactants_smiles or full_reactants_smiles == ".":
        return "未提供有效的反應物 SMILES。"

    print(f" 正在請求 AskCOS 正向預測服務分析反應物: {full_reactants_smiles[:40]}...")
    
    # 構造 Curl 命令的 payload，使用連接後的單個 SMILES 字符串
    payload_data = {"smiles": [full_reactants_smiles]} 
    payload_str = json.dumps(payload_data)
    # 構造 Curl 命令列表
    command = [
        "curl",
        ASKCOS_FORWARD_URL,
        "--header", "Content-Type: application/json",
        "--request", "POST",
        "--data", payload_str  # 使用編碼後的 JSON 字符串
    ]
    
    try:
        print(f"--- 檢查點 2: Curl 命令執行 ---")
        print(f"  目標 URL: {ASKCOS_FORWARD_URL}")
        
        # 1. 執行 Curl 命令
        result = subprocess.run(
            command,
            capture_output=True, # 捕獲輸出
            text=True,           # 將輸出解碼為字符串
            check=True,          # 如果返回非零狀態碼，則拋出異常
            timeout=60           # 設置超時
        )
        
        # 2. 檢查 Curl 命令狀態
        print(f"  Curl 狀態: 成功 (stdout 長度: {len(result.stdout)})")
        
        # 3. 解析 JSON 字符串
        data = json.loads(result.stdout)
        
        # ... (後續的解析邏輯與之前 requests 成功時一致)
        
        # 4. 檢查返回結構
        if not data or not isinstance(data, list) or not data[0]:
            return "AskCOS 服務返回空或格式不正確的列表。"
             
        results_obj = data[0]
        products = results_obj.get('products', [])
        scores = results_obj.get('scores', [])
        
        # 5. 檢查結果數量和打印原始數據 (DEBUG)
        print(f"--- 檢查點 3: 數據解析 ---")
        print(f"  解析結果數量: {len(products)} 種產物")
        print(f"  原始產物列表 (Top 5): {products[:5]}...")
        print(f"  原始分數列表 (Top 5): {scores[:5]}...")
        
        if not products or not scores or len(products) != len(scores):
            return "AskCOS 正向預測調用成功，但產物或得分數據缺失/不匹配。"
            
        # 6. 提取前 Top-K 產物 (摘要邏輯不變)
        limit = min(top_k, len(products))
        summary_parts = []
        
        for i in range(limit):
            product = products[i] if products[i] else "N/A"
            score = scores[i] if scores[i] else 0.0
            
            summary_part = (
                f"--- 第 {i+1} 名預測 (得分: {score:.4f}) ---\n"
                f"  - 產物 SMILES: {product}\n"
                f"  - 完整反應 SMILES: {full_reactants_smiles}>>{product}"
            )
            summary_parts.append(summary_part)
        
        # 7. 構造最終摘要
        final_summary = (
            f"AskCOS 正向預測完成。共找到 {len(products)} 種潛在產物。\n"
            f"以下是您請求的前 {limit} 名預測結果:\n"
        )
        final_summary += "\n".join(summary_parts)
        return final_summary
        
    except subprocess.CalledProcessError as e:
        # Curl 命令失敗 (例如 4xx/5xx HTTP 錯誤或 curl 本身失敗)
        return f"調用 AskCOS API 失敗，Curl 退出代碼: {e.returncode}。錯誤詳情:\n{e.stderr}"
    except subprocess.TimeoutExpired:
        return "調用 AskCOS API 失敗，Curl 命令執行超時。"
    except FileNotFoundError:
        return "錯誤: 找不到 'curl' 命令。請確保它已安裝在系統 PATH 中。"
    except json.JSONDecodeError:
        return f"Curl 成功執行，但返回的不是有效的 JSON 格式。原始輸出: {result.stdout[:200]}..."
    except Exception as e:
        return f"發生未知錯誤: {e}"