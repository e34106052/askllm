import json
import subprocess
from typing import List, Optional

# AskCOS 雜質預測服務的確切 URL
ASKCOS_IMPURITY_URL = "http://0.0.0.0:9691/impurity" 

def run_askcos_impurity_prediction(
    reactants_smiles: str, 
    product_smiles: Optional[str] = "", 
    solvent_smiles: Optional[str] = "", 
    reagent_smiles: Optional[str] = ""
) -> str:
    """
    執行 AskCOS 雜質預測模型，並解析實際返回的 'predict_expand' 結構。
    """
    
    if not reactants_smiles:
        return "錯誤：未提供有效的反應物 SMILES。"

    payload_data = {
        "rct_smi": reactants_smiles,
        "prd_smi": product_smiles,
        "sol_smi": solvent_smiles,
        "rea_smi": reagent_smiles
    }
    payload_str = json.dumps(payload_data)
    
    command = [
        "curl", ASKCOS_IMPURITY_URL,
        "--header", "Content-Type: application/json",
        "--request", "POST",
        "--data", payload_str
    ]
    
    try:
        print(f"--- 檢查點 2: Curl 命令執行 ---")
        
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=120 
        )
        
        data = json.loads(result.stdout)
        
        # 1. 检查状态
        if data.get("status") == "FAIL":
            error_msg = data.get("error", "服務端返回未知錯誤。")
            return f"AskCOS 雜質預測服務端執行失敗：{error_msg[:200]}..."
        
        # 2. 提取核心数据：'predict_expand' 列表 (包含主产物和杂质)
        expand_results = data.get("results", {}).get("predict_expand", [])
        
        if not expand_results:
            return "AskCOS 雜質預測調用成功，但在 'predict_expand' 中未找到任何結果。"
            
        # 3. 构造摘要
        summary_parts = []
        # 第 1 名通常是主要產物
        major_product_smiles = expand_results[0].get('prd_smiles', 'N/A')
        
        # 遍历所有结果，区分主要产物和杂质
        # 我们只列出前 5 名结果以保持摘要简洁
        limit = min(7, len(expand_results)) 

        for i in range(limit):
            item = expand_results[i]
            product_smiles = item.get('prd_smiles', 'N/A')
            mode_name = item.get('modes_name', 'N/A')
            avg_score = item.get('avg_insp_score', 0)
            
            # 判断是主要产物还是杂质
            type_label = "【主要產物】" if i == 0 else "【潛在雜質】"
            
            summary_parts.append(
                f"--- {type_label} 第 {i+1} 名 (得分: {avg_score:.4f}) ---"
                f"\n  - 產物 SMILES: {product_smiles}"
                f"\n  - 形成模式: {mode_name}"
            )
        
        # 4. 构造最终摘要
        final_summary = (
            f"AskCOS 雜質/副產物預測完成。共找到 {len(expand_results)} 條潛在結果。\n"
            f"以下是得分最高的前 {limit} 條結果分析:\n"
        )
        final_summary += "\n".join(summary_parts)
        
        # 附加主要产物信息
        final_summary += (
            f"\n--- 總結 ---\n"
            f"**預期主要產物 (No. 1)**: {major_product_smiles} "
            f"(形成模式: {expand_results[0].get('modes_name', 'N/A')})"
        )
        
        return final_summary
        
    except subprocess.CalledProcessError as e:
        error_output = e.stderr if e.stderr else f"Curl 退出代碼: {e.returncode}"
        return f"調用 AskCOS 雜質預測 API 失敗，錯誤詳情:\n{error_output}"
    except Exception as e:
        return f"發生未知錯誤或 JSON 解析失敗: {e}"