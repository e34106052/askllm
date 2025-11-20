import requests
import json
from typing import Optional
from urllib.parse import quote # 用於 URL 編碼

PUBCHEM_API_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"

def resolve_smiles_from_name(compound_name: str) -> str:
    """
    使用 PubChem PUG-REST API 將化合物名稱轉換為 SMILES 字符串。
    
    Args:
        compound_name: 化合物名稱（例如 "CO2", "Aspirin"）。
                          
    Returns:
        如果成功，返回 SMILES 字符串；否則返回錯誤信息。
    """
    if not compound_name:
        return "錯誤：未提供化合物名稱。"
    
    # 構建 API URL：/compound/name/[name]/property/SMILES/JSON
    # 使用 quote 對名稱進行 URL 編碼
    encoded_name = quote(compound_name)
    
    url = f"{PUBCHEM_API_BASE}/name/{encoded_name}/property/SMILES/JSON"
    
    print(f" 正在請求 PubChem API 解析化合物名稱: {compound_name}...")
    
    try:
        # 發送請求
        response = requests.get(url, timeout=10)
        response.raise_for_status() # 檢查 HTTP 錯誤
        
        data = response.json()
        
        # 解析 PubChem API 響應
        properties = data.get("PropertyTable", {}).get("Properties", [])
        
        if properties:
            # 提取 Canonical SMILES
            smiles = properties[0].get("SMILES")
            if smiles:
                print(f"  -> 成功解析 SMILES: {smiles}")
                return f"化合物 '{compound_name}' 的 SMILES 字符串是: {smiles}"
            
        # 檢查是否有 PubChem API 錯誤信息
        if data.get("Fault"):
            message = data["Fault"]["Message"]
            return f"PubChem API 錯誤：{message}"
            
        return f"PubChem API 成功響應，但未找到 SMILES 屬性。"
        
    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 404:
            return f"化合物 '{compound_name}' 在 PubChem 數據庫中未找到 (404 Error)。"
        return f"HTTP 錯誤：{http_err} (狀態碼: {response.status_code})"
    except requests.exceptions.ConnectionError:
        return "連接錯誤：無法連接到 PubChem API。"
    except requests.exceptions.Timeout:
        return "請求超時：連接到 PubChem API 超時。"
    except Exception as e:
        return f"發生未知錯誤: {e}"