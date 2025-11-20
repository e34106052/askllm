# image_tool.py (或者添加到 test.py/test2.py)

from rdkit import Chem
from rdkit.Chem.Draw import MolToFile
import os
import base64
from io import BytesIO

# 定義一個存儲生成圖片的目錄
IMAGE_DIR = "generated_mol_images"
os.makedirs(IMAGE_DIR, exist_ok=True) # 確保目錄存在

def generate_molecule_image(smiles: str, file_prefix: str = "molecule") -> str:
    """
    Args:
        smiles: 目標分子的 SMILES 字符串。
        file_prefix: 生成圖片的文件名前綴。
        
    Returns:
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return f"錯誤：無法從SMILES '{smiles}' 解析分子。"
        
        # 定義一個唯一的文件名
        image_filename = f"{file_prefix}_{abs(hash(smiles))}.png"
        image_path = os.path.join(IMAGE_DIR, image_filename)
        
        # 生成並保存圖片
        MolToFile(mol, image_path, size=(300, 300))
        # 或者在實際的網頁應用中，您會直接返回圖片路徑
        return f"圖片已生成並保存到 '{image_path}'"
        
    except Exception as e:
        return f"生成分子圖片失敗: {e}"

# 注意：如果您只是想在終端看到圖片路徑，可以修改返回值為 image_path
# 但如果您要集成到 Web UI (例如 Gradio/Streamlit)，Base64 是更好的選擇。