## 概述 (Overview)

AskCOS 化學 Agent 是一個結合了 Google Gemini 模型和 AskCOS 化學服務的智能應用程序。它能夠解析用戶的化學查詢（包括化合物名稱），自動執行逆合成、正向預測、雜質預測、反應條件預測，以及分子結構圖生成等複雜的化學任務，並以清晰的中文摘要形式返回結果。

本 Agent 的核心優勢在於能夠進行**多步工具調用**，例如：
1.  將**中文名稱**翻譯成**英文**。
2.  調用 `resolve_smiles_from_name` 將**英文名稱**轉換為 **SMILES**。
3.  使用 **SMILES** 調用 **AskCOS** 核心服務。

##  核心功能 (Features)

| 功能模塊 | 說明 | 對應工具 |
| :--- | :--- | :--- |
| **逆合成預測 (Retrosynthesis)** | 根據目標分子 SMILES，預測可能的合成前體和反應路徑。 | `run_askcos_retrosynthesis_outcomes` |
| **正向預測 (Forward Prediction)** | 根據反應物 SMILES，預測可能的產物和反應得分。 | `run_askcos_forward_prediction` |
| **雜質/副產物預測 (Impurity)** | 預測反應可能產生的雜質或副產物，並分析其形成模式。 | `run_askcos_impurity_prediction` |
| **反應條件預測 (Condition)** | 根據反應物與產物 SMILES，推薦合適的溶劑、試劑和溫度。 | `run_askcos_condition_prediction` |
| **分子結構圖生成 (Image)** | 將 SMILES 字符串轉換為可視化的 PNG 結構圖。 | `generate_molecule_image` |
| **SMILES 解析 (Resolver)** | 通過 PubChem API 將化學名稱轉換為標準 SMILES 字符串。 | `resolve_smiles_from_name` |

##  環境配置與安裝 (Installation)

### 1. 依賴項

本 Agent 需要 Python 依賴庫以及運行在本地的 AskCOS 服務和 `curl` 命令。

```bash
# Python 依賴
pip install google-genai requests rdkit-pypi
```

### 2. AskCOS 服務 (本地化學計算引擎)
**重要：** 本 Agent 假定您已經在本地運行了 AskCOS 的相關 Docker 服務，並且它們監聽在以下指定端口：
|**服務** |**端口** |**對應文件中的 URL** |
|:--|
|**逆合成 (Retro)** |`9410` |`http://0.0.0.0:9410/predictions/reaxys` |
|**正向預測 (Forward)** |`9510` |`http://0.0.0.0:9510/predictions/pistachio_23Q3` |
|**雜質預測 (Impurity)** |`9691` |`http://0.0.0.0:9691/impurity` |
|**條件預測 (Condition)** |`9901` |`http://0.0.0.0:9901/api/v2/condition/GRAPH` |

### 3. API Key 配置
請在項目根目錄下創建一個名為 `config.py` 的文件，並填入您的 Gemini API Key。
**`config.py` 內容**
```python
GEMINI_API_KEY = "您的實際 Gemini API Key"
```
### 4. 運行程序
執行主程序進入交互式模式：
```bash
python ASKLLM.py
```
##  使用範例 (Usage Examples)
進入交互模式後，Agent 會自動選擇並鏈接工具來回答您的查詢。

### 1. 逆合成查詢 (中文名稱 -> SMILES -> Retro)

**用戶輸入：**
>乙酰水楊酸的逆合成路徑是什麼？

**Agent 流程：**
1. **翻譯**：`乙酰水楊酸` -> `Aspirin`
2. **解析**：調用 `resolve_smiles_from_name("Aspirin")` 得到 `CC(=O)Oc1ccccc1C(=O)O`。
3. **逆合成**：調用 `run_askcos_retrosynthesis_outcomes` 進行預測。
4. **總結**：生成中文結果。


### 2. 正向預測 (SMILES)


**用戶輸入：**
>C1CCOC1.O=C(Cl)C(Cl)Cl 的反應產物是什麼？

**Agent 流程：**
1. **正向預測**：調用 `run_askcos_forward_prediction`。
2. **總結**：生成預測產物列表和得分。


### 3. 反應條件預測 (Condition)


**用戶輸入：**
>對於反應 CCC=O>>CCC(O)C(C)O，推薦的最佳反應條件是什麼？

**Agent 流程：**
1. **條件預測**：調用 `run_askcos_condition_prediction`。
2. **總結**：生成推薦的溫度、溶劑和試劑清單。


### 4. 分子結構圖生成 (SMILES -> Image)


**用戶輸入：**
>請為 CC(C)C(=O)OCC(C)C 生成結構圖。

**Agent 流程：**
1. **圖片生成**：調用 `generate_molecule_image`。
2. **結果**：返回圖片文件的路徑，圖片將存儲在 `generated_mol_images/` 目錄中。

## 項目結構 (File Structure)
```
.
├── ASKLLM.py                # 核心 Agent 邏輯和交互循環
├── config.py                # API Key 配置文件
├── forward_prediction.py    # AskCOS 正向預測工具
├── retrosynthesis.py        # AskCOS 逆合成預測工具
├── impurity_prediction.py   # AskCOS 雜質預測工具
├── condition_prediction.py  # AskCOS 反應條件預測工具
├── smiles_resolver.py       # PubChem 名稱到 SMILES 解析工具
├── image_tool.py            # RDKit 分子結構圖生成工具
└── README.md                # 項目說明文件 (您當前正在閱讀)
```