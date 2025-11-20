from google.genai import types
from forward_prediction import run_askcos_forward_prediction
from retrosynthesis import run_askcos_retrosynthesis
from image_tool import generate_molecule_image
from impurity_prediction import run_askcos_impurity_prediction
from smiles_resolver import resolve_smiles_from_name
from condition_prediction import run_askcos_condition_prediction
import os, sys
from google import genai
from typing import List, Union

# --- 導入配置 (假設 config.py 存在) ---
try:
    from config import GEMINI_API_KEY
except ImportError:
    print("錯誤：找不到 config.py 文件或其中未定義 GEMINI_API_KEY。請先創建 config.py！")
    sys.exit(1)
# ----------------------------------------

PRIMARY_MODEL = 'gemini-2.5-flash-lite'
# PRIMARY_MODEL = 'gemini-2.5-pro'
BACKUP_MODEL = 'gemini-2.5-flash'

# 初始化 Gemini 客戶端
try:
    client = genai.Client(api_key=GEMINI_API_KEY)
    # 註冊所有 AskCOS 和圖像工具
    askcos_tools = [
        run_askcos_forward_prediction, 
        run_askcos_retrosynthesis, 
        generate_molecule_image,
        run_askcos_impurity_prediction,
        resolve_smiles_from_name,
        run_askcos_condition_prediction
    ]
except Exception as e:
    print(f"Gemini 客戶端初始化失敗：{e}")
    sys.exit(1)

def run_interactive_agent(user_prompt: str, tools_to_use: list, history: list) -> str:
    """
    處理單輪對話的核心函數，已禁用長期記憶，並強制模型在工具執行後返回結果。
    """
    system_instruction = (
        "您是一個多功能的 AI 助手，專注於化學。您的首要任務是回答用戶的化學相關問題。 "
        "請務必遵循以下工具使用規則和優先級："
        "\n1. **自我確認與翻譯**：如果用戶的查詢是關於化學的，您必須**首先將用戶輸入中的所有中文化合物名稱或術語翻譯成英文**，並使用英文名稱進行所有後續的工具調用。在第一次回應中，請將您翻譯的英文名稱顯示給自己確認。"
        "\n2. **多步調用**：為了解決單一複雜問題（例如：名稱到產物圖片），您可以在第一次工具調用完成後，根據其結果**繼續調用其他必要的工具**，直到問題完全解決為止。**您不一定要在每次工具調用後立即生成最終回答**。"
        "\n3. **名稱解析優先**：在您確認了英文名稱後，如果查詢是使用化合物名稱而不是 SMILES 字符串，您必須**調用 `resolve_smiles_from_name` 工具**來獲取 SMILES。"
        "\n4. **工具調用**：只有在獲得所有必要的 SMILES 字符串後，才能調用 AskCOS 工具（如 `run_askcos_forward_prediction` 或 `run_askcos_retrosynthesis_outcomes`）。"
        "\n5. **條件預測**：當用戶詢問**反應條件、溶劑或催化劑**時，您必須使用 `run_askcos_condition_prediction` 工具。請確保將反應 SMILES (RCTS>>PRD) 作為參數傳入。"
        "\n6. **非化學問題**：對於簡單的數學、常識或閒聊問題，請直接使用您的自身知識以中文作答，不要調用工具。"
    )
    
    current_model = PRIMARY_MODEL
    full_conversation = [user_prompt] # 由於禁用記憶，只傳遞當前提示
    
    # 辅助函数：调用模型 (保持不變)
    def call_model(model_name: str, contents: List[Union[str, types.Content, types.Part]], tools: list) -> types.GenerateContentResponse:
        return client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction, 
                tools=tools
            ),
        )

    last_tool_output_result = None #  儲存最新的工具輸出結果字符串

    # --- 1. 階段一：初次調用和模型切換 ---
    try:
        response = call_model(current_model, full_conversation, tools_to_use)
    except Exception as e:
        # ... (模型切換和錯誤處理邏輯不變)
        return f"主備模型均調用失敗，請檢查 API Key 或配額。\n主模型錯誤: {e}\n備用模型錯誤:"
    
    # --- 2. 處理函數調用 (Function Calling Loop) ---
    while response.function_calls:
        print(f"\n[ Agent] 正在使用 ({current_model}) 發現需要外部工具來回答問題...")
        
        model_request = response.candidates[0].content 
        tool_outputs = []
        
        for tool_call in response.function_calls:
            function_name = tool_call.name
            function_args = dict(tool_call.args)
            
            print(f"  -> 選中的工具: {function_name}")
            print(f"  -> 提取的參數: {function_args}")
            
            # 執行工具 (邏輯不變)
            if function_name == "run_askcos_forward_prediction":
                tool_output = run_askcos_forward_prediction(**function_args)
            elif function_name == "run_askcos_retrosynthesis_outcomes":
                tool_output = run_askcos_retrosynthesis(**function_args)
            elif function_name == "generate_molecule_image":
                tool_output = generate_molecule_image(**function_args)
            elif function_name == "run_askcos_impurity_prediction":
                tool_output = run_askcos_impurity_prediction(**function_args)
            elif function_name == "resolve_smiles_from_name":
                tool_output = resolve_smiles_from_name(**function_args)
            elif function_name == "run_askcos_condition_prediction":
                tool_output = run_askcos_condition_prediction(**function_args)
            else:
                tool_output = f"未知的工具 {function_name}"
            
            last_tool_output_result = tool_output #  持久化最新的工具輸出
            
            print(f"  -> 工具執行結果: {tool_output}")
            
            # 準備工具執行結果，用於反饋給模型
            tool_outputs.append(
                types.Part.from_function_response(
                    name=function_name,
                    response={"tool_result": tool_output},
                )
            )

        # 3. 階段二：將工具執行結果反饋給模型
        # contents_feedback = 當前用戶提示 + 模型請求 + 工具輸出
        contents_feedback = full_conversation + [model_request] + tool_outputs
        
        try:
            response = call_model(current_model, contents_feedback, tools_to_use)
        except Exception as e:
            return f"模型 {current_model} 在工具調用反饋階段失敗: {e}"
        
    # 4. Agent 給出最終回覆
    final_response_text = response.text
    
    #  關鍵修正：強制模型回答
    if final_response_text is None:
        if last_tool_output_result is not None:
            #  返回清晰的強制文本，包含最後一次成功的工具結果
            final_response_text = f"Agent 成功執行所有工具，但模型未能生成完整總結。以下是最後一次工具的結果：\n{last_tool_output_result}"
        else:
            final_response_text = "Agent 完成計算，但模型返回了空響應。"
    
    # 由於已禁用記憶，無需處理 history.extend
    
    return final_response_text

# --- 主交互循環 ---
def main_loop():
    print("--- 歡迎使用 AskCOS 化學 Agent ---")
    print("您可以查詢正向預測、逆合成和分子結構圖。")
    print("輸入 'exit' 或 'quit' 退出程序。")
    
    # 雖然存在，但在 run_interactive_agent 中已被忽略
    chat_history = [] 
    
    while True:
        try:
            user_input = input("\n[ 用戶] 請輸入您的查詢: ")
            
            if user_input.lower() in ['exit', 'quit']:
                print("\n[ Agent] 謝謝使用，再見！")
                break
                
            if not user_input.strip():
                continue

            # 運行 Agent (每次都是獨立的單輪會話)
            final_answer = run_interactive_agent(user_input, askcos_tools, chat_history)
            
            print("\n[ Agent] 最終回覆:")
            print(final_answer)
            
        except Exception as e:
            print(f"\n[ 錯誤] 發生意外錯誤: {e}")
            print("請重試或檢查 AskCOS 服務是否運行正常。")

if __name__ == "__main__":
    main_loop()