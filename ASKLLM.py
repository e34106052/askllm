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

# 已採用 Pro 模型以增強推理和多工具調用能力
PRIMARY_MODEL = 'gemini-2.5-pro'
PRIMARY_MODEL = 'gemini-2.5-flash'
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

def run_interactive_agent(user_prompt: str, history: List[types.Content], tools_to_use: list) -> str:
    """
    處理單輪對話的核心函數，現已啟用上下文記憶，並強制模型在工具執行後返回結果。
    
    Args:
        user_prompt: 當前用戶的查詢。
        history: 包含先前對話的 Content 對象列表 (List[types.Content])。
        tools_to_use: 可用的工具列表。
        
    Returns:
        模型的最終中文回覆。
    """
    system_instruction = (
        "The agent's primary function is to act as a professional chemistry assistant using the provided tools (AskCOS and image generation). "
        "Your ultimate goal is to generate a helpful final response based on the tool outputs. "
        "\n"
        "1. **Output Language**: Your final response to the user **must always be in Traditional Chinese (繁體中文)**, regardless of the intermediate steps or tool names."
        "\n"
        "2. **Priority 1: Name Resolution**: If the user's query contains a chemical compound name (e.g., Aspirin, 乙酸乙酯), you must first translate the name into English, and then use the `resolve_smiles_from_name` tool to obtain the SMILES string. This step is mandatory before calling any core AskCOS service (Retrosynthesis, Forward, Impurity, Condition)."
        "\n"
        "3. **Priority 2: Chemical Tasks**: Once the SMILES string is obtained, call the appropriate AskCOS tool based on the user's intent (e.g., `run_askcos_retrosynthesis` for synthesis path, `run_askcos_forward_prediction` for reaction product, `generate_molecule_image` for structure)."
        "\n"
        "4. **Multi-Step Tool Use**: You are permitted and encouraged to perform multi-step reasoning and tool calling (e.g., Name -> SMILES -> AskCOS Service)."
        "\n"
        "5. **Non-Chemical Questions**: For general knowledge, mathematics, or simple conversation, answer directly using your knowledge base without calling any tools."
        "\n"
        "6. **Final Response Guarantee (CRITICAL)**: After all necessary tools have been executed and the final data is available, you **must generate a clear, complete summary in Traditional Chinese as the final response**. **You must never return an empty text.** If a tool returns a result, you must summarize it."
    )
    
    current_model = PRIMARY_MODEL
    
    # --- 記憶功能相關變量 ---
    tool_call_made = False 
    last_tool_output_result = None
    
    # 構造當前輪次的用戶提示 (User Content)
    current_user_content = types.Content(role="user", parts=[types.Part(text=str(user_prompt))])
    
    # 構造完整的對話上下文：過去的歷史 + 當前用戶的提示
    contents = history + [current_user_content] 
    
    # 辅助函數：調用模型 (保持不變)
    def call_model(model_name: str, contents: List[Union[str, types.Content, types.Part]], tools: list) -> types.GenerateContentResponse:
        return client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction, 
                tools=tools
            ),
        )

    # --- 1. 階段一：初次調用和模型切換 ---
    try:
        # 第一次調用：將完整的歷史傳遞給模型
        response = call_model(current_model, contents, tools_to_use)
    except Exception as e:
        # ... (模型切換和錯誤處理邏輯不變)
        return f"主備模型均調用失敗，請檢查 API Key 或配額。\n主模型錯誤: {e}\n備用模型錯誤:"
    
    # --- 2. 處理函數調用 (Function Calling Loop) ---
    while response.function_calls:
        print(f"\n[ Agent] 正在使用 ({current_model}) 發現需要外部工具來回答問題...")
        
        tool_call_made = True # 標記本輪有工具調用
        model_request_content = response.candidates[0].content 
        tool_outputs = []
        
        for tool_call in response.function_calls:
            function_name = tool_call.name
            function_args = dict(tool_call.args)
            
            print(f"  -> 選中的工具: {function_name}")
            print(f"  -> 提取的參數: {function_args}")
            
            # 執行工具 (邏輯不變)
            if function_name == "run_askcos_forward_prediction":
                tool_output = run_askcos_forward_prediction(**function_args)
            elif function_name == "run_askcos_retrosynthesis":
                # 注意：這裡應該是 run_askcos_retrosynthesis，確保與 import 一致
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
            
            last_tool_output_result = tool_output # 持久化最新的工具輸出
            
            print(f"  -> 工具執行結果: {tool_output}")
            
            # 準備工具執行結果，用於反饋給模型
            tool_outputs.append(
                types.Part.from_function_response(
                    name=function_name,
                    response={"tool_result": tool_output},
                )
            )
        
        # 將工具輸出包裝成 Content 對象
        tool_outputs_content = types.Content(role="tool", parts=tool_outputs)

        # 3. 階段二：將工具執行結果反饋給模型
        # 構造用於第二次調用的內容：完整歷史 (contents) + 第一次響應 (模型請求) + 工具輸出
        contents_feedback = contents + [model_request_content] + [tool_outputs_content]
        
        try:
            # 第二次調用：模型根據工具結果生成最終回覆或下一個工具調用
            response = call_model(current_model, contents_feedback, tools_to_use)
            
            # 如果模型在第二次調用中決定繼續調用工具，則 contents 必須更新，以包含中間步驟
            # 更新 contents 以便下一個 while 循環使用完整的歷史
            contents = contents_feedback 
            
        except Exception as e:
            return f"模型 {current_model} 在工具調用反饋階段失敗: {e}"
        
    # 4. Agent 給出最終回覆
    final_response_text = response.text
    
    # 5. **新增：更新對話歷史記錄 (記憶功能)**
    if response.candidates and response.candidates[0].content:
        # 1. 記錄用戶請求
        history.append(current_user_content) 
        
        # 2. 記錄模型的最終響應 (包含最終文本、或僅包含工具呼叫，或空文本)
        # 注意：使用 response.candidates[0].content 捕捉模型生成的整個 Content 對象
        history.append(response.candidates[0].content)

    # 6. 關鍵修正：強制模型回答
    if final_response_text is None or final_response_text.strip() == "":
        if last_tool_output_result is not None:
            # 返回清晰的強制文本，包含最後一次成功的工具結果
            final_response_text = f"Agent 成功執行所有工具，但模型未能生成完整總結。以下是最後一次工具的結果：\n{last_tool_output_result}"
        else:
            final_response_text = "Agent 完成計算，但模型返回了空響應。"
            
    return final_response_text

# --- 主交互循環 ---
def main_loop():
    print("--- 歡迎使用 AskCOS 化學 Agent ---")
    print("您可以查詢正向預測、逆合成和分子結構圖。")
    print("輸入 'exit' 或 'quit' 退出程序。")
    
    # 啟用對話歷史：使用 List[types.Content] 存儲上下文
    chat_history: List[types.Content] = [] 
    
    while True:
        try:
            user_input = input("\n[ 用戶] 請輸入您的查詢: ")
            
            if user_input.lower() in ['exit', 'quit']:
                print("\n[ Agent] 謝謝使用，再見！")
                break
                
            if not user_input.strip():
                continue

            # 運行 Agent。chat_history 會在 run_interactive_agent 內部更新
            final_response = run_interactive_agent(user_input, chat_history, askcos_tools) 
            
            print(f"\n[ Agent] 最終回覆:\n{final_response}")
            
        except Exception as e:
            print(f"\n[ 錯誤] 發生意外錯誤: {e}")
            print("請重試或檢查 AskCOS 服務是否運行正常。")

if __name__ == "__main__":
    main_loop()