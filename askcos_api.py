# askllm_api.py
from flask import Flask, request, jsonify
from ASKLLM import run_interactive_agent, askcos_tools
from google.genai import types

app = Flask(__name__)

# 全局對話歷史（單一會話用，之後你可以改成用 session_id 做多使用者）
chat_history = []  # List[types.Content]

@app.route("/askllm", methods=["POST"])
def askllm():
    data = request.get_json(force=True)
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "query is required"}), 400

    try:
        # 這裡 run_interactive_agent 裡面會自己用 chat_history 更新記憶
        answer = run_interactive_agent(
            user_prompt=query,
            history=chat_history,
            tools_to_use=askcos_tools,
        )
        return jsonify({
            "answer": answer
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
