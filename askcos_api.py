from flask import Flask, request, jsonify
from ASKLLM import run_interactive_agent, askcos_tools
from providers import QuotaLimitError

app = Flask(__name__)

# 以 session_id 維護多會話歷史；預設走 "default"
chat_histories = {}

@app.route("/askllm", methods=["POST"])
def askllm():
    data = request.get_json(force=True)
    query = data.get("query", "").strip()
    session_id = str(data.get("session_id", "default")).strip() or "default"

    if not query:
        return jsonify({"error": "query is required"}), 400

    history = chat_histories.setdefault(session_id, [])
    try:
        answer = run_interactive_agent(
            user_prompt=query,
            history=history,
            tools_to_use=askcos_tools,
        )
        return jsonify({
            "session_id": session_id,
            "answer": answer
        })
    except QuotaLimitError as e:
        return jsonify(
            {
                "session_id": session_id,
                "error": str(e),
                "action": "請稍後重試、切換模型，或檢查 API key 配額設定。",
            }
        ), 429
    except Exception as e:
        return jsonify({"session_id": session_id, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
