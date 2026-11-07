"""Flask backend: GET /, POST /chat, GET /health"""
import os
from flask import Flask, jsonify, render_template, request
from chatbot import Chatbot

app = Flask(__name__)
bot = Chatbot()

@app.get("/")
def index(): return render_template("index.html")

@app.get("/health")
def health(): return jsonify(status="ok", model_loaded=bot.model is not None, threshold=bot.threshold)

@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "")
    if not isinstance(question, str) or not question.strip():
        return jsonify(error="Please type a question."), 400
    if len(question) > 500:
        return jsonify(error="Question is too long (max 500 characters)."), 400
    try:
        return jsonify(bot.reply(question))
    except Exception as exc:                       # never leak a stack trace to the client
        app.logger.exception("chat failed")
        return jsonify(error="Something went wrong on the server. Please try again."), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
