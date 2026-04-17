"""
app_v1.py — Версия 1: Текстовый чат-бот

Стек: Flask + OpenAI-совместимый API
Запуск: python app_v1.py
Открой: http://localhost:5001
"""

import os
from flask import Flask, request, jsonify, render_template_string
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# ═══════════════════════════════════════════════════════════════
#  КОНФИГУРАЦИЯ — все важные параметры здесь
# ═══════════════════════════════════════════════════════════════

API_KEY  = os.environ.get("POLLINATIONS_API_KEY", "")
BASE_URL = os.environ.get("POLLINATIONS_BASE_URL", "")

CHAT_MODEL    = os.environ.get("CHAT_MODEL", "")
SYSTEM_PROMPT = "Ты — полезный ассистент. Отвечай кратко и по делу."
PORT          = 5001

# ═══════════════════════════════════════════════════════════════

app     = Flask(__name__)
client  = OpenAI(api_key=API_KEY, base_url=BASE_URL)
history = []

print("API KEY")
print(API_KEY)
print(BASE_URL)
print(f"Model: {CHAT_MODEL}")

HTML = """
<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Чат v1 — текстовый</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, sans-serif; background: #f8fafc; color: #1e293b;
           display: flex; flex-direction: column; height: 100vh; }
    header { padding: 14px 20px; background: #ffffff; border-bottom: 1px solid #e2e8f0;
             font-weight: 700; font-size: 18px; display: flex; align-items: center; gap: 10px; }
    .tag { font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 10px;
           background: #dbeafe; color: #1d4ed8; border: 1px solid #bfdbfe; }
    #chat { flex: 1; overflow-y: auto; padding: 20px; display: flex;
            flex-direction: column; gap: 12px; }
    .msg { max-width: 75%; padding: 10px 14px; border-radius: 12px;
           line-height: 1.6; font-size: 15px; white-space: pre-wrap; }
    .msg.user      { align-self: flex-end; background: #2563eb; color: white; }
    .msg.assistant { align-self: flex-start; background: #ffffff; color: #1e293b;
                     border: 1px solid #e2e8f0; box-shadow: 0 1px 3px #0001; }
    .msg.system    { align-self: center; font-size: 12px; color: #94a3b8; font-style: italic; }
    .msg.error     { align-self: center; background: #fee2e2; color: #dc2626;
                     border: 1px solid #fca5a5; font-size: 13px; }
    .meta { font-size: 11px; color: #94a3b8; margin-top: 2px; padding: 0 4px; }
    .meta.right { text-align: right; }
    #form { display: flex; gap: 8px; padding: 16px 20px; background: #ffffff;
            border-top: 1px solid #e2e8f0; }
    #input { flex: 1; padding: 10px 14px; border-radius: 8px; border: 1px solid #cbd5e1;
             background: #f8fafc; color: #1e293b; font-size: 15px; outline: none;
             resize: none; height: 44px; }
    #input:focus { border-color: #2563eb; box-shadow: 0 0 0 3px #dbeafe; }
    button { padding: 10px 18px; border-radius: 8px; border: none; cursor: pointer;
             font-size: 14px; font-weight: 600; transition: opacity .2s; }
    button:hover    { opacity: 0.85; }
    button:disabled { opacity: 0.4; cursor: not-allowed; }
    #send-btn  { background: #2563eb; color: white; }
    #reset-btn { background: #f1f5f9; color: #64748b; }
  </style>
</head>
<body>
  <header>
    💬 Чат-бот
    <span class="tag">v1 — текстовый</span>
  </header>

  <div id="chat">
    <div class="msg system">Начните разговор — введите сообщение ниже.</div>
  </div>

  <div id="form">
    <textarea id="input" placeholder="Введите сообщение... (Enter — отправить, Shift+Enter — перенос)"></textarea>
    <button id="send-btn">Отправить</button>
    <button id="reset-btn">↺ Сброс</button>
  </div>

  <script>
    const chat    = document.getElementById('chat');
    const input   = document.getElementById('input');
    const sendBtn = document.getElementById('send-btn');

    function addMsg(text, role) {
      const div = document.createElement('div');
      div.className = 'msg ' + role;
      div.textContent = text;
      chat.appendChild(div);
      chat.scrollTop = chat.scrollHeight;
      return div;
    }

    async function sendMessage() {
      const text = input.value.trim();
      if (!text) return;
      input.value = '';
      sendBtn.disabled = true;
      addMsg(text, 'user');
      const thinking = addMsg('...', 'assistant');
      try {
        const res  = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text })
        });
        const data = await res.json();
        if (!res.ok || data.error) {
          thinking.className = 'msg error';
          thinking.textContent = 'Ошибка: ' + (data.error || res.status);
        } else {
          thinking.textContent = data.reply;
          const meta = document.createElement('div');
          meta.className = 'meta';
          meta.textContent = 'токенов: ' + data.tokens;
          chat.appendChild(meta);
        }
      } catch (e) {
        thinking.className = 'msg error';
        thinking.textContent = 'Ошибка: ' + e.message;
      } finally {
        sendBtn.disabled = false;
        input.focus();
        chat.scrollTop = chat.scrollHeight;
      }
    }

    sendBtn.onclick = sendMessage;
    input.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    });
    document.getElementById('reset-btn').onclick = async () => {
      await fetch('/reset', { method: 'POST' });
      chat.innerHTML = '<div class="msg system">История сброшена. Начните новый разговор.</div>';
    };
  </script>
</body>
</html>
"""

# ─── Маршруты ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        print(user_message)
        if not user_message:
            return jsonify({"error": "Пустое сообщение"}), 400

        history.append({"role": "user", "content": user_message})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

        response = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
        reply  = response.choices[0].message.content
        tokens = response.usage.total_tokens

        history.append({"role": "assistant", "content": reply})
        return jsonify({"reply": reply, "tokens": tokens, "history_len": len(history)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/reset', methods=['POST'])
def reset():
    history.clear()
    return jsonify({"status": "ok"})


# ─── Запуск ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 50)
    print(f"Чат-бот v1 — текстовый  |  http://localhost:{PORT}")
    print("=" * 50)
    app.run(debug=True, port=PORT)
