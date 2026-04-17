"""
app_v3.py — Версия 3: Чат + явная генерация изображений (кнопка)

Что добавилось по сравнению с v2:
  - Новый маршрут /generate: принимает промпт, возвращает сгенерированное изображение
  - Кнопка "Нарисуй" во фронтенде
  - Пользователь явно нажимает кнопку чтобы запустить генерацию

Стек: Flask + OpenAI-совместимый API (chat: gpt-4o-mini, генерация: gpt-image-1)
Запуск: python app_v3.py
Открой: http://localhost:5003
"""

import os
import base64
from flask import Flask, request, jsonify, render_template_string
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# ═══════════════════════════════════════════════════════════════
#  КОНФИГУРАЦИЯ — все важные параметры здесь
# ═══════════════════════════════════════════════════════════════

# Ключ и базовый URL берём из переменных окружения
# Установи перед запуском:
API_KEY  = os.environ.get("POLLINATIONS_API_KEY", "")
BASE_URL = os.environ.get("POLLINATIONS_BASE_URL", "")

CHAT_MODEL    = os.environ.get("CHAT_MODEL")
IMAGE_MODEL   = os.environ.get("IMAGE_MODEL")    # модель генерации изображений
IMAGE_SIZE    = "1024x1024"      # размер генерируемого изображения
SYSTEM_PROMPT = "Ты — полезный ассистент. Отвечай кратко и по делу."
PORT          = 5003

# ═══════════════════════════════════════════════════════════════

app     = Flask(__name__)
client  = OpenAI(api_key=API_KEY, base_url=BASE_URL)
history = []

HTML = """
<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Чат v3 — генерация изображений</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, sans-serif; background: #f8fafc; color: #1e293b;
           display: flex; flex-direction: column; height: 100vh; }
    header { padding: 14px 20px; background: #ffffff; border-bottom: 1px solid #e2e8f0;
             font-weight: 700; font-size: 18px; display: flex; align-items: center; gap: 10px; }
    .tag { font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 10px;
           background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }
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
    .msg.image-result { align-self: flex-start; background: transparent; padding: 0; }
    .msg.image-result img { max-width: 380px; border-radius: 12px;
                             border: 2px solid #e2e8f0; display: block;
                             box-shadow: 0 4px 12px #0002; }
    .msg img.inline { max-width: 220px; border-radius: 8px; display: block; margin-bottom: 6px; }
    #form { padding: 12px 20px 16px; background: #ffffff; border-top: 1px solid #e2e8f0; }
    #preview-area { display: none; align-items: center; gap: 8px; margin-bottom: 10px;
                    padding: 8px 10px; background: #f1f5f9; border-radius: 8px; }
    #preview-area img { max-height: 80px; border-radius: 6px; border: 1px solid #e2e8f0; }
    #preview-area button { background: none; border: none; color: #94a3b8;
                           cursor: pointer; font-size: 18px; }
    #inputs { display: flex; gap: 8px; }
    #file-label { padding: 10px 12px; border-radius: 8px; background: #f1f5f9; color: #64748b;
                  cursor: pointer; font-size: 18px; border: 1px solid #e2e8f0;
                  display: flex; align-items: center; }
    #file-label:hover { background: #e2e8f0; }
    #input { flex: 1; padding: 10px 14px; border-radius: 8px; border: 1px solid #cbd5e1;
             background: #f8fafc; color: #1e293b; font-size: 15px; outline: none;
             resize: none; height: 44px; }
    #input:focus { border-color: #2563eb; box-shadow: 0 0 0 3px #dbeafe; }
    .btn { padding: 10px 16px; border-radius: 8px; border: none; cursor: pointer;
           font-size: 14px; font-weight: 600; transition: opacity .2s; white-space: nowrap; }
    .btn:hover    { opacity: 0.85; }
    .btn:disabled { opacity: 0.4; cursor: not-allowed; }
    .btn-chat  { background: #2563eb; color: white; }
    .btn-gen   { background: #15803d; color: white; }
    .btn-reset { background: #f1f5f9; color: #64748b; }
    .spinner { display: inline-block; animation: spin 1s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <header>
    🎨 Чат-бот
    <span class="tag">v3 — генерация изображений</span>
  </header>

  <div id="chat">
    <div class="msg system">
      Напишите сообщение и нажмите «Отправить», или введите описание и нажмите «🎨 Нарисуй».
    </div>
  </div>

  <div id="form">
    <div id="preview-area">
      <img id="preview-img" src="" alt="">
      <span style="font-size:13px;color:#64748b;flex:1" id="preview-name"></span>
      <button onclick="clearImage()">✕</button>
    </div>
    <div id="inputs">
      <label id="file-label" for="file-input" title="Прикрепить изображение">📎</label>
      <input id="file-input" type="file" accept="image/*" style="display:none">
      <textarea id="input" placeholder="Текст или описание для генерации..."></textarea>
      <button class="btn btn-chat" id="send-btn">Отправить</button>
      <button class="btn btn-gen"  id="gen-btn">🎨 Нарисуй</button>
      <button class="btn btn-reset" id="reset-btn">↺</button>
    </div>
  </div>

  <script>
    const chat    = document.getElementById('chat');
    const input   = document.getElementById('input');
    const sendBtn = document.getElementById('send-btn');
    const genBtn  = document.getElementById('gen-btn');
    let selectedFile = null;

    document.getElementById('file-input').onchange = (e) => {
      selectedFile = e.target.files[0];
      if (!selectedFile) return;
      document.getElementById('preview-img').src = URL.createObjectURL(selectedFile);
      document.getElementById('preview-name').textContent = selectedFile.name;
      document.getElementById('preview-area').style.display = 'flex';
    };

    function clearImage() {
      selectedFile = null;
      document.getElementById('file-input').value = '';
      document.getElementById('preview-area').style.display = 'none';
    }

    function addMsg(text, role) {
      const div = document.createElement('div');
      div.className = 'msg ' + role;
      div.textContent = text;
      chat.appendChild(div);
      chat.scrollTop = chat.scrollHeight;
      return div;
    }

    // ── Текстовый / мультимодальный чат ───────────────────────────────────────
    async function sendMessage() {
      const text = input.value.trim();
      if (!text && !selectedFile) return;

      const formData = new FormData();
      if (text)         formData.append('message', text);
      if (selectedFile) formData.append('image', selectedFile);

      const userDiv = document.createElement('div');
      userDiv.className = 'msg user';
      if (selectedFile) {
        const img = document.createElement('img');
        img.className = 'inline';
        img.src = URL.createObjectURL(selectedFile);
        userDiv.appendChild(img);
      }
      if (text) userDiv.appendChild(document.createTextNode(text));
      chat.appendChild(userDiv);
      chat.scrollTop = chat.scrollHeight;

      input.value = '';
      clearImage();
      sendBtn.disabled = true;

      const thinking = addMsg('...', 'assistant');
      try {
        const res  = await fetch('/chat', { method: 'POST', body: formData });
        const data = await res.json();
        if (!res.ok || data.error) {
          thinking.className = 'msg error';
          thinking.textContent = 'Ошибка: ' + (data.error || res.status);
        } else {
          thinking.textContent = data.reply;
        }
      } catch (e) {
        thinking.className = 'msg error';
        thinking.textContent = 'Ошибка: ' + e.message;
      } finally {
        sendBtn.disabled = false;
        input.focus();
      }
    }

    // ── Генерация изображения по кнопке ───────────────────────────────────────
    async function generateImage() {
      const prompt = input.value.trim();
      if (!prompt) { alert('Введите описание того, что нарисовать'); return; }

      addMsg(prompt, 'user');
      input.value = '';

      const status = addMsg('🎨 Генерирую... это займёт ~10–20 сек', 'system');
      genBtn.disabled = true;
      genBtn.innerHTML = '<span class="spinner">⏳</span> Генерирую...';

      try {
        const res  = await fetch('/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt })
        });
        const data = await res.json();
        status.remove();

        if (!res.ok || data.error) {
          addMsg('Ошибка: ' + (data.error || res.status), 'error');
          return;
        }

        const imgDiv = document.createElement('div');
        imgDiv.className = 'msg image-result';
        const img = document.createElement('img');
        img.src = data.image_url;
        img.alt = 'Сгенерированное изображение';
        imgDiv.appendChild(img);
        chat.appendChild(imgDiv);
        chat.scrollTop = chat.scrollHeight;
      } catch (e) {
        status.remove();
        addMsg('Ошибка: ' + e.message, 'error');
      } finally {
        genBtn.disabled = false;
        genBtn.textContent = '🎨 Нарисуй';
      }
    }

    sendBtn.onclick = sendMessage;
    genBtn.onclick  = generateImage;
    input.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    });
    document.getElementById('reset-btn').onclick = async () => {
      await fetch('/reset', { method: 'POST' });
      chat.innerHTML = '<div class="msg system">История сброшена.</div>';
    };
  </script>
</body>
</html>
"""

# ─── Вспомогательная функция генерации изображения ────────────────────────────

def do_generate_image(prompt: str) -> str:
    """Вызывает gpt-image-1, возвращает изображение как data URL (base64)."""
    response = client.images.generate(
        model=IMAGE_MODEL,
        prompt=prompt,
        size=IMAGE_SIZE,
        n=1
    )
    # gpt-image-1 возвращает base64 (b64_json), не URL
    b64 = response.data[0].b64_json
    return f"data:image/png;base64,{b64}"

# ─── Маршруты ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.form.get('message', '').strip()
        image_file   = request.files.get('image')

        if image_file and image_file.filename:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
            mime_type  = image_file.content_type or 'image/jpeg'
            content = [
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime_type};base64,{image_data}"}},
                {"type": "text",
                 "text": user_message if user_message else "Опиши картинку"}
            ]
            history_text = f"[картинка] {user_message}" if user_message else "[картинка]"
        else:
            if not user_message:
                return jsonify({"error": "Пустое сообщение"}), 400
            content      = user_message
            history_text = user_message

        history.append({"role": "user", "content": content})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
        response = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
        reply    = response.choices[0].message.content

        history[-1] = {"role": "user", "content": history_text}
        history.append({"role": "assistant", "content": reply})
        return jsonify({"reply": reply, "tokens": response.usage.total_tokens})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/generate', methods=['POST'])
def generate():
    try:
        data   = request.get_json()
        prompt = data.get('prompt', '').strip()
        if not prompt:
            return jsonify({"error": "Промпт не может быть пустым"}), 400
        image_url = do_generate_image(prompt)
        return jsonify({"image_url": image_url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/reset', methods=['POST'])
def reset():
    history.clear()
    return jsonify({"status": "ok"})


# ─── Запуск ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 50)
    print(f"Чат-бот v3 — генерация изображений  |  http://localhost:{PORT}")
    print("=" * 50)
    app.run(debug=True, port=PORT)
