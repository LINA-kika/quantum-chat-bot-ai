import os, json, base64, io
from flask import Flask, request, jsonify, render_template_string
from openai import OpenAI

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# ================= CONFIG =================
API_KEY  = os.environ.get("POLLINATIONS_API_KEY", "")
BASE_URL = os.environ.get("POLLINATIONS_BASE_URL", "")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
app = Flask(__name__)
history = []

# ================= SYSTEM PROMPT =================
SYSTEM_PROMPT = """
Ты — ассистент по квантовой физике с доступом к вычислительным инструментам.

ТВОЯ ЗАДАЧА:
1. Если пользователь вводит потенциал или просит решить уравнение Шрёдингера:
   ОБЯЗАТЕЛЬНО вызывай solve_schrodinger

2. Если пользователь говорит про кубиты или гейты:
   вызывай simulate_quantum_circuit

3. Перед вызовом функции ТЫ ОБЯЗАН:
   - нормализовать вход
   - привести к строгому формату

ПРАВИЛА:

Для solve_schrodinger:
- извлеки V(x)
- формат: Python expression
- x^2 → x**2
- 1/2 x^2 → 0.5*x**2
- cos x → cos(x)
- exp x → exp(x)

Примеры:
"гармонический осциллятор" → {"potential": "x**2"}
"V(x)=1/2 x^2" → {"potential": "0.5*x**2"}

Для simulate_quantum_circuit:
"H X Z" → {"gates": ["H","X","Z"]}

ПОСЛЕ вызова функции ты ОБЯЗАН:
- указать инструмент
- вывести pipeline
- объяснить результат

Формат ответа:
[Инструмент]
[Pipeline]
[Результат]
"""

# ================= TOOLS =================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "solve_schrodinger",
            "parameters": {
                "type": "object",
                "properties": {
                    "potential": {"type": "string"}
                },
                "required": ["potential"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_quantum_circuit",
            "parameters": {
                "type": "object",
                "properties": {
                    "gates": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["gates"]
            }
        }
    }
]

# ================= HTML (НЕ ТРОГАЕМ) =================
HTML = """
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Quantum Chat</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui; background: #f8fafc; display: flex; flex-direction: column; height: 100vh; }
header { padding: 14px 20px; background: #fff; border-bottom: 1px solid #e2e8f0; font-weight:700 }
#chat { flex:1; overflow-y:auto; padding:20px; display:flex; flex-direction:column; gap:12px }
.msg { max-width:80%; padding:10px 14px; border-radius:12px; white-space:pre-wrap }
.user { align-self:flex-end; background:#2563eb; color:white }
.assistant { align-self:flex-start; background:white; border:1px solid #e2e8f0 }
.system { align-self:center; font-size:12px; color:#64748b }
img { max-width:350px; border-radius:10px; margin-top:6px }
#form { padding:12px; background:#fff; border-top:1px solid #e2e8f0 }
#inputs { display:flex; gap:8px }
#input { flex:1; padding:10px }
button { padding:10px }
</style>
</head>
<body>

<header>⚛️ Quantum Chat</header>

<div id="chat">
<div class="msg system">
Примеры:
• Реши уравнение Шрёдингера для гармонического осциллятора
• V(x)=x^2
• Прогони кубит через H и X
</div>
</div>

<div id="form">
<div id="inputs">
<textarea id="input"></textarea>
<button id="send-btn">Отправить</button>
<button id="reset-btn">↺</button>
</div>
</div>

<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send-btn');
const resetBtn = document.getElementById('reset-btn');

function addMsg(text, role){
 let div=document.createElement('div');
 div.className='msg '+role;
 div.textContent=text;
 chat.appendChild(div);
 chat.scrollTop=chat.scrollHeight;
 return div;
}

async function sendMessage(){
 let text=input.value.trim();
 if(!text) return;

 addMsg(text,'user');
 input.value='';

 let thinking=addMsg('🤔 Думаю...','assistant');

 try{
   let res=await fetch('/chat',{
     method:'POST',
     headers:{'Content-Type':'application/json'},
     body:JSON.stringify({message:text})
   });

   let data=await res.json();
   thinking.remove();

   addMsg(data.reply,'assistant');

   if(data.tool_result){
     if(data.tool_result.plot){
       let img=document.createElement('img');
       img.src=data.tool_result.plot;
       chat.appendChild(img);
     }
     if(data.tool_result.animation){
       let img=document.createElement('img');
       img.src=data.tool_result.animation;
       chat.appendChild(img);
     }
   }

 }catch(e){
   thinking.remove();
   addMsg('Ошибка: '+e.message,'assistant');
 }
}

sendBtn.onclick=sendMessage;

input.addEventListener('keydown',e=>{
 if(e.key==='Enter' && !e.shiftKey){
  e.preventDefault();
  sendMessage();
 }
});

resetBtn.onclick=async ()=>{
 await fetch('/reset',{method:'POST'});
 chat.innerHTML='<div class="msg system">История сброшена</div>';
};
</script>

</body>
</html>
"""

# ================= HELPERS =================

def safe_sympify(expr):
    try:
        return sp.sympify(expr)
    except Exception as e:
        raise ValueError(f"Ошибка парсинга выражения: {expr} | {str(e)}")

# ================= PHYSICS =================

def solve_schrodinger(expr):
    original_expr = expr
    print("RAW INPUT:", expr)

    expr = expr.lower()
    expr = expr.replace("v(x)=", "").replace("v(x)", "")
    expr = expr.replace("^", "**")
    expr = expr.replace(" ", "*")
    expr = expr.replace("1/2*", "0.5*")

    try:
        n = 200
        x = np.linspace(-5,5,n)
        dx = x[1]-x[0]

        xs = sp.symbols('x')
        V_expr = safe_sympify(expr)
        V = sp.lambdify(xs, V_expr, 'numpy')(x)

    except Exception as e:
        return {"error": str(e)}

    diag = 1/dx**2 + V
    off = -0.5/dx**2*np.ones(n-1)
    H = np.diag(diag)+np.diag(off,1)+np.diag(off,-1)

    vals,vecs = eigh(H)

    psi0 = vecs[:,0]
    psi0 = psi0/np.sqrt(np.trapezoid(abs(psi0)**2,x))

    # === ВИЗУАЛИЗАЦИЯ ===
    plt.figure()
    plt.plot(x, V, label="V(x)")
    plt.plot(x, psi0.real, label="Re(ψ0)")
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf,format='png')
    buf.seek(0)
    plot = base64.b64encode(buf.read()).decode()

    return {
        "plot": f"data:image/png;base64,{plot}",
        "energies": vals[:3].tolist(),
        "pipeline": [
            "Парсинг через SymPy",
            "Дискретизация пространства",
            "Построение гамильтониана",
            "Диагонализация (eigh)",
            "Нормировка ψ",
            "Построение графика ψ и V"
        ],
        "input": original_expr
    }


def simulate_quantum_circuit(gates):
    qc=QuantumCircuit(1)
    applied=[]

    for g in gates:
        g=g.lower()
        if g=="h":
            qc.h(0)
            applied.append("H")
        if g=="x":
            qc.x(0)
            applied.append("X")
        if g=="z":
            qc.z(0)
            applied.append("Z")

    state = Statevector.from_instruction(qc).data

    return {
        "statevector": str(state),
        "pipeline": [
            "Создание схемы",
            f"Применение гейтов: {', '.join(applied)}",
            "Расчёт statevector"
        ]
    }

# ================= ROUTES =================

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/chat', methods=['POST'])
def chat():
    user_msg=request.json.get('message')
    history.append({"role":"user","content":user_msg})

    messages=[{"role":"system","content":SYSTEM_PROMPT}] + history

    resp=client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        tools=TOOLS
    )

    msg=resp.choices[0].message

    if msg.tool_calls:
        tc=msg.tool_calls[0]
        args=json.loads(tc.function.arguments)

        try:
            if tc.function.name=="solve_schrodinger":
                result=solve_schrodinger(args["potential"])
            elif tc.function.name=="simulate_quantum_circuit":
                result=simulate_quantum_circuit(args["gates"])
            else:
                result={"error":"Unknown tool"}
        except Exception as e:
            result={"error":str(e)}

        messages.append({"role":"assistant","tool_calls":[tc.model_dump()]})
        messages.append({
            "role":"tool",
            "tool_call_id":tc.id,
            "content":json.dumps(result)
        })

        final=client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages
        )

        reply=final.choices[0].message.content
        history.append({"role":"assistant","content":reply})

        return jsonify({"reply":reply,"tool_result":result})

    reply=msg.content
    history.append({"role":"assistant","content":reply})

    return jsonify({"reply":reply})

@app.route('/reset', methods=['POST'])
def reset():
    history.clear()
    return jsonify({"status":"ok"})

if __name__=='__main__':
    app.run(port=5006,debug=True)