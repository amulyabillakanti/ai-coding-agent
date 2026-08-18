import os
import sys
import io
import traceback
from typing import TypedDict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

app = FastAPI(
    title="AI Coding Agent",
    description="AI-powered coding, testing and execution pipeline",
    version="2.0.0"
)

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-3.6-flash"
llm = None

if api_key:
    try:
        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=api_key
        )
    except Exception as exc:
        print(f"Gemini initialization error: {exc}")
else:
    print("WARNING: GEMINI_API_KEY is not configured.")


class CrewState(TypedDict, total=False):
    messages: List[BaseMessage]
    code: Optional[str]
    report: Optional[str]


class TaskRequest(BaseModel):
    task: str


def extract_content(response) -> str:
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
            else:
                parts.append(str(item))

        return "\n".join(parts)

    return str(content)


def clean_python_code(code: str) -> str:
    code = str(code).strip()

    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```Python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]

    if code.endswith("```"):
        code = code[:-3]

    return code.strip()


def run_python_code(code: str) -> str:
    clean_code = clean_python_code(code)

    old_stdout = sys.stdout
    new_stdout = io.StringIO()

    namespace = {
        "__name__": "__main__",
        "__builtins__": __builtins__
    }

    sys.stdout = new_stdout

    try:
        exec(clean_code, namespace, namespace)
        result = new_stdout.getvalue()
    except Exception:
        result = "Execution Error:\n" + traceback.format_exc()
    finally:
        sys.stdout = old_stdout

    result = result.strip()

    return result if result else "Success (no terminal output)"


def generate_test_cases(task: str) -> str:
    if llm is None:
        return "Test generation unavailable because Gemini is not configured."

    prompt = f"""
You are a Senior QA Engineer.

Generate exactly 4 highly specific test scenarios for this coding task:

{task}

Include:
1. Normal case
2. Boundary case
3. Edge case
4. Invalid input case

Return only a numbered list.
"""

    return extract_content(llm.invoke(prompt))


def real_time_developer(state: CrewState):
    if llm is None:
        raise ValueError(
            "Gemini API is not configured. Set GEMINI_API_KEY."
        )

    messages = state.get("messages", [])

    if not messages:
        raise ValueError("No coding task was provided.")

    task = extract_content(messages[-1])

    prompt = f"""
You are an expert Python developer.

Write a complete executable Python program for this task:

{task}

Requirements:
- Return only Python source code.
- Do not use Markdown.
- Do not include code fences.
- The program must run directly with Python.
- Use clear variable names.
- Handle reasonable edge cases.
- Print the result when output is required.
"""

    response = llm.invoke(prompt)
    code = clean_python_code(extract_content(response))

    if not code:
        raise ValueError("Gemini returned empty Python code.")

    return {"code": code}


def real_time_tester(state: CrewState):
    messages = state.get("messages", [])

    if not messages:
        raise ValueError("No coding task available.")

    code = state.get("code", "")

    if not code:
        raise ValueError("No generated code available.")

    task = extract_content(messages[-1])
    tests = generate_test_cases(task)
    output = run_python_code(code)

    report = (
        "EXECUTION OUTPUT\n\n"
        + output
        + "\n\n"
        + "TEST SCENARIOS\n\n"
        + tests
    )

    return {"report": report}


workflow = StateGraph(CrewState)

workflow.add_node("developer", real_time_developer)
workflow.add_node("tester", real_time_tester)

workflow.add_edge(START, "developer")
workflow.add_edge("developer", "tester")
workflow.add_edge("tester", END)

rt_app = workflow.compile()


@app.get("/", response_class=HTMLResponse)
async def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>AI Coding Agent</title>
<style>
*{box-sizing:border-box}
:root{
--bg:#12091f;
--purple:#7c3aed;
--violet:#a855f7;
--lavender:#c084fc;
--pink:#ec4899;
--white:#f5f3ff;
--muted:#c4b5fd;
--glass:rgba(255,255,255,.075);
--border:rgba(255,255,255,.14)
}
html{scroll-behavior:smooth}
body{
margin:0;
min-height:100vh;
font-family:"Space Grotesk","Segoe UI",Arial,sans-serif;
color:#fff;
background:
radial-gradient(circle at 5% 10%,rgba(168,85,247,.32),transparent 27%),
radial-gradient(circle at 95% 15%,rgba(236,72,153,.22),transparent 25%),
radial-gradient(circle at 50% 100%,rgba(124,58,237,.28),transparent 35%),
linear-gradient(135deg,#12091f,#1c0b31 45%,#10081c);
overflow-x:hidden
}
body:before{
content:"";
position:fixed;
inset:0;
pointer-events:none;
background-image:
linear-gradient(rgba(255,255,255,.018) 1px,transparent 1px),
linear-gradient(90deg,rgba(255,255,255,.018) 1px,transparent 1px);
background-size:50px 50px;
mask-image:linear-gradient(to bottom,black,transparent)
}
.container{
width:min(1250px,100%);
margin:auto;
padding:28px 22px 60px;
position:relative
}
.nav{
height:66px;
display:flex;
align-items:center;
justify-content:space-between;
padding:0 20px;
border:1px solid var(--border);
background:rgba(255,255,255,.065);
backdrop-filter:blur(20px);
border-radius:20px;
box-shadow:0 15px 45px rgba(0,0,0,.2);
margin-bottom:70px
}
.brand{
display:flex;
align-items:center;
gap:12px;
font-weight:700
}
.brand-orb{
width:36px;
height:36px;
border-radius:12px;
display:grid;
place-items:center;
background:linear-gradient(135deg,var(--pink),var(--violet),var(--purple));
box-shadow:0 0 25px rgba(168,85,247,.55)
}
.online{
display:flex;
align-items:center;
gap:8px;
color:#c4b5fd;
font-size:13px
}
.online-dot{
width:8px;
height:8px;
border-radius:50%;
background:#4ade80;
box-shadow:0 0 12px #4ade80
}
.hero{
text-align:center;
margin-bottom:55px
}
.big-orb{
width:96px;
height:96px;
margin:0 auto 25px;
border-radius:30px;
display:grid;
place-items:center;
font-size:42px;
background:radial-gradient(circle at 35% 25%,#f5d0fe,#c084fc 25%,#9333ea 60%,#4c1d95);
box-shadow:0 0 35px rgba(168,85,247,.6),0 0 90px rgba(236,72,153,.18);
animation:orb 4s ease-in-out infinite
}
@keyframes orb{
0%,100%{transform:translateY(0) rotate(0)}
50%{transform:translateY(-7px) rotate(2deg)}
}
.hero h1{
margin:0;
font-size:clamp(42px,6vw,70px);
letter-spacing:-3px;
line-height:1;
background:linear-gradient(90deg,#fff,#e9d5ff,#f0abfc,#c084fc);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent
}
.hero p{
color:#c4b5fd;
font-size:18px;
margin:20px 0 0
}
.dashboard{
display:grid;
grid-template-columns:1fr 1.25fr;
gap:22px
}
.card{
background:var(--glass);
border:1px solid var(--border);
backdrop-filter:blur(24px);
-webkit-backdrop-filter:blur(24px);
border-radius:24px;
padding:25px;
box-shadow:0 25px 70px rgba(0,0,0,.22);
transition:.3s ease
}
.card:hover{
transform:translateY(-3px);
border-color:rgba(192,132,252,.25)
}
.card-title{
display:flex;
align-items:center;
justify-content:space-between;
margin-bottom:20px
}
.card-title h2{
font-size:18px;
margin:0;
color:#f5f3ff
}
.label{
font-size:11px;
text-transform:uppercase;
letter-spacing:1.5px;
color:#a78bfa
}
textarea{
width:100%;
height:260px;
resize:none;
padding:20px;
border-radius:18px;
border:1px solid rgba(196,181,253,.18);
background:rgba(5,2,15,.55);
color:#fff;
font:15px/1.7 "Space Grotesk",sans-serif;
outline:none;
transition:.25s
}
textarea::placeholder{color:#81749b}
textarea:focus{
border-color:#a855f7;
box-shadow:0 0 0 4px rgba(168,85,247,.1)
}
.run{
width:100%;
margin-top:16px;
padding:16px;
border:0;
border-radius:16px;
background:linear-gradient(110deg,#7c3aed,#a855f7,#ec4899,#a855f7);
background-size:250% 100%;
color:white;
font-size:15px;
font-weight:700;
cursor:pointer;
box-shadow:0 12px 35px rgba(124,58,237,.35);
transition:.25s
}
.run:hover{
background-position:100% 0;
transform:translateY(-2px);
box-shadow:0 16px 40px rgba(168,85,247,.45)
}
.run:disabled{
opacity:.55;
cursor:not-allowed;
transform:none
}
.steps{
margin-top:20px;
display:grid;
gap:9px
}
.step{
display:flex;
align-items:center;
gap:10px;
color:#716783;
font-size:13px
}
.step.active{color:#e9d5ff}
.step.done{color:#86efac}
.step-icon{
width:21px;
height:21px;
border-radius:50%;
display:grid;
place-items:center;
border:1px solid #4b3b62;
font-size:10px
}
.step.active .step-icon{
border-color:#a855f7;
background:rgba(168,85,247,.15);
box-shadow:0 0 14px rgba(168,85,247,.35)
}
.step.done .step-icon{
border-color:#4ade80;
color:#4ade80
}
.code-window{
overflow:hidden;
border-radius:18px;
background:#080510;
border:1px solid rgba(255,255,255,.1)
}
.window-bar{
height:42px;
display:flex;
align-items:center;
gap:7px;
padding:0 14px;
border-bottom:1px solid rgba(255,255,255,.07);
background:rgba(255,255,255,.035)
}
.dot{
width:10px;
height:10px;
border-radius:50%
}
.red{background:#fb7185}
.yellow{background:#fbbf24}
.green{background:#4ade80}
.file-name{
margin-left:8px;
color:#827694;
font-size:12px
}
pre{
margin:0;
height:260px;
padding:18px;
overflow:auto;
white-space:pre-wrap;
word-break:break-word;
color:#ddd6fe;
font:13px/1.7 "JetBrains Mono","Fira Code",Consolas,monospace
}
.actions{
display:flex;
gap:9px;
margin-top:14px
}
.small-btn{
flex:1;
padding:10px 12px;
border-radius:12px;
border:1px solid rgba(196,181,253,.15);
background:rgba(255,255,255,.05);
color:#ddd6fe;
cursor:pointer;
font-size:12px;
transition:.2s
}
.small-btn:hover{
background:rgba(168,85,247,.13);
border-color:#8b5cf6;
color:#fff
}
.results{margin-top:22px}
.result-grid{
display:grid;
grid-template-columns:repeat(3,1fr);
gap:14px
}
.stat{
padding:20px;
border-radius:18px;
background:rgba(5,2,15,.35);
border:1px solid rgba(255,255,255,.08)
}
.stat-number{
font-size:30px;
font-weight:800;
color:#fff
}
.stat-label{
font-size:11px;
color:#8f83a4;
text-transform:uppercase;
letter-spacing:1px;
margin-top:3px
}
.stat.passed .stat-number{color:#86efac}
.stat.failed .stat-number{color:#fb7185}
.test-list{
display:grid;
grid-template-columns:repeat(2,1fr);
gap:10px;
margin-top:16px
}
.test-item{
padding:13px 15px;
border-radius:14px;
background:rgba(255,255,255,.04);
color:#c4b5fd;
font-size:13px;
border:1px solid rgba(255,255,255,.06)
}
.test-item:before{
content:"✓";
color:#4ade80;
margin-right:8px;
font-weight:bold
}
.output{margin-top:18px}
.output pre{
height:180px;
background:rgba(5,2,15,.5);
border-radius:16px
}
.status{
text-align:center;
min-height:22px;
margin-top:13px;
font-size:13px
}
.success{color:#86efac}
.error{color:#fb7185}
.footer{
text-align:center;
margin-top:35px;
color:#695c7d;
font-size:12px
}
@media(max-width:850px){
.dashboard{grid-template-columns:1fr}
.nav{margin-bottom:45px}
.hero{margin-bottom:35px}
}
@media(max-width:600px){
.container{padding:15px 12px 40px}
.nav{
height:58px;
padding:0 14px;
border-radius:16px
}
.brand{font-size:13px}
.hero h1{letter-spacing:-2px}
.hero p{font-size:15px}
.big-orb{
width:76px;
height:76px;
font-size:32px;
border-radius:24px
}
.card{
padding:18px;
border-radius:19px
}
textarea,pre{height:220px}
.result-grid{grid-template-columns:1fr}
.test-list{grid-template-columns:1fr}
.actions{flex-direction:column}
}
</style>
</head>
<body>
<div class="container">
<nav class="nav">
<div class="brand">
<div class="brand-orb">✦</div>
<span>AI Coding Agent</span>
</div>
<div class="online">
<span class="online-dot"></span>
AI Online
</div>
</nav>

<section class="hero">
<div class="big-orb">✦</div>
<h1>Build with Intelligence.</h1>
<p>Generate · Test · Execute Python with AI</p>
</section>

<main class="dashboard">
<section class="card">
<div class="card-title">
<h2>✦ Your Coding Task</h2>
<span class="label">Input</span>
</div>

<textarea id="task" placeholder="Describe what you want the AI to build...

Example: Create a Python program that checks whether a number is prime."></textarea>

<button class="run" id="runButton" onclick="runAgent()">
✦ Run AI Agent
</button>

<div id="status" class="status"></div>

<div class="steps">
<div class="step" id="step1">
<span class="step-icon">1</span>
Understanding task
</div>

<div class="step" id="step2">
<span class="step-icon">2</span>
Generating code
</div>

<div class="step" id="step3">
<span class="step-icon">3</span>
Testing code
</div>

<div class="step" id="step4">
<span class="step-icon">4</span>
Preparing report
</div>
</div>
</section>

<section class="card">
<div class="card-title">
<h2>⌘ Generated Code</h2>
<span class="label">Python</span>
</div>

<div class="code-window">
<div class="window-bar">
<span class="dot red"></span>
<span class="dot yellow"></span>
<span class="dot green"></span>
<span class="file-name">generated_code.py</span>
</div>

<pre id="code">Your generated Python code will appear here.</pre>
</div>

<div class="actions">
<button class="small-btn" onclick="copyCode()">Copy Code</button>
<button class="small-btn" onclick="downloadCode()">Download .py</button>
<button class="small-btn" onclick="runAgent()">Run Again</button>
</div>
</section>
</main>

<section class="card results">
<div class="card-title">
<h2>◈ Test & Execution</h2>
<span class="label">Results</span>
</div>

<div class="result-grid">
<div class="stat">
<div class="stat-number" id="totalTests">—</div>
<div class="stat-label">Tests</div>
</div>

<div class="stat passed">
<div class="stat-number" id="passedTests">—</div>
<div class="stat-label">Passed</div>
</div>

<div class="stat failed">
<div class="stat-number" id="failedTests">—</div>
<div class="stat-label">Failed</div>
</div>
</div>

<div class="test-list" id="testList">
<div class="test-item">Test scenarios will appear here</div>
</div>

<div class="output">
<div class="card-title">
<h2>Terminal Output</h2>
<span class="label">Execution</span>
</div>

<pre id="report">Your execution output and test report will appear here.</pre>
</div>

<div class="actions">
<button class="small-btn" onclick="runAgent()">Run Tests Again</button>
<button class="small-btn" onclick="runAgent()">Regenerate Code</button>
</div>
</section>

<div class="footer">
AI Coding Agent · Gemini · LangGraph · FastAPI
</div>
</div>

<script>
let lastCode="";

function setStep(step){
for(let i=1;i<=4;i++){
const element=document.getElementById("step"+i);
element.className="step";
}

if(step>0){
for(let i=1;i<step;i++){
document.getElementById("step"+i).className="step done";
}

document.getElementById("step"+step).className="step active";
}
}

function updateStats(report){
const scenarios=[
"Normal Case",
"Boundary Case",
"Edge Case",
"Invalid Input"
];

const total=4;
const lower=report.toLowerCase();

let failed=0;

if(
lower.includes("execution error") ||
lower.includes("traceback") ||
lower.includes("failed")
){
failed=1;
}

const passed=Math.max(total-failed,0);

document.getElementById("totalTests").innerText=total;
document.getElementById("passedTests").innerText=passed;
document.getElementById("failedTests").innerText=failed;

const list=document.getElementById("testList");
list.innerHTML="";

scenarios.forEach(name=>{
const div=document.createElement("div");
div.className="test-item";
div.innerText=name;
list.appendChild(div);
});
}

async function runAgent(){
const task=document.getElementById("task").value.trim();
const status=document.getElementById("status");
const code=document.getElementById("code");
const report=document.getElementById("report");
const button=document.getElementById("runButton");

if(!task){
status.innerText="Please enter a coding task.";
status.className="status error";
return;
}

button.disabled=true;
button.innerText="AI Agent Running...";
status.innerText="AI is working...";
status.className="status";

code.innerText="Generating code...";
report.innerText="Running tests...";

setStep(1);

setTimeout(()=>{
setStep(2);
},700);

setTimeout(()=>{
setStep(3);
},1600);

try{
const response=await fetch("/run",{
method:"POST",
headers:{
"Content-Type":"application/json"
},
body:JSON.stringify({
task:task
})
});

let data;

try{
data=await response.json();
}catch{
throw new Error("Server returned an invalid response.");
}

if(!response.ok){
throw new Error(data.detail||"Server error.");
}

lastCode=data.code||"";
code.innerText=lastCode||"No code generated.";
report.innerText=data.report||"No report generated.";

updateStats(data.report||"");

setStep(4);

setTimeout(()=>{
setStep(0);
status.innerText="AI Agent completed successfully.";
status.className="status success";
},500);

}catch(error){
setStep(0);
status.innerText=error.message;
status.className="status error";
code.innerText="";
report.innerText="";
}finally{
button.disabled=false;
button.innerText="✦ Run AI Agent";
}
}

function copyCode(){
if(!lastCode){
return;
}

navigator.clipboard.writeText(lastCode);

const status=document.getElementById("status");

status.innerText="Code copied to clipboard.";
status.className="status success";
}

function downloadCode(){
if(!lastCode){
return;
}

const blob=new Blob(
[lastCode],
{type:"text/plain"}
);

const url=URL.createObjectURL(blob);
const link=document.createElement("a");

link.href=url;
link.download="generated_code.py";
link.click();

URL.revokeObjectURL(url);
}
</script>
</body>
</html>
"""


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "gemini_configured": bool(api_key),
        "model": MODEL_NAME
    }


@app.post("/run")
def run_agent(request: TaskRequest):
    task = request.task.strip()

    if not task:
        raise HTTPException(
            status_code=400,
            detail="Task cannot be empty."
        )

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured on the server."
        )

    if llm is None:
        raise HTTPException(
            status_code=500,
            detail="Gemini could not be initialized. Check GEMINI_API_KEY."
        )

    try:
        initial_state: CrewState = {
            "messages": [HumanMessage(content=task)],
            "code": None,
            "report": None
        }

        result = rt_app.invoke(
            initial_state,
            config={"recursion_limit": 20}
        )

        return {
            "success": True,
            "task": task,
            "code": result.get("code", ""),
            "report": result.get("report", "")
        }

    except Exception as exc:
        print(traceback.format_exc())

        raise HTTPException(
            status_code=500,
            detail=f"AI Agent Error: {str(exc)}"
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False
    )
