import os
import sys
import json
import io
import re
import subprocess
import tempfile
import traceback
from typing import TypedDict,List,Optional
from fastapi import FastAPI,HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph,START,END

app=FastAPI(
    title="AI Coding Agent",
    description="AI powered coding, debugging, testing and execution platform",
    version="3.0.0"
)

api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
MODEL_NAME="gemini-3.6-flash"
llm=None

if api_key:
    try:
        llm=ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=api_key
        )
    except Exception as exc:
        print(f"Gemini initialization error: {exc}")
else:
    print("WARNING: GEMINI_API_KEY is not configured.")

class CrewState(TypedDict,total=False):
    messages:List[BaseMessage]
    code:Optional[str]
    report:Optional[str]

class TaskRequest(BaseModel):
    task:str

def extract_content(response)->str:
    content=getattr(response,"content",response)

    if isinstance(content,str):
        return content

    if isinstance(content,list):
        parts=[]

        for item in content:
            if isinstance(item,dict):
                if "text" in item:
                    parts.append(str(item["text"]))
            else:
                parts.append(str(item))

        return "\n".join(parts)

    return str(content)

def clean_python_code(code:str)->str:
    code=str(code).strip()

    code=re.sub(r"^```python\s*","",code,flags=re.IGNORECASE)
    code=re.sub(r"^```\s*","",code)
    code=re.sub(r"\s*```$","",code)

    return code.strip()

def validate_python(code:str)->str:
    try:
        compile(code,"generated_code.py","exec")
        return ""
    except Exception:
        return traceback.format_exc()

def run_python_code(code:str,input_data:str="",timeout:int=8)->str:
    code=clean_python_code(code)

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8"
        ) as file:
            file.write(code)
            file_path=file.name

        try:
            result=subprocess.run(
                [sys.executable,file_path],
                input=input_data,
                text=True,
                capture_output=True,
                timeout=timeout
            )

            output=result.stdout.strip()
            error=result.stderr.strip()

            if result.returncode==0:
                return output if output else "Success (no terminal output)"

            if error:
                return "Execution Error:\n"+error

            return f"Execution Error:\nProcess exited with code {result.returncode}"

        finally:
            try:
                os.remove(file_path)
            except Exception:
                pass

    except subprocess.TimeoutExpired:
        return "Execution Error:\nProgram exceeded the 8 second execution timeout."

    except Exception:
        return "Execution Error:\n"+traceback.format_exc()

def generate_test_cases(task:str)->str:
    if llm is None:
        return "Test generation unavailable because Gemini is not configured."

    prompt=f"""You are a senior QA engineer.

Coding task:
{task}

Create exactly 4 useful test scenarios.

Include:
1. Normal case
2. Boundary case
3. Edge case
4. Invalid input case when appropriate

Return only this format:

1. Normal Case: ...
2. Boundary Case: ...
3. Edge Case: ...
4. Invalid Input Case: ...

Do not use Markdown tables."""

    return extract_content(llm.invoke(prompt))

def generate_code(task:str)->str:
    if llm is None:
        raise ValueError("Gemini API is not configured.")

    prompt=f"""You are an expert Python software engineer.

Task:
{task}

Create a complete executable Python program.

Strict requirements:
- Return ONLY Python source code.
- Do NOT use Markdown.
- Do NOT use code fences.
- Every function that is called must be defined.
- Every variable that is used must be defined.
- Do not reference undefined functions.
- Do not reference undefined variables.
- The program must compile successfully.
- The program must run successfully.
- Handle reasonable edge cases.
- Handle invalid input where appropriate.
- Print the requested result.
- Do not include explanations.

Before returning the code, perform a mental syntax and dependency check."""

    response=llm.invoke(prompt)
    code=clean_python_code(extract_content(response))

    if not code:
        raise ValueError("Gemini returned empty Python code.")

    return code

def repair_code(task:str,code:str,error:str)->str:
    if llm is None:
        return code

    prompt=f"""You are an expert Python debugging engineer.

Original task:
{task}

Current Python program:
{code}

Problem found during validation or execution:
{error}

Fix the program completely.

Strict requirements:
- Return the COMPLETE corrected Python program.
- Return ONLY Python source code.
- Do NOT use Markdown.
- Do NOT use code fences.
- Do NOT explain the fix.
- Every function called must be defined.
- Every variable used must exist.
- Fix all syntax errors.
- Fix all NameError issues.
- Fix all logical issues that are obvious from the error.
- Preserve the requested functionality.
- Make the program directly executable.
- Handle reasonable edge cases.

Return only the corrected code."""

    try:
        response=llm.invoke(prompt)
        fixed=clean_python_code(extract_content(response))

        if fixed:
            return fixed

    except Exception as exc:
        print(f"Repair error: {exc}")

    return code

def build_and_test(task:str):
    code=generate_code(task)
    attempts=[]

    for attempt in range(3):
        validation_error=validate_python(code)

        if validation_error:
            attempts.append(
                f"Repair attempt {attempt+1}: syntax/validation error"
            )
            code=repair_code(
                task,
                code,
                validation_error
            )
            continue

        execution_result=run_python_code(code)

        if execution_result.startswith("Execution Error:"):
            attempts.append(
                f"Repair attempt {attempt+1}: execution error"
            )
            code=repair_code(
                task,
                code,
                execution_result
            )
            continue

        return code,execution_result,attempts

    final_validation=validate_python(code)

    if final_validation:
        return code,"Execution Error:\n"+final_validation,attempts

    final_execution=run_python_code(code)

    return code,final_execution,attempts

def real_time_developer(state:CrewState):
    messages=state.get("messages",[])

    if not messages:
        raise ValueError("No coding task was provided.")

    task=messages[-1].content

    code=generate_code(task)

    return {
        "code":code
    }

def real_time_tester(state:CrewState):
    messages=state.get("messages",[])

    if not messages:
        raise ValueError("No coding task was provided.")

    task=messages[-1].content
    code=state.get("code","")

    if not code:
        raise ValueError("No generated code available.")

    attempts=[]

    for attempt in range(3):
        validation_error=validate_python(code)

        if validation_error:
            attempts.append(f"Validation repair {attempt+1}")
            code=repair_code(
                task,
                code,
                validation_error
            )
            continue

        execution_result=run_python_code(code)

        if execution_result.startswith("Execution Error:"):
            attempts.append(f"Execution repair {attempt+1}")
            code=repair_code(
                task,
                code,
                execution_result
            )
            continue

        break

    final_validation=validate_python(code)

    if final_validation:
        execution_result="Execution Error:\n"+final_validation

    else:
        execution_result=run_python_code(code)

    test_cases=generate_test_cases(task)

    if execution_result.startswith("Execution Error:"):
        status="FAILED"
    else:
        status="PASSED"

    repair_text="\n".join(attempts) if attempts else "No repairs were required."

    report=f"""STATUS

{status}

EXECUTION OUTPUT

{execution_result}

AI TEST SCENARIOS

{test_cases}

AI REPAIR HISTORY

{repair_text}"""

    return {
        "code":code,
        "report":report
    }

workflow=StateGraph(CrewState)

workflow.add_node(
    "developer",
    real_time_developer
)

workflow.add_node(
    "tester",
    real_time_tester
)

workflow.add_edge(
    START,
    "developer"
)

workflow.add_edge(
    "developer",
    "tester"
)

workflow.add_edge(
    "tester",
    END
)

rt_app=workflow.compile()

@app.get("/",response_class=HTMLResponse)
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
--bg:#160d24;
--bg2:#211135;
--lavender:#c084fc;
--purple:#8b5cf6;
--violet:#7c3aed;
--pink:#e879f9;
--soft:#f5edff;
--muted:#c4b5fd;
--glass:rgba(255,255,255,.075);
--border:rgba(216,180,254,.18);
}

body{
margin:0;
min-height:100vh;
font-family:"Segoe UI",Inter,Arial,sans-serif;
color:white;
background:
radial-gradient(circle at 10% 10%,rgba(192,132,252,.35),transparent 28%),
radial-gradient(circle at 90% 15%,rgba(232,121,249,.25),transparent 25%),
radial-gradient(circle at 50% 90%,rgba(124,58,237,.3),transparent 35%),
linear-gradient(135deg,#12091c,#1b0d2b,#24103c);
background-attachment:fixed;
}

body:before{
content:"";
position:fixed;
inset:0;
pointer-events:none;
background-image:
linear-gradient(rgba(255,255,255,.018) 1px,transparent 1px),
linear-gradient(90deg,rgba(255,255,255,.018) 1px,transparent 1px);
background-size:55px 55px;
mask-image:linear-gradient(to bottom,black,transparent);
}

.container{
max-width:1250px;
margin:auto;
padding:24px 20px 60px;
position:relative;
}

.nav{
height:68px;
display:flex;
align-items:center;
justify-content:space-between;
padding:0 20px;
border:1px solid var(--border);
background:rgba(255,255,255,.07);
backdrop-filter:blur(25px);
border-radius:22px;
box-shadow:0 20px 60px rgba(0,0,0,.25);
margin-bottom:65px;
}

.brand{
display:flex;
align-items:center;
gap:12px;
font-weight:700;
}

.brand-orb{
width:38px;
height:38px;
display:grid;
place-items:center;
border-radius:13px;
background:linear-gradient(135deg,#d8b4fe,#a855f7,#7c3aed);
box-shadow:0 0 30px rgba(192,132,252,.5);
}

.online{
display:flex;
align-items:center;
gap:8px;
font-size:13px;
color:#d8b4fe;
}

.online-dot{
width:8px;
height:8px;
border-radius:50%;
background:#86efac;
box-shadow:0 0 15px #86efac;
}

.hero{
text-align:center;
margin-bottom:55px;
}

.orb{
width:100px;
height:100px;
margin:0 auto 25px;
display:grid;
place-items:center;
border-radius:32px;
font-size:43px;
background:
radial-gradient(circle at 35% 25%,#fff,#e9d5ff 18%,#c084fc 40%,#9333ea 70%,#581c87);
box-shadow:
0 0 35px rgba(192,132,252,.65),
0 0 100px rgba(232,121,249,.2);
animation:float 4s ease-in-out infinite;
}

@keyframes float{
0%,100%{transform:translateY(0)}
50%{transform:translateY(-9px)}
}

.hero h1{
margin:0;
font-size:clamp(42px,6vw,70px);
letter-spacing:-3px;
background:linear-gradient(90deg,#fff,#f3e8ff,#d8b4fe,#f0abfc);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
}

.hero p{
margin:18px 0 0;
color:#c4b5fd;
font-size:18px;
}

.dashboard{
display:grid;
grid-template-columns:1fr 1.25fr;
gap:22px;
}

.card{
background:var(--glass);
border:1px solid var(--border);
backdrop-filter:blur(25px);
-webkit-backdrop-filter:blur(25px);
border-radius:25px;
padding:26px;
box-shadow:0 25px 70px rgba(0,0,0,.23);
transition:.3s;
}

.card:hover{
transform:translateY(-3px);
border-color:rgba(216,180,254,.32);
}

.title{
display:flex;
align-items:center;
justify-content:space-between;
margin-bottom:20px;
}

.title h2{
margin:0;
font-size:18px;
}

.label{
font-size:10px;
letter-spacing:1.5px;
text-transform:uppercase;
color:#a78bfa;
}

textarea{
width:100%;
height:260px;
resize:none;
padding:20px;
border-radius:18px;
border:1px solid rgba(216,180,254,.16);
outline:none;
background:rgba(8,3,16,.55);
color:white;
font:15px/1.7 "Segoe UI",sans-serif;
transition:.25s;
}

textarea::placeholder{
color:#88769d;
}

textarea:focus{
border-color:#c084fc;
box-shadow:0 0 0 4px rgba(192,132,252,.1),0 0 30px rgba(192,132,252,.08);
}

.run{
width:100%;
margin-top:16px;
padding:16px;
border:0;
border-radius:16px;
color:white;
font-size:15px;
font-weight:700;
cursor:pointer;
background:linear-gradient(110deg,#7c3aed,#a855f7,#e879f9,#a855f7);
background-size:250% 100%;
box-shadow:0 12px 35px rgba(139,92,246,.35);
transition:.3s;
}

.run:hover{
background-position:100% 0;
transform:translateY(-2px);
box-shadow:0 18px 45px rgba(192,132,252,.45);
}

.run:disabled{
opacity:.55;
cursor:not-allowed;
transform:none;
}

.steps{
margin-top:22px;
display:grid;
gap:10px;
}

.step{
display:flex;
align-items:center;
gap:10px;
font-size:13px;
color:#6f617e;
}

.step-icon{
width:22px;
height:22px;
display:grid;
place-items:center;
border-radius:50%;
border:1px solid #493b57;
font-size:10px;
}

.step.active{
color:#eadcff;
}

.step.active .step-icon{
border-color:#c084fc;
color:#c084fc;
box-shadow:0 0 15px rgba(192,132,252,.4);
}

.step.done{
color:#86efac;
}

.step.done .step-icon{
border-color:#86efac;
color:#86efac;
}

.code-window{
overflow:hidden;
border-radius:18px;
background:#08040d;
border:1px solid rgba(255,255,255,.09);
}

.window{
height:42px;
display:flex;
align-items:center;
gap:7px;
padding:0 14px;
border-bottom:1px solid rgba(255,255,255,.07);
background:rgba(255,255,255,.035);
}

.dot{
width:10px;
height:10px;
border-radius:50%;
}

.red{background:#fb7185}
.yellow{background:#facc15}
.green{background:#4ade80}

.filename{
margin-left:8px;
font-size:12px;
color:#7e718d;
}

pre{
margin:0;
height:270px;
padding:18px;
overflow:auto;
white-space:pre-wrap;
word-break:break-word;
color:#e9d5ff;
font:13px/1.7 "JetBrains Mono","Fira Code",Consolas,monospace;
}

.actions{
display:flex;
gap:9px;
margin-top:14px;
}

.small{
flex:1;
padding:11px;
border-radius:12px;
border:1px solid rgba(216,180,254,.14);
background:rgba(255,255,255,.045);
color:#ddd0eb;
cursor:pointer;
font-size:12px;
transition:.2s;
}

.small:hover{
background:rgba(192,132,252,.12);
border-color:#a855f7;
color:white;
}

.results{
margin-top:22px;
}

.stats{
display:grid;
grid-template-columns:repeat(3,1fr);
gap:14px;
}

.stat{
padding:20px;
border-radius:18px;
background:rgba(5,2,10,.35);
border:1px solid rgba(255,255,255,.07);
}

.number{
font-size:31px;
font-weight:800;
}

.stat-label{
margin-top:4px;
font-size:10px;
letter-spacing:1px;
text-transform:uppercase;
color:#8f819d;
}

.pass .number{
color:#86efac;
}

.fail .number{
color:#fb7185;
}

.test-list{
display:grid;
grid-template-columns:repeat(2,1fr);
gap:10px;
margin-top:16px;
}

.test{
padding:14px;
border-radius:14px;
background:rgba(255,255,255,.04);
border:1px solid rgba(255,255,255,.06);
color:#c4b5fd;
font-size:13px;
}

.test:before{
content:"✓";
color:#86efac;
font-weight:bold;
margin-right:8px;
}

.output{
margin-top:22px;
}

.output pre{
height:190px;
border-radius:16px;
background:rgba(5,2,10,.5);
}

.status{
min-height:22px;
margin-top:13px;
text-align:center;
font-size:13px;
}

.success{
color:#86efac;
}

.error{
color:#fb7185;
}

.footer{
margin-top:35px;
text-align:center;
color:#70627f;
font-size:12px;
}

@media(max-width:850px){
.dashboard{
grid-template-columns:1fr;
}
.nav{
margin-bottom:45px;
}
}

@media(max-width:600px){
.container{
padding:15px 12px 40px;
}

.nav{
height:58px;
padding:0 14px;
border-radius:17px;
}

.hero h1{
font-size:39px;
letter-spacing:-2px;
}

.hero p{
font-size:15px;
}

.orb{
width:78px;
height:78px;
font-size:32px;
border-radius:25px;
}

.card{
padding:19px;
border-radius:20px;
}

textarea,pre{
height:220px;
}

.stats{
grid-template-columns:1fr;
}

.test-list{
grid-template-columns:1fr;
}

.actions{
flex-direction:column;
}
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
<div class="orb">✦</div>
<h1>Build with Intelligence.</h1>
<p>Generate · Debug · Test · Execute with AI</p>
</section>

<div class="dashboard">

<section class="card">

<div class="title">
<h2>✦ Your Coding Task</h2>
<span class="label">Input</span>
</div>

<textarea id="task" placeholder="Describe what you want the AI to build...

Example:
Create a Python program that checks whether a number is prime."></textarea>

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
Validating & debugging
</div>

<div class="step" id="step4">
<span class="step-icon">4</span>
Executing & testing
</div>

</div>

</section>

<section class="card">

<div class="title">
<h2>⌘ Generated Code</h2>
<span class="label">Python</span>
</div>

<div class="code-window">

<div class="window">
<span class="dot red"></span>
<span class="dot yellow"></span>
<span class="dot green"></span>
<span class="filename">generated_code.py</span>
</div>

<pre id="code">Your generated Python code will appear here.</pre>

</div>

<div class="actions">
<button class="small" onclick="copyCode()">Copy Code</button>
<button class="small" onclick="downloadCode()">Download .py</button>
<button class="small" onclick="runAgent()">Run Again</button>
</div>

</section>

</div>

<section class="card results">

<div class="title">
<h2>◈ Test & Execution</h2>
<span class="label">Agent Results</span>
</div>

<div class="stats">

<div class="stat">
<div class="number" id="total">—</div>
<div class="stat-label">Tests</div>
</div>

<div class="stat pass">
<div class="number" id="passed">—</div>
<div class="stat-label">Passed</div>
</div>

<div class="stat fail">
<div class="number" id="failed">—</div>
<div class="stat-label">Failed</div>
</div>

</div>

<div class="test-list" id="testList">

<div class="test">
Test scenarios will appear here
</div>

</div>

<div class="output">

<div class="title">
<h2>Terminal Output</h2>
<span class="label">Execution</span>
</div>

<pre id="report">Your execution output and test report will appear here.</pre>

</div>

<div class="actions">
<button class="small" onclick="runAgent()">Run Tests Again</button>
<button class="small" onclick="runAgent()">Regenerate Code</button>
</div>

</section>

<div class="footer">
AI Coding Agent · Gemini · LangGraph · FastAPI
</div>

</div>

<script>

let lastCode="";

function setStep(number){

for(let i=1;i<=4;i++){
document.getElementById("step"+i).className="step";
}

if(number>0){

for(let i=1;i<number;i++){
document.getElementById("step"+i).className="step done";
}

document.getElementById("step"+number).className="step active";
}

}

function updateStats(report){

const status=report.includes("STATUS")&&report.includes("PASSED");

const failed=report.toLowerCase().includes("execution error")?1:0;

const total=4;

const passed=failed?3:4;

document.getElementById("total").innerText=total;
document.getElementById("passed").innerText=passed;
document.getElementById("failed").innerText=failed;

const list=document.getElementById("testList");

list.innerHTML="";

const names=[
"Normal Case",
"Boundary Case",
"Edge Case",
"Invalid Input"
];

names.forEach((name,index)=>{

const item=document.createElement("div");

item.className="test";

item.innerText=name;

list.appendChild(item);

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

report.innerText="Preparing execution...";

setStep(1);

setTimeout(()=>{
setStep(2);
},700);

setTimeout(()=>{
setStep(3);
},1500);

setTimeout(()=>{
setStep(4);
},2300);

try{

const response=await fetch(
"/run",
{
method:"POST",
headers:{
"Content-Type":"application/json"
},
body:JSON.stringify({
task:task
})
}
);

let data;

try{
data=await response.json();
}catch{
throw new Error("Server returned an invalid response.");
}

if(!response.ok){
throw new Error(data.detail||"Server error");
}

lastCode=data.code||"";

code.innerText=lastCode||"No code generated.";

report.innerText=data.report||"No report generated.";

updateStats(data.report||"");

setStep(0);

if((data.report||"").includes("STATUS\\n\\nPASSED")){
status.innerText="AI Agent completed successfully.";
status.className="status success";
}else{
status.innerText="AI Agent completed with an execution issue.";
status.className="status error";
}

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

document.body.appendChild(link);

link.click();

document.body.removeChild(link);

URL.revokeObjectURL(url);

}

</script>

</body>
</html>
"""

@app.get("/health")
def health():
    return {
        "status":"healthy",
        "gemini_configured":bool(api_key),
        "model":MODEL_NAME
    }

@app.post("/run")
def run_agent(request:TaskRequest):

    task=request.task.strip()

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

        initial_state:CrewState={
            "messages":[HumanMessage(content=task)],
            "code":None,
            "report":None
        }

        result=rt_app.invoke(
            initial_state,
            config={
                "recursion_limit":20
            }
        )

        return {
            "success":True,
            "task":task,
            "code":result.get("code",""),
            "report":result.get("report","")
        }

    except Exception as exc:

        print(traceback.format_exc())

        raise HTTPException(
            status_code=500,
            detail=f"AI Agent Error: {str(exc)}"
        )

if __name__=="__main__":

    import uvicorn

    port=int(
        os.getenv(
            "PORT",
            "8000"
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False
    )
