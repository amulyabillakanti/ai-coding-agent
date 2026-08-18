import os,sys,io,traceback
from typing import TypedDict,List,Optional
from fastapi import FastAPI,HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph,START,END

app=FastAPI(title="AI Coding Agent",description="AI-powered coding, testing and execution pipeline",version="2.0.0")

api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
MODEL_NAME="gemini-3.6-flash"
llm=None

if api_key:
    try:
        llm=ChatGoogleGenerativeAI(model=MODEL_NAME,google_api_key=api_key)
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
    if code.startswith("```python"):
        code=code[len("```python"):]
    elif code.startswith("```Python"):
        code=code[len("```Python"):]
    elif code.startswith("```"):
        code=code[3:]
    if code.endswith("```"):
        code=code[:-3]
    return code.strip()

def run_python_code(code:str)->str:
    clean_code=clean_python_code(str(code))
    old_stdout=sys.stdout
    new_stdout=io.StringIO()
    sys.stdout=new_stdout
    try:
        exec(clean_code,{"__name__":"__main__"},{})
        result=new_stdout.getvalue()
    except Exception:
        result="Execution Error:\n"+traceback.format_exc()
    finally:
        sys.stdout=old_stdout
    result=result.strip()
    return result if result else "Success (no terminal output)"

def generate_test_cases(task:str)->str:
    if llm is None:
        return "Test generation unavailable because Gemini is not configured."
    prompt=f"""You are a Senior QA Engineer.
Generate 3 to 5 highly specific test scenarios for this coding task:
{task}
Include:
1. Normal cases
2. Boundary cases
3. Edge cases
4. Invalid input cases where appropriate
Return only a numbered list."""
    return extract_content(llm.invoke(prompt))

def real_time_developer(state:CrewState):
    if llm is None:
        raise ValueError("Gemini API is not configured. Set GEMINI_API_KEY.")
    messages=state.get("messages",[])
    if not messages:
        raise ValueError("No coding task was provided.")
    task=messages[-1].content
    prompt=f"""You are an expert Python developer.
Write a complete executable Python program for this task:
{task}
Requirements:
- Return ONLY Python source code.
- Do NOT use Markdown.
- Do NOT include code fences.
- The program must run directly with Python.
- Use clear variable names.
- Handle reasonable edge cases.
- Print the result when output is required."""
    code=clean_python_code(extract_content(llm.invoke(prompt)))
    if not code:
        raise ValueError("Gemini returned empty Python code.")
    return {"code":code}

def real_time_tester(state:CrewState):
    messages=state.get("messages",[])
    if not messages:
        raise ValueError("No coding task available.")
    task=messages[-1].content
    code=state.get("code","")
    if not code:
        raise ValueError("No generated code available.")
    tests=generate_test_cases(task)
    output=run_python_code(code)
    report=f"""EXECUTION OUTPUT

{output}

TEST SCENARIOS

{tests}"""
    return {"report":report}

workflow=StateGraph(CrewState)
workflow.add_node("developer",real_time_developer)
workflow.add_node("tester",real_time_tester)
workflow.add_edge(START,"developer")
workflow.add_edge("developer","tester")
workflow.add_edge("tester",END)
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
body{margin:0;min-height:100vh;font-family:Inter,"Segoe UI",Arial,sans-serif;color:#fff;background:radial-gradient(circle at 10% 10%,rgba(168,85,247,.45),transparent 30%),radial-gradient(circle at 90% 85%,rgba(192,38,211,.4),transparent 30%),linear-gradient(135deg,#160b2e,#24103d,#351052,#1b0b32);background-attachment:fixed}
.container{width:100%;max-width:1100px;margin:auto;padding:50px 20px 60px}
.hero{text-align:center;margin-bottom:40px}
.hero-icon{width:80px;height:80px;margin:0 auto 20px;display:flex;align-items:center;justify-content:center;border-radius:24px;font-size:38px;background:linear-gradient(135deg,#c084fc,#9333ea,#7e22ce);box-shadow:0 15px 50px rgba(168,85,247,.5)}
h1{margin:0;font-size:46px;font-weight:800;background:linear-gradient(90deg,#fff,#e9d5ff,#c084fc);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.subtitle{margin-top:12px;color:#ddd6fe;font-size:18px}
.card{background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.16);backdrop-filter:blur(20px);padding:28px;border-radius:22px;margin-bottom:25px;box-shadow:0 20px 50px rgba(0,0,0,.25)}
.card h2{margin:0 0 18px;font-size:22px;color:#f5f3ff}
textarea{width:100%;min-height:170px;padding:18px;resize:vertical;border-radius:14px;border:1px solid rgba(196,181,253,.3);background:rgba(10,4,25,.7);color:#fff;font-size:16px;line-height:1.6;outline:none}
textarea::placeholder{color:#a78bfa}
textarea:focus{border-color:#c084fc;box-shadow:0 0 0 4px rgba(168,85,247,.16)}
button{width:100%;margin-top:18px;padding:16px;border:0;border-radius:14px;background:linear-gradient(135deg,#c026d3,#9333ea,#7c3aed);color:#fff;font-size:17px;font-weight:700;cursor:pointer;box-shadow:0 10px 30px rgba(124,58,237,.45)}
button:hover{transform:translateY(-2px);box-shadow:0 15px 40px rgba(168,85,247,.55)}
button:disabled{opacity:.55;cursor:not-allowed;transform:none}
.status{margin-top:16px;text-align:center;font-weight:600;min-height:24px}
.success{color:#86efac}
.error{color:#fca5a5}
pre{margin:0;min-height:130px;padding:20px;border-radius:14px;overflow-x:auto;white-space:pre-wrap;word-wrap:break-word;background:rgba(5,2,15,.82);border:1px solid rgba(167,139,250,.18);color:#e9d5ff;font-family:"Fira Code",Consolas,monospace;font-size:14px;line-height:1.6}
.footer{text-align:center;margin-top:30px;color:#c4b5fd;font-size:14px}
@media(max-width:700px){.container{padding:30px 15px 50px}h1{font-size:34px}.subtitle{font-size:15px}.card{padding:20px}.hero-icon{width:65px;height:65px;font-size:30px}}
</style>
</head>
<body>
<div class="container">
<div class="hero">
<div class="hero-icon">🤖</div>
<h1>AI Coding Agent</h1>
<p class="subtitle">Generate • Test • Execute Python Code using AI</p>
</div>
<div class="card">
<h2>📝 Enter Coding Task</h2>
<textarea id="task" placeholder="Example: Write a Python program to check whether a number is prime."></textarea>
<button id="runButton" onclick="runAgent()">🚀 Run AI Agent</button>
<div id="status" class="status"></div>
</div>
<div class="card">
<h2>💻 Generated Code</h2>
<pre id="code">Your generated code will appear here.</pre>
</div>
<div class="card">
<h2>🧪 Test & Execution Report</h2>
<pre id="report">Your test report will appear here.</pre>
</div>
<div class="footer">Powered by Gemini + LangGraph + FastAPI</div>
</div>
<script>
async function runAgent(){
const task=document.getElementById("task").value.trim();
const status=document.getElementById("status");
const code=document.getElementById("code");
const report=document.getElementById("report");
const button=document.getElementById("runButton");
if(!task){
status.innerText="⚠️ Please enter a coding task.";
status.className="status error";
return;
}
button.disabled=true;
button.innerText="⏳ AI Agent Running...";
status.innerText="⏳ Generating code and running tests...";
status.className="status";
code.innerText="Generating code...";
report.innerText="Running tests...";
try{
const response=await fetch("/run",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({task:task})});
let data;
try{data=await response.json()}catch{throw new Error("Server returned an invalid response.")}
if(!response.ok)throw new Error(data.detail||"Server error");
code.innerText=data.code||"No code generated.";
report.innerText=data.report||"No report generated.";
status.innerText="✅ AI Agent completed successfully.";
status.className="status success";
}catch(error){
status.innerText="❌ "+error.message;
status.className="status error";
code.innerText="";
report.innerText="";
}finally{
button.disabled=false;
button.innerText="🚀 Run AI Agent";
}
}
</script>
</body>
</html>
"""

@app.get("/health")
def health():
    return {"status":"healthy","gemini_configured":bool(api_key),"model":MODEL_NAME}

@app.post("/run")
def run_agent(request:TaskRequest):
    task=request.task.strip()
    if not task:
        raise HTTPException(status_code=400,detail="Task cannot be empty.")
    if not api_key:
        raise HTTPException(status_code=500,detail="GEMINI_API_KEY is not configured on the server.")
    if llm is None:
        raise HTTPException(status_code=500,detail="Gemini could not be initialized. Check GEMINI_API_KEY.")
    try:
        initial_state:CrewState={"messages":[HumanMessage(content=task)],"code":None,"report":None}
        result=rt_app.invoke(initial_state,config={"recursion_limit":20})
        return {"success":True,"task":task,"code":result.get("code",""),"report":result.get("report","")}
    except Exception as exc:
        print(traceback.format_exc())
        raise HTTPException(status_code=500,detail=f"AI Agent Error: {str(exc)}")

if __name__=="__main__":
    import uvicorn
    port=int(os.getenv("PORT","8000"))
    uvicorn.run(app,host="0.0.0.0",port=port,reload=False)
