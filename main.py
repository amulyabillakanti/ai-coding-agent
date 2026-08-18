import io
import os
import sys
import traceback
from typing import List, Optional, TypedDict

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel


# ============================================================
# 1. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI Coding Agent",
    description="AI-powered coding, testing and execution pipeline",
    version="2.0.0",
)


# ============================================================
# 2. GEMINI CONFIGURATION
# ============================================================

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)

llm = None

if api_key:
    try:
        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=api_key,
        )
    except Exception as exc:
        print(f"Gemini initialization error: {exc}")
        llm = None
else:
    print("WARNING: GEMINI_API_KEY is not configured.")


# ============================================================
# 3. STATE
# ============================================================

class CrewState(TypedDict, total=False):
    messages: List[BaseMessage]
    code: Optional[str]
    report: Optional[str]


# ============================================================
# 4. REQUEST MODEL
# ============================================================

class TaskRequest(BaseModel):
    task: str


# ============================================================
# 5. HELPERS
# ============================================================

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

            else:
                parts.append(str(item))

        return "\n".join(parts)

    return str(content)


def clean_python_code(code: str) -> str:

    code = str(code).strip()

    if code.startswith("```python"):
        code = code[len("```python"):]

    elif code.startswith("```Python"):
        code = code[len("```Python"):]

    elif code.startswith("```"):
        code = code[3:]

    if code.endswith("```"):
        code = code[:-3]

    return code.strip()


# ============================================================
# 6. PYTHON EXECUTION
# ============================================================

def run_python_code(code: str) -> str:

    if not isinstance(code, str):
        code = str(code)

    clean_code = clean_python_code(code)

    old_stdout = sys.stdout
    new_stdout = io.StringIO()

    sys.stdout = new_stdout

    try:

        local_scope = {}

        exec(
            clean_code,
            {
                "__name__": "__main__",
            },
            local_scope,
        )

        result = new_stdout.getvalue()

    except Exception:

        result = (
            "Execution Error:\n"
            + traceback.format_exc()
        )

    finally:

        sys.stdout = old_stdout

    result = result.strip()

    if result:
        return result

    return "Success (no terminal output)"


# ============================================================
# 7. GENERATE TEST CASES
# ============================================================

def generate_test_cases(task_description: str) -> str:

    if llm is None:

        return (
            "Test generation unavailable because Gemini "
            "is not configured."
        )

    prompt = f"""
You are a Senior QA Engineer.

Generate 3 to 5 highly specific test scenarios
for this Python coding task:

{task_description}

Include:

1. Normal cases
2. Boundary cases
3. Edge cases
4. Invalid input cases where appropriate

Return only a numbered list.
Do not write Python code.
Do not add explanations outside the numbered list.
"""

    response = llm.invoke(prompt)

    return extract_content(response)


# ============================================================
# 8. DEVELOPER NODE
# ============================================================

def real_time_developer(state: CrewState):

    print("[Developer] Generating Python code...")

    if llm is None:

        raise ValueError(
            "Gemini API is not configured. "
            "Set GEMINI_API_KEY in Render Environment Variables."
        )

    messages = state.get("messages", [])

    if not messages:
        raise ValueError("No coding task was provided.")

    task = messages[-1].content

    developer_prompt = f"""
You are an expert Python developer.

Write a complete, executable Python program for:

{task}

Requirements:

- Return ONLY Python source code.
- Do NOT use Markdown.
- Do NOT include ```python.
- The program must be executable directly with Python.
- Use clear variable names.
- Handle reasonable edge cases.
- If the task asks for output, print the result.
- Do not explain the code outside the Python source.
- Do not install external packages.
- Prefer Python standard library functionality.
- Avoid interactive input unless specifically requested.
"""

    response = llm.invoke(developer_prompt)

    code_str = clean_python_code(
        extract_content(response)
    )

    if not code_str:
        raise ValueError(
            "Gemini returned empty Python code."
        )

    print("\nGenerated Code:")
    print(code_str)

    return {
        "code": code_str
    }


# ============================================================
# 9. TESTER NODE
# ============================================================

def real_time_tester(state: CrewState):

    print(
        "[Tester] Generating tests and executing code..."
    )

    messages = state.get("messages", [])

    if not messages:
        raise ValueError(
            "No task available for testing."
        )

    task = messages[-1].content

    generated_code = state.get("code", "")

    if not generated_code:
        raise ValueError(
            "No generated code available for testing."
        )

    test_cases = generate_test_cases(task)

    execution_result = run_python_code(
        generated_code
    )

    report = (
        "### EXECUTION OUTPUT\n\n"
        f"{execution_result}\n\n"
        "### TEST SCENARIOS\n\n"
        f"{test_cases}"
    )

    print("\nTest Report:")
    print(report)

    return {
        "report": report
    }


# ============================================================
# 10. LANGGRAPH WORKFLOW
# ============================================================

workflow = StateGraph(CrewState)

workflow.add_node(
    "developer",
    real_time_developer,
)

workflow.add_node(
    "tester",
    real_time_tester,
)

workflow.add_edge(
    START,
    "developer",
)

workflow.add_edge(
    "developer",
    "tester",
)

workflow.add_edge(
    "tester",
    END,
)

rt_app = workflow.compile()


# ============================================================
# 11. LIGHT BLUE AESTHETIC FRONTEND
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home():

    return """
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>AI Coding Agent 💙</title>


<link
    rel="preconnect"
    href="https://fonts.googleapis.com"
>

<link
    rel="preconnect"
    href="https://fonts.gstatic.com"
    crossorigin
>

<link
    href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Pacifico&display=swap"
    rel="stylesheet"
>


<style>

/* =========================================================
   BASIC
   ========================================================= */

* {
    box-sizing: border-box;
}

html {
    scroll-behavior: smooth;
}

body {

    margin: 0;

    min-height: 100vh;

    font-family: "DM Sans", sans-serif;

    color: #29445c;

    background:

        radial-gradient(
            circle at 5% 5%,
            rgba(186, 230, 253, 0.85),
            transparent 30%
        ),

        radial-gradient(
            circle at 95% 10%,
            rgba(191, 219, 254, 0.8),
            transparent 30%
        ),

        radial-gradient(
            circle at 50% 100%,
            rgba(224, 242, 254, 0.95),
            transparent 40%
        ),

        linear-gradient(
            135deg,
            #f7fcff,
            #e9f6ff 45%,
            #dfefff
        );

    background-attachment: fixed;

    overflow-x: hidden;
}


/* =========================================================
   DECORATIVE ELEMENTS
   ========================================================= */

.decor {

    position: fixed;

    pointer-events: none;

    z-index: 0;

    user-select: none;
}

.decor.one {

    top: 8%;

    left: 4%;

    color: rgba(96, 165, 250, 0.28);

    font-size: 42px;

    animation:
        floatOne 5s ease-in-out infinite;
}

.decor.two {

    top: 30%;

    right: 5%;

    color: rgba(56, 189, 248, 0.27);

    font-size: 34px;

    animation:
        floatTwo 6s ease-in-out infinite;
}

.decor.three {

    bottom: 10%;

    left: 7%;

    color: rgba(125, 211, 252, 0.4);

    font-size: 32px;

    animation:
        floatOne 7s ease-in-out infinite;
}


/* =========================================================
   CONTAINER
   ========================================================= */

.container {

    position: relative;

    z-index: 1;

    width: min(1050px, 92%);

    margin: auto;

    padding: 55px 0 70px;
}


/* =========================================================
   HERO
   ========================================================= */

.hero {

    text-align: center;

    margin-bottom: 42px;
}


.logo {

    width: 90px;

    height: 90px;

    margin: 0 auto 20px;

    display: flex;

    align-items: center;

    justify-content: center;

    border-radius: 30px;

    background:

        linear-gradient(
            135deg,
            #7dd3fc,
            #60a5fa
        );

    box-shadow:

        0 20px 45px
        rgba(59, 130, 246, 0.23);

    font-size: 45px;

    animation:
        logoFloat 4s ease-in-out infinite;

    transform: rotate(-4deg);
}

.logo:hover {

    transform:
        rotate(4deg)
        scale(1.08);
}


h1 {

    margin: 0;

    font-size: clamp(42px, 7vw, 68px);

    font-weight: 700;

    letter-spacing: -2px;

    color: #285477;
}


.brand-script {

    font-family: "Pacifico", cursive;

    color: #3b82f6;

    font-size: 0.72em;
}


.subtitle {

    margin: 13px 0 0;

    color: #62809a;

    font-size: 18px;

    font-weight: 500;
}


.tagline {

    display: inline-block;

    margin-top: 20px;

    padding: 9px 18px;

    border-radius: 999px;

    background:
        rgba(255, 255, 255, 0.72);

    border:
        1px solid
        rgba(255, 255, 255, 0.95);

    color: #4780aa;

    font-size: 14px;

    box-shadow:
        0 8px 25px
        rgba(50, 110, 160, 0.10);

    backdrop-filter: blur(10px);
}


/* =========================================================
   GLASS CARDS
   ========================================================= */

.card {

    margin-bottom: 26px;

    padding: 30px;

    border-radius: 30px;

    background:
        rgba(255,255,255,0.70);

    border:
        1px solid
        rgba(255,255,255,0.96);

    box-shadow:

        0 25px 60px
        rgba(60, 110, 150, 0.12);

    backdrop-filter: blur(18px);

    -webkit-backdrop-filter: blur(18px);

    transition:
        transform 0.3s ease,
        box-shadow 0.3s ease;
}


.card:hover {

    transform: translateY(-3px);

    box-shadow:

        0 30px 70px
        rgba(60, 110, 150, 0.17);
}


.card-header {

    display: flex;

    align-items: center;

    gap: 13px;

    margin-bottom: 19px;
}


.card-icon {

    width: 46px;

    height: 46px;

    display: flex;

    align-items: center;

    justify-content: center;

    border-radius: 15px;

    background:

        linear-gradient(
            135deg,
            #dff5ff,
            #dbeafe
        );

    font-size: 23px;

    box-shadow:

        0 7px 20px
        rgba(60, 130, 180, 0.10);
}


h2 {

    margin: 0;

    color: #315d7c;

    font-size: 22px;

    font-weight: 700;
}


/* =========================================================
   TEXTAREA
   ========================================================= */

textarea {

    width: 100%;

    min-height: 180px;

    padding: 20px;

    border-radius: 21px;

    border:
        2px solid
        #bfdbfe;

    outline: none;

    resize: vertical;

    background:
        rgba(248, 252, 255, 0.94);

    color: #29445c;

    font-family:
        "DM Sans",
        sans-serif;

    font-size: 16px;

    line-height: 1.65;

    transition:
        border-color 0.25s ease,
        box-shadow 0.25s ease,
        background 0.25s ease;
}


textarea::placeholder {

    color: #8aa9bf;
}


textarea:focus {

    border-color: #60a5fa;

    background: white;

    box-shadow:

        0 0 0 5px
        rgba(96, 165, 250, 0.12);
}


/* =========================================================
   BUTTON
   ========================================================= */

button {

    width: 100%;

    margin-top: 18px;

    padding: 17px 25px;

    border: none;

    border-radius: 19px;

    background:

        linear-gradient(
            135deg,
            #60a5fa,
            #38bdf8
        );

    color: white;

    font-family:
        "DM Sans",
        sans-serif;

    font-size: 17px;

    font-weight: 700;

    cursor: pointer;

    box-shadow:

        0 14px 30px
        rgba(59, 130, 246, 0.24);

    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease,
        filter 0.2s ease;
}


button:hover {

    transform: translateY(-2px);

    filter: brightness(1.04);

    box-shadow:

        0 18px 38px
        rgba(59, 130, 246, 0.32);
}


button:active {

    transform: translateY(1px);
}


button:disabled {

    background:

        linear-gradient(
            135deg,
            #a9bdce,
            #a7c3d4
        );

    cursor: not-allowed;

    box-shadow: none;

    transform: none;
}


/* =========================================================
   STATUS
   ========================================================= */

.status {

    min-height: 25px;

    margin-top: 17px;

    text-align: center;

    font-size: 14px;

    font-weight: 700;
}


.success {

    color: #16805b;
}


.error {

    color: #d14f6d;
}


/* =========================================================
   OUTPUT
   ========================================================= */

.output-wrapper {

    position: relative;
}


.output-label {

    position: absolute;

    top: 13px;

    right: 14px;

    padding: 5px 11px;

    border-radius: 999px;

    background:
        rgba(147, 197, 253, 0.13);

    border:
        1px solid
        rgba(147, 197, 253, 0.18);

    color: #a9d8f7;

    font-size: 10px;

    font-weight: 700;

    letter-spacing: 1px;
}


pre {

    margin: 0;

    min-height: 160px;

    padding: 27px 22px;

    border-radius: 21px;

    overflow-x: auto;

    white-space: pre-wrap;

    word-break: break-word;

    background:

        linear-gradient(
            145deg,
            #172b3d,
            #142536
        );

    border:
        1px solid
        rgba(255,255,255,0.08);

    color: #d9f1ff;

    font-family:
        "Courier New",
        monospace;

    font-size: 14px;

    line-height: 1.75;

    box-shadow:

        inset 0 1px 0
        rgba(255,255,255,0.04);
}


/* =========================================================
   FOOTER
   ========================================================= */

.footer {

    margin-top: 38px;

    text-align: center;

    color: #7895aa;

    font-size: 13px;
}


.footer-heart {

    color: #60a5fa;

    font-size: 17px;
}


/* =========================================================
   ANIMATIONS
   ========================================================= */

@keyframes logoFloat {

    0% {
        transform:
            translateY(0)
            rotate(-4deg);
    }

    50% {
        transform:
            translateY(-8px)
            rotate(2deg);
    }

    100% {
        transform:
            translateY(0)
            rotate(-4deg);
    }
}


@keyframes floatOne {

    0% {
        transform:
            translateY(0)
            rotate(0deg);
    }

    50% {
        transform:
            translateY(-18px)
            rotate(10deg);
    }

    100% {
        transform:
            translateY(0)
            rotate(0deg);
    }
}


@keyframes floatTwo {

    0% {
        transform:
            translateY(0);
    }

    50% {
        transform:
            translateY(15px);
    }

    100% {
        transform:
            translateY(0);
    }
}


/* =========================================================
   MOBILE
   ========================================================= */

@media (max-width: 650px) {

    .container {

        width: 94%;

        padding-top: 35px;
    }


    .logo {

        width: 75px;

        height: 75px;

        font-size: 37px;

        border-radius: 24px;
    }


    h1 {

        font-size: 42px;

        letter-spacing: -1.5px;
    }


    .subtitle {

        font-size: 15px;
    }


    .tagline {

        font-size: 12px;
    }


    .card {

        padding: 20px;

        border-radius: 23px;
    }


    h2 {

        font-size: 19px;
    }


    textarea {

        min-height: 150px;

        font-size: 15px;
    }


    pre {

        font-size: 12px;
    }


    .decor {

        display: none;
    }
}

</style>

</head>


<body>


<!-- ========================================================
     FLOATING DECORATIONS
     ======================================================== -->

<div class="decor one">
    ♡ ✦
</div>

<div class="decor two">
    ✧ ♡
</div>

<div class="decor three">
    ♡ ✧
</div>


<div class="container">


<!-- ========================================================
     HEADER
     ======================================================== -->

<div class="hero">

    <div class="logo">
        🤖
    </div>

    <h1>

        <span class="brand-script">
            AI
        </span>

        Coding Agent

    </h1>


    <p class="subtitle">

        Generate • Test • Execute Python Code with AI

    </p>


    <div class="tagline">

        ✨ Your little AI coding assistant ✨

    </div>

</div>


<!-- ========================================================
     TASK
     ======================================================== -->

<div class="card">

    <div class="card-header">

        <div class="card-icon">
            📝
        </div>

        <h2>
            Enter Coding Task
        </h2>

    </div>


    <textarea
        id="task"
        placeholder="Tell your AI coding assistant what you want to build... 💙

Example:
Write a Python program that calculates the factorial of 5."
    ></textarea>


    <button
        id="runButton"
        onclick="runAgent()"
    >

        🚀 Run AI Agent

    </button>


    <div
        id="status"
        class="status"
    ></div>

</div>


<!-- ========================================================
     CODE
     ======================================================== -->

<div class="card">

    <div class="card-header">

        <div class="card-icon">
            💻
        </div>

        <h2>
            Generated Code
        </h2>

    </div>


    <div class="output-wrapper">

        <div class="output-label">
            PYTHON
        </div>

        <pre id="code">Your generated code will appear here. ✨</pre>

    </div>

</div>


<!-- ========================================================
     REPORT
     ======================================================== -->

<div class="card">

    <div class="card-header">

        <div class="card-icon">
            🧪
        </div>

        <h2>
            Test & Execution Report
        </h2>

    </div>


    <div class="output-wrapper">

        <div class="output-label">
            REPORT
        </div>

        <pre id="report">Your test report will appear here. 🩵</pre>

    </div>

</div>


<!-- ========================================================
     FOOTER
     ======================================================== -->

<div class="footer">

    Made with

    <span class="footer-heart">
        ♥
    </span>

    using Gemini + LangGraph

</div>


</div>


<!-- ========================================================
     JAVASCRIPT
     ======================================================== -->

<script>

async function runAgent() {

    const task =
        document
        .getElementById("task")
        .value
        .trim();


    const status =
        document
        .getElementById("status");


    const code =
        document
        .getElementById("code");


    const report =
        document
        .getElementById("report");


    const button =
        document
        .getElementById("runButton");


    if (!task) {

        status.innerText =
            "🩵 Please enter a coding task first.";

        status.className =
            "status error";

        return;
    }


    button.disabled = true;

    button.innerText =
        "💙 AI is thinking...";


    status.innerText =
        "✨ Generating code and running tests...";

    status.className =
        "status";


    code.innerText =
        "✨ Your AI is writing the code...";


    report.innerText =
        "🧪 Preparing tests...";


    try {

        const response =
            await fetch(
                "/run",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        task: task
                    })
                }
            );


        let data;


        try {

            data =
                await response.json();

        } catch {

            throw new Error(
                "Server returned an invalid response."
            );
        }


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Server error"
            );
        }


        code.innerText =
            data.code ||
            "No code generated.";


        report.innerText =
            data.report ||
            "No report generated.";


        status.innerText =
            "💙 Done! Your AI agent completed the task.";

        status.className =
            "status success";


    } catch (error) {

        status.innerText =
            "💔 " + error.message;

        status.className =
            "status error";


        code.innerText =
            "No code generated.";


        report.innerText =
            "Something went wrong.";

    } finally {

        button.disabled = false;

        button.innerText =
            "🚀 Run AI Agent";
    }

}

</script>


</body>

</html>
"""


# ============================================================
# 12. HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "gemini_configured": bool(api_key),
        "model": MODEL_NAME,
    }


# ============================================================
# 13. RUN AI AGENT
# ============================================================

@app.post("/run")
def run_agent(request: TaskRequest):

    task = request.task.strip()


    if not task:

        raise HTTPException(
            status_code=400,
            detail="Task cannot be empty.",
        )


    if not api_key:

        raise HTTPException(
            status_code=500,
            detail=(
                "GEMINI_API_KEY is not configured "
                "on the Render server. "
                "Add GEMINI_API_KEY under Render "
                "Environment Variables and redeploy."
            ),
        )


    if llm is None:

        raise HTTPException(
            status_code=500,
            detail=(
                "Gemini could not be initialized. "
                "Check GEMINI_API_KEY, GEMINI_MODEL, "
                "and installed package versions."
            ),
        )


    try:

        initial_state: CrewState = {

            "messages": [
                HumanMessage(content=task)
            ],

            "code": None,

            "report": None,
        }


        result = rt_app.invoke(

            initial_state,

            config={
                "recursion_limit": 20
            },
        )


        return {

            "success": True,

            "task": task,

            "code": result.get(
                "code",
                ""
            ),

            "report": result.get(
                "report",
                ""
            ),
        }


    except Exception as exc:

        print(
            traceback.format_exc()
        )


        raise HTTPException(

            status_code=500,

            detail=(
                f"AI Agent Error: {str(exc)}"
            ),
        )


# ============================================================
# 14. LOCAL RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn


    port = int(
        os.getenv(
            "PORT",
            "8000"
        )
    )


    uvicorn.run(

        app,

        host="0.0.0.0",

        port=port,

        reload=False,
    )
