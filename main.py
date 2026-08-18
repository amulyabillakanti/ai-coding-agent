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
    "gemini-3.6-flash",
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
    """Safely convert a LangChain Gemini response to plain text."""

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
    """Remove accidental Markdown code fences."""

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
    """
    Execute generated Python code.

    WARNING:
    This uses exec() inside the application process.
    Do NOT use this approach for untrusted users in production.
    """

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
            {"__name__": "__main__"},
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
    """Ask Gemini to generate QA test scenarios."""

    if llm is None:
        return (
            "Test generation unavailable because "
            "Gemini is not configured."
        )

    prompt = f"""
You are a Senior QA Engineer.

Generate 3 to 5 highly specific test scenarios
for the following coding task:

{task_description}

Include:

1. Normal cases
2. Boundary cases
3. Edge cases
4. Invalid input cases where appropriate

Return only a numbered list.
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
            "Set GEMINI_API_KEY in the Render Environment variables."
        )

    messages = state.get("messages", [])

    if not messages:
        raise ValueError("No coding task was provided.")

    task = messages[-1].content

    developer_prompt = f"""
You are an expert Python developer.

Write a complete, executable Python program
for the following task:

{task}

Requirements:

- Return ONLY Python source code.
- Do NOT use Markdown.
- Do NOT include ```python.
- The program must be executable directly with Python.
- Use clear variable names.
- Handle reasonable edge cases.
- If the task asks for output, make sure the program prints the result.
- Do not explain the code outside the Python source.
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

    task = state["messages"][-1].content

    generated_code = state.get("code", "")

    if not generated_code:
        raise ValueError(
            "No generated code available for testing."
        )

    # Generate QA scenarios
    test_cases = generate_test_cases(task)

    # Execute generated Python
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
# 11. FRONTEND
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

    <title>AI Coding Agent</title>

    <style>

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            min-height: 100vh;

            font-family:
                "Inter",
                "Segoe UI",
                Arial,
                sans-serif;

            color: #ffffff;

            background:
                radial-gradient(
                    circle at top left,
                    #8b5cf6 0%,
                    transparent 35%
                ),
                radial-gradient(
                    circle at bottom right,
                    #c026d3 0%,
                    transparent 35%
                ),
                linear-gradient(
                    135deg,
                    #1e1033,
                    #321052,
                    #4c1d68,
                    #24103d
                );

            background-attachment: fixed;
        }


        /* ====================================================
           MAIN CONTAINER
        ==================================================== */

        .container {
            width: 100%;
            max-width: 1100px;

            margin: auto;

            padding: 50px 20px 70px;
        }


        /* ====================================================
           HEADER
        ==================================================== */

        .hero {
            text-align: center;

            margin-bottom: 40px;
        }

        .hero-icon {
            width: 80px;
            height: 80px;

            margin: 0 auto 20px;

            display: flex;
            align-items: center;
            justify-content: center;

            border-radius: 24px;

            font-size: 38px;

            background:
                linear-gradient(
                    135deg,
                    #a855f7,
                    #7c3aed
                );

            box-shadow:
                0 15px 45px
                rgba(168, 85, 247, 0.45);
        }

        h1 {
            margin: 0;

            font-size: 46px;

            font-weight: 800;

            letter-spacing: -1px;

            background:
                linear-gradient(
                    90deg,
                    #ffffff,
                    #e9d5ff,
                    #c084fc
                );

            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .subtitle {
            margin-top: 12px;

            color: #ddd6fe;

            font-size: 18px;
        }


        /* ====================================================
           GLASS CARDS
        ==================================================== */

        .card {

            background:
                rgba(255, 255, 255, 0.09);

            border:
                1px solid
                rgba(255, 255, 255, 0.16);

            backdrop-filter:
                blur(18px);

            -webkit-backdrop-filter:
                blur(18px);

            padding: 28px;

            border-radius: 22px;

            margin-bottom: 25px;

            box-shadow:
                0 20px 50px
                rgba(0, 0, 0, 0.25);

            transition:
                transform 0.25s ease,
                box-shadow 0.25s ease;
        }

        .card:hover {

            transform:
                translateY(-3px);

            box-shadow:
                0 25px 60px
                rgba(0, 0, 0, 0.32);
        }


        /* ====================================================
           CARD HEADINGS
        ==================================================== */

        .card h2 {

            margin-top: 0;

            margin-bottom: 18px;

            font-size: 22px;

            color: #f5f3ff;
        }


        /* ====================================================
           TEXTAREA
        ==================================================== */

        textarea {

            width: 100%;

            min-height: 170px;

            padding: 18px;

            resize: vertical;

            border-radius: 14px;

            border:
                1px solid
                rgba(196, 181, 253, 0.3);

            background:
                rgba(15, 7, 30, 0.65);

            color: #ffffff;

            font-size: 16px;

            line-height: 1.6;

            outline: none;

            transition:
                border 0.2s ease,
                box-shadow 0.2s ease;
        }

        textarea::placeholder {
            color: #a78bfa;
        }

        textarea:focus {

            border-color:
                #a855f7;

            box-shadow:
                0 0 0 4px
                rgba(168, 85, 247, 0.15);
        }


        /* ====================================================
           BUTTON
        ==================================================== */

        button {

            width: 100%;

            margin-top: 18px;

            padding: 16px;

            border: none;

            border-radius: 14px;

            background:
                linear-gradient(
                    135deg,
                    #a855f7,
                    #7c3aed,
                    #9333ea
                );

            color: white;

            font-size: 17px;

            font-weight: 700;

            cursor: pointer;

            box-shadow:
                0 10px 30px
                rgba(124, 58, 237, 0.4);

            transition:
                transform 0.2s ease,
                box-shadow 0.2s ease,
                opacity 0.2s ease;
        }

        button:hover {

            transform:
                translateY(-2px);

            box-shadow:
                0 15px 35px
                rgba(168, 85, 247, 0.5);
        }

        button:active {
            transform:
                translateY(0);
        }

        button:disabled {

            opacity: 0.55;

            cursor: not-allowed;

            transform: none;
        }


        /* ====================================================
           STATUS
        ==================================================== */

        .status {

            margin-top: 16px;

            text-align: center;

            font-weight: 600;

            min-height: 24px;
        }

        .success {
            color: #86efac;
        }

        .error {
            color: #fca5a5;
        }


        /* ====================================================
           CODE / REPORT
        ==================================================== */

        pre {

            margin: 0;

            min-height: 130px;

            padding: 20px;

            border-radius: 14px;

            overflow-x: auto;

            white-space: pre-wrap;

            word-wrap: break-word;

            background:
                rgba(8, 4, 18, 0.78);

            border:
                1px solid
                rgba(167, 139, 250, 0.18);

            color: #e9d5ff;

            font-family:
                "Cascadia Code",
                "Fira Code",
                Consolas,
                monospace;

            font-size: 14px;

            line-height: 1.6;
        }


        /* ====================================================
           FOOTER
        ==================================================== */

        .footer {

            text-align: center;

            margin-top: 30px;

            color: #c4b5fd;

            font-size: 14px;
        }


        /* ====================================================
           RESPONSIVE
        ==================================================== */

        @media (max-width: 700px) {

            .container {
                padding:
                    30px 15px 50px;
            }

            h1 {
                font-size: 34px;
            }

            .subtitle {
                font-size: 15px;
            }

            .card {
                padding: 20px;
                border-radius: 18px;
            }

            .hero-icon {
                width: 65px;
                height: 65px;
                font-size: 30px;
            }
        }

    </style>

</head>


<body>

    <div class="container">


        <!-- ==================================================
             HEADER
        =================================================== -->

        <div class="hero">

            <div class="hero-icon">
                🤖
            </div>

            <h1>
                AI Coding Agent
            </h1>

            <p class="subtitle">
                Generate • Test • Execute Python Code using AI
            </p>

        </div>


        <!-- ==================================================
             TASK INPUT
        =================================================== -->

        <div class="card">

            <h2>
                📝 Enter Coding Task
            </h2>

            <textarea
                id="task"
                placeholder="Example: Write a Python program to check whether a number is prime."
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


        <!-- ==================================================
             GENERATED CODE
        =================================================== -->

        <div class="card">

            <h2>
                💻 Generated Code
            </h2>

            <pre id="code">Your generated code will appear here.</pre>

        </div>


        <!-- ==================================================
             TEST REPORT
        =================================================== -->

        <div class="card">

            <h2>
                🧪 Test & Execution Report
            </h2>

            <pre id="report">Your test report will appear here.</pre>

        </div>


        <div class="footer">
            Powered by Gemini + LangGraph + FastAPI
        </div>

    </div>


    <!-- ======================================================
         JAVASCRIPT
    ======================================================= -->

    <script>

        async function runAgent() {

            const task =
                document
                    .getElementById("task")
                    .value
                    .trim();

            const status =
                document.getElementById("status");

            const code =
                document.getElementById("code");

            const report =
                document.getElementById("report");

            const button =
                document.getElementById("runButton");


            /* ----------------------------------------------
               Validate task
            ---------------------------------------------- */

            if (!task) {

                status.innerText =
                    "⚠️ Please enter a coding task.";

                status.className =
                    "status error";

                return;
            }


            /* ----------------------------------------------
               Loading state
            ---------------------------------------------- */

            button.disabled = true;

            button.innerText =
                "⏳ AI Agent Running...";

            status.innerText =
                "⏳ Generating code and running tests...";

            status.className =
                "status";

            code.innerText =
                "Generating code...";

            report.innerText =
                "Running tests...";


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


                /* ------------------------------------------
                   Display result
                ------------------------------------------ */

                code.innerText =
                    data.code ||
                    "No code generated.";

                report.innerText =
                    data.report ||
                    "No report generated.";

                status.innerText =
                    "✅ AI Agent completed successfully.";

                status.className =
                    "status success";

            }

            catch (error) {

                status.innerText =
                    "❌ " + error.message;

                status.className =
                    "status error";

                code.innerText = "";

                report.innerText = "";

            }

            finally {

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
# 13. RUN AI CODING AGENT
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
                "Add GEMINI_API_KEY under "
                "Render → Environment Variables "
                "and redeploy."
            ),
        )


    if llm is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Gemini could not be initialized. "
                "Check GEMINI_API_KEY and the "
                "installed package versions."
            ),
        )


    try:

        initial_state: CrewState = {

            "messages": [
                HumanMessage(
                    content=task
                )
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

            "code":
                result.get(
                    "code",
                    ""
                ),

            "report":
                result.get(
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
