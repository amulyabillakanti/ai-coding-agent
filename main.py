````python
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
    title="Lavender AI Coding Agent",
    description="AI-powered coding, testing and execution pipeline",
    version="3.0.0",
)


# ============================================================
# 2. GEMINI CONFIGURATION
# ============================================================

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Gemini 3.6 Flash is the current stable model recommended
# by Google for coding and agentic workflows.
DEFAULT_MODEL = "gemini-3.6-flash"

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    DEFAULT_MODEL
).strip()

# Automatically migrate the old model if it is still present
# in Render Environment Variables.
OLD_MODELS = {
    "gemini-2.5-flash",
    "gemini-2.5-flash-preview-09-2025",
    "gemini-2.5-flash-preview-05-20",
}

if MODEL_NAME in OLD_MODELS:
    print(
        f"[Gemini] Replacing deprecated model "
        f"{MODEL_NAME} with {DEFAULT_MODEL}"
    )
    MODEL_NAME = DEFAULT_MODEL


llm = None

if api_key:
    try:
        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=api_key,
        )

        print(
            f"[Gemini] Initialized successfully with model: "
            f"{MODEL_NAME}"
        )

    except Exception as exc:
        print(
            f"[Gemini] Initialization error: {exc}"
        )
        llm = None

else:
    print(
        "[Gemini] WARNING: GEMINI_API_KEY is not configured."
    )


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
    content = getattr(
        response,
        "content",
        response
    )

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(
                        str(item["text"])
                    )
                else:
                    parts.append(
                        str(item)
                    )
            else:
                parts.append(
                    str(item)
                )

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
    """
    WARNING:
    This uses exec() and is NOT a secure sandbox.

    Do not expose arbitrary code execution publicly without
    adding proper isolation/containerization.
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

    return "Success — no terminal output."


# ============================================================
# 7. GENERATE TEST CASES
# ============================================================

def generate_test_cases(
    task_description: str
) -> str:

    if llm is None:
        return (
            "Test generation unavailable because "
            "Gemini is not configured."
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

def real_time_developer(
    state: CrewState
):

    print(
        "[Developer] Generating Python code..."
    )

    if llm is None:
        raise ValueError(
            "Gemini API is not configured. "
            "Set GEMINI_API_KEY in Render Environment Variables."
        )

    messages = state.get(
        "messages",
        []
    )

    if not messages:
        raise ValueError(
            "No coding task was provided."
        )

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

    response = llm.invoke(
        developer_prompt
    )

    code_str = clean_python_code(
        extract_content(response)
    )

    if not code_str:
        raise ValueError(
            "Gemini returned empty Python code."
        )

    print(
        "\n[Developer] Generated Code:"
    )
    print(code_str)

    return {
        "code": code_str
    }


# ============================================================
# 9. TESTER NODE
# ============================================================

def real_time_tester(
    state: CrewState
):

    print(
        "[Tester] Generating tests "
        "and executing code..."
    )

    messages = state.get(
        "messages",
        []
    )

    if not messages:
        raise ValueError(
            "No task available for testing."
        )

    task = messages[-1].content

    generated_code = state.get(
        "code",
        ""
    )

    if not generated_code:
        raise ValueError(
            "No generated code available for testing."
        )

    test_cases = generate_test_cases(
        task
    )

    execution_result = run_python_code(
        generated_code
    )

    report = (
        "EXECUTION OUTPUT\n"
        "\n"
        f"{execution_result}\n"
        "\n"
        "TEST SCENARIOS\n"
        "\n"
        f"{test_cases}"
    )

    print(
        "\n[Tester] Test Report:"
    )
    print(report)

    return {
        "report": report
    }


# ============================================================
# 10. LANGGRAPH WORKFLOW
# ============================================================

workflow = StateGraph(
    CrewState
)

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
# 11. LAVENDER FRONTEND
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
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

<title>Lavender AI Coding Agent ✦</title>


<!-- ======================================================
     AESTHETIC GOOGLE FONTS
     ====================================================== -->

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
    href="https://fonts.googleapis.com/css2?family=
    Cormorant+Garamond:wght@500;600;700&
    family=Dancing+Script:wght@500;600;700&
    family=DM+Sans:wght@400;500;600;700&
    display=swap"
    rel="stylesheet"
>


<style>

/* =========================================================
   ROOT
   ========================================================= */

:root {

    --lavender-50: #fbf9ff;
    --lavender-100: #f6f0ff;
    --lavender-200: #eee4ff;
    --lavender-300: #dfceff;
    --lavender-400: #c8aaff;
    --lavender-500: #a979ff;
    --lavender-600: #8f5bea;
    --lavender-700: #7544c7;

    --ink: #40334f;
    --muted: #81738f;

    --white: rgba(255, 255, 255, 0.78);

    --shadow:
        0 25px 70px
        rgba(112, 76, 154, 0.12);

    --soft-shadow:
        0 12px 35px
        rgba(112, 76, 154, 0.10);
}


/* =========================================================
   RESET
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

    font-family:
        "DM Sans",
        sans-serif;

    color: var(--ink);

    background:

        radial-gradient(
            circle at 10% 0%,
            rgba(217, 194, 255, 0.65),
            transparent 28%
        ),

        radial-gradient(
            circle at 92% 8%,
            rgba(226, 209, 255, 0.75),
            transparent 28%
        ),

        radial-gradient(
            circle at 50% 100%,
            rgba(235, 221, 255, 0.9),
            transparent 35%
        ),

        linear-gradient(
            135deg,
            #fcf9ff 0%,
            #f5edff 48%,
            #eee4ff 100%
        );

    background-attachment: fixed;

    overflow-x: hidden;
}


/* =========================================================
   BACKGROUND BLOBS
   ========================================================= */

.blob {

    position: fixed;

    border-radius: 50%;

    pointer-events: none;

    filter: blur(2px);

    opacity: 0.45;

    z-index: 0;
}

.blob-one {

    width: 230px;
    height: 230px;

    top: 8%;
    left: -80px;

    background:
        radial-gradient(
            circle,
            rgba(196, 167, 255, 0.45),
            transparent 70%
        );
}

.blob-two {

    width: 280px;
    height: 280px;

    right: -110px;
    top: 42%;

    background:
        radial-gradient(
            circle,
            rgba(213, 191, 255, 0.48),
            transparent 70%
        );
}

.blob-three {

    width: 260px;
    height: 260px;

    bottom: -120px;
    left: 35%;

    background:
        radial-gradient(
            circle,
            rgba(229, 208, 255, 0.6),
            transparent 70%
        );
}


/* =========================================================
   DECORATIVE STARS
   ========================================================= */

.decor {

    position: fixed;

    pointer-events: none;

    user-select: none;

    color:
        rgba(139, 92, 246, 0.25);

    z-index: 0;

    animation:
        gentleFloat
        6s
        ease-in-out
        infinite;
}

.decor-one {

    top: 11%;
    left: 5%;

    font-size: 34px;
}

.decor-two {

    top: 34%;
    right: 5%;

    font-size: 30px;

    animation-delay: 1.5s;
}

.decor-three {

    bottom: 10%;
    left: 7%;

    font-size: 28px;

    animation-delay: 3s;
}


/* =========================================================
   MAIN CONTAINER
   ========================================================= */

.container {

    position: relative;

    z-index: 1;

    width:
        min(
            1080px,
            92%
        );

    margin: 0 auto;

    padding:
        55px 0 70px;
}


/* =========================================================
   HERO
   ========================================================= */

.hero {

    text-align: center;

    margin-bottom: 42px;
}

.logo {

    width: 94px;
    height: 94px;

    margin:
        0 auto 20px;

    display: flex;

    align-items: center;
    justify-content: center;

    border-radius: 32px;

    background:

        linear-gradient(
            145deg,
            #c7a7ff,
            #a979ff
        );

    border:
        1px solid
        rgba(255,255,255,0.85);

    box-shadow:

        0 20px 45px
        rgba(125, 79, 211, 0.20),

        inset 0 1px 0
        rgba(255,255,255,0.75);

    font-size: 46px;

    transform: rotate(-3deg);

    animation:
        logoFloat
        4s
        ease-in-out
        infinite;
}

.logo:hover {

    transform:
        rotate(3deg)
        scale(1.05);
}


h1 {

    margin: 0;

    color:
        #4d3960;

    font-family:
        "Cormorant Garamond",
        serif;

    font-size:
        clamp(
            52px,
            8vw,
            78px
        );

    line-height: 0.95;

    font-weight: 700;

    letter-spacing: -1px;
}

.brand-script {

    font-family:
        "Dancing Script",
        cursive;

    color:
        var(--lavender-600);

    font-size: 0.78em;

    margin-right: 7px;
}

.subtitle {

    margin:
        18px 0 0;

    color:
        #7d6d8b;

    font-size: 17px;

    font-weight: 500;

    letter-spacing: 0.1px;
}

.tagline {

    display: inline-flex;

    align-items: center;

    gap: 7px;

    margin-top: 20px;

    padding:
        10px 18px;

    border-radius: 999px;

    background:
        rgba(
            255,
            255,
            255,
            0.68
        );

    border:
        1px solid
        rgba(
            255,
            255,
            255,
            0.95
        );

    color:
        #80639f;

    font-size: 13px;

    box-shadow:
        var(--soft-shadow);

    backdrop-filter:
        blur(15px);
}


/* =========================================================
   GLASS CARDS
   ========================================================= */

.card {

    margin-bottom: 25px;

    padding: 30px;

    border-radius: 30px;

    background:
        rgba(
            255,
            255,
            255,
            0.68
        );

    border:
        1px solid
        rgba(
            255,
            255,
            255,
            0.95
        );

    box-shadow:
        var(--shadow);

    backdrop-filter:
        blur(22px);

    -webkit-backdrop-filter:
        blur(22px);

    transition:
        transform 0.3s ease,
        box-shadow 0.3s ease;
}

.card:hover {

    transform:
        translateY(-3px);

    box-shadow:
        0 30px 80px
        rgba(
            112,
            76,
            154,
            0.16
        );
}


/* =========================================================
   CARD HEADER
   ========================================================= */

.card-header {

    display: flex;

    align-items: center;

    gap: 14px;

    margin-bottom: 19px;
}

.card-icon {

    width: 48px;
    height: 48px;

    display: flex;

    align-items: center;
    justify-content: center;

    flex-shrink: 0;

    border-radius: 16px;

    background:

        linear-gradient(
            145deg,
            #eee3ff,
            #f8f3ff
        );

    border:
        1px solid
        rgba(
            176,
            141,
            233,
            0.18
        );

    box-shadow:
        0 8px 22px
        rgba(
            113,
            73,
            156,
            0.09
        );

    font-size: 22px;
}

h2 {

    margin: 0;

    color:
        #57416c;

    font-family:
        "Cormorant Garamond",
        serif;

    font-size: 28px;

    font-weight: 700;
}


/* =========================================================
   TEXTAREA
   ========================================================= */

textarea {

    width: 100%;

    min-height: 185px;

    padding: 21px;

    border-radius: 22px;

    border:
        2px solid
        #e1d3f4;

    outline: none;

    resize: vertical;

    background:
        rgba(
            255,
            252,
            255,
            0.85
        );

    color:
        #44364f;

    font-family:
        "DM Sans",
        sans-serif;

    font-size: 15.5px;

    line-height: 1.7;

    transition:
        border-color 0.25s ease,
        box-shadow 0.25s ease,
        background 0.25s ease;
}

textarea::placeholder {

    color:
        #ad9dbb;
}

textarea:focus {

    border-color:
        #a979ff;

    background:
        rgba(
            255,
            255,
            255,
            0.96
        );

    box-shadow:

        0 0 0 5px
        rgba(
            167,
            122,
            255,
            0.11
        );
}


/* =========================================================
   BUTTON
   ========================================================= */

button {

    width: 100%;

    margin-top: 17px;

    padding: 17px 25px;

    border: none;

    border-radius: 19px;

    background:

        linear-gradient(
            135deg,
            #9870ed,
            #b179ef
        );

    color: white;

    font-family:
        "DM Sans",
        sans-serif;

    font-size: 16px;

    font-weight: 700;

    letter-spacing: 0.1px;

    cursor: pointer;

    box-shadow:

        0 15px 32px
        rgba(
            128,
            81,
            211,
            0.23
        );

    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease,
        filter 0.2s ease;
}

button:hover {

    transform:
        translateY(-2px);

    filter:
        brightness(1.04);

    box-shadow:

        0 20px 40px
        rgba(
            128,
            81,
            211,
            0.30
        );
}

button:active {

    transform:
        translateY(1px);
}

button:disabled {

    background:
        linear-gradient(
            135deg,
            #c7b8d4,
            #cfc2da
        );

    cursor:
        not-allowed;

    box-shadow: none;

    transform: none;
}


/* =========================================================
   STATUS
   ========================================================= */

.status {

    min-height: 25px;

    margin-top: 16px;

    text-align: center;

    font-size: 13px;

    font-weight: 700;
}

.success {

    color:
        #398467;
}

.error {

    color:
        #bd537a;
}


/* =========================================================
   OUTPUT
   ========================================================= */

.output-wrapper {

    position: relative;
}

.output-label {

    position: absolute;

    top: 12px;
    right: 13px;

    z-index: 2;

    padding:
        5px 10px;

    border-radius: 999px;

    background:
        rgba(
            255,
            255,
            255,
            0.09
        );

    border:
        1px solid
        rgba(
            255,
            255,
            255,
            0.10
        );

    color:
        #cdbce3;

    font-size: 9px;

    font-weight: 700;

    letter-spacing: 1.2px;
}

pre {

    margin: 0;

    min-height: 155px;

    padding:
        26px 21px;

    border-radius: 21px;

    overflow-x: auto;

    white-space: pre-wrap;

    word-break: break-word;

    background:

        linear-gradient(
            145deg,
            #2c2039,
            #21182d
        );

    border:
        1px solid
        rgba(
            255,
            255,
            255,
            0.08
        );

    color:
        #eee4fa;

    font-family:
        "Courier New",
        monospace;

    font-size: 13px;

    line-height: 1.75;

    box-shadow:

        inset 0 1px 0
        rgba(
            255,
            255,
            255,
            0.04
        );
}


/* =========================================================
   FOOTER
   ========================================================= */

.footer {

    margin-top: 36px;

    text-align: center;

    color:
        #8b779a;

    font-size: 12.5px;
}

.footer-heart {

    color:
        #9a68e9;

    font-size: 16px;
}


/* =========================================================
   ANIMATIONS
   ========================================================= */

@keyframes logoFloat {

    0% {
        transform:
            translateY(0)
            rotate(-3deg);
    }

    50% {
        transform:
            translateY(-8px)
            rotate(2deg);
    }

    100% {
        transform:
            translateY(0)
            rotate(-3deg);
    }
}

@keyframes gentleFloat {

    0% {
        transform:
            translateY(0)
            rotate(0deg);
    }

    50% {
        transform:
            translateY(-15px)
            rotate(8deg);
    }

    100% {
        transform:
            translateY(0)
            rotate(0deg);
    }
}


/* =========================================================
   MOBILE
   ========================================================= */

@media (max-width: 650px) {

    .container {

        width: 94%;

        padding-top: 34px;
    }

    .logo {

        width: 76px;
        height: 76px;

        font-size: 37px;

        border-radius: 25px;
    }

    h1 {

        font-size: 48px;
    }

    .subtitle {

        font-size: 14px;

        line-height: 1.5;
    }

    .tagline {

        font-size: 11px;
    }

    .card {

        padding: 20px;

        border-radius: 24px;
    }

    h2 {

        font-size: 24px;
    }

    .card-icon {

        width: 43px;
        height: 43px;
    }

    textarea {

        min-height: 155px;

        font-size: 14px;
    }

    pre {

        font-size: 11.5px;
    }

    .decor,
    .blob {

        display: none;
    }
}

</style>

</head>


<body>


<!-- ======================================================
     BACKGROUND
     ====================================================== -->

<div class="blob blob-one"></div>
<div class="blob blob-two"></div>
<div class="blob blob-three"></div>


<div class="decor decor-one">
    ♡ ✦
</div>

<div class="decor decor-two">
    ✧ ♡
</div>

<div class="decor decor-three">
    ♡ ✧
</div>


<!-- ======================================================
     MAIN
     ====================================================== -->

<div class="container">


<!-- ======================================================
     HERO
     ====================================================== -->

<div class="hero">

    <div class="logo">
        🤖
    </div>

    <h1>
        <span class="brand-script">AI</span>
        Coding Agent
    </h1>

    <p class="subtitle">
        Generate · Test · Execute Python Code with AI
    </p>

    <div class="tagline">
        ✨ Your little lavender coding companion ✨
    </div>

</div>


<!-- ======================================================
     TASK CARD
     ====================================================== -->

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
        placeholder="Tell your AI coding assistant what you want to build... 💜

Example:
Write a Python program that calculates
the factorial of 5."
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


<!-- ======================================================
     GENERATED CODE
     ====================================================== -->

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


<!-- ======================================================
     TEST REPORT
     ====================================================== -->

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

        <pre id="report">Your test report will appear here. 🪻</pre>

    </div>

</div>


<!-- ======================================================
     FOOTER
     ====================================================== -->

<div class="footer">

    Made with
    <span class="footer-heart">♥</span>
    using Gemini + LangGraph

</div>


</div>


<!-- ======================================================
     JAVASCRIPT
     ====================================================== -->

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


    if (!task) {

        status.innerText =
            "🪻 Please enter a coding task first.";

        status.className =
            "status error";

        return;
    }


    button.disabled = true;

    button.innerText =
        "💜 AI is thinking...";


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
            "💜 Done! Your AI agent completed the task.";

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
# 13. RUN AI CODING AGENT
# ============================================================

@app.post("/run")
def run_agent(
    request: TaskRequest
):

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
                "and the installed package versions."
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
````
