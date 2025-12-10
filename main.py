import os
import random
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# --- IMPORTS ---
# Ensure these files exist in your folder
from ai_engine import run_agent
from course_data import COURSE_MODULES
from quiz_data import QUIZ_BANK
import stats 

app = FastAPI()

# Crash protection for static files
if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ==========================
#        PAGE ROUTES
# ==========================

@app.get("/", response_class=HTMLResponse)
async def read_home(request: Request):
    current_stats = stats.get_stats()
    return templates.TemplateResponse("index.html", {
        "request": request, "stats": current_stats
    })

@app.get("/about", response_class=HTMLResponse)
async def read_about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/course", response_class=HTMLResponse)
async def read_course_index(request: Request):
    return templates.TemplateResponse("course_overview.html", {
        "request": request, "modules": COURSE_MODULES
    })

@app.get("/course/{module_id}", response_class=HTMLResponse)
async def read_module(request: Request, module_id: int):
    module = next((m for m in COURSE_MODULES if m["id"] == module_id), None)
    if not module: raise HTTPException(status_code=404, detail="Module not found")
    
    total_modules = len(COURSE_MODULES)
    progress = int((module_id / total_modules) * 100)
    prev_id = module_id - 1 if module_id > 1 else None
    next_id = module_id + 1 if module_id < total_modules else None

    return templates.TemplateResponse("module.html", {
        "request": request, "module": module, "prev_id": prev_id, "next_id": next_id, "progress": progress
    })

@app.get("/quiz", response_class=HTMLResponse)
async def read_quiz(request: Request):
    num_questions = min(15, len(QUIZ_BANK))
    selected_questions = random.sample(QUIZ_BANK, num_questions)
    return templates.TemplateResponse("quiz.html", {
        "request": request, "questions": selected_questions, "total_questions": num_questions
    })

# ==========================
#       AI TOOL ROUTES
# ==========================

@app.get("/tool", response_class=HTMLResponse)
async def read_tool(request: Request):
    return templates.TemplateResponse("tool.html", {"request": request})

# --- THIS IS THE CRITICAL ROUTE ---
@app.post("/generate")
async def generate_email(
    request: Request, 
    target: str = Form(...), 
    service: str = Form(...), 
    source: str = Form(...),
    length: str = Form("Short (Sniper)"), 
    # Checkboxes return a list. If empty, default to 'Direct Value'
    styles: list[str] = Form(["Direct Value"]),
    # The Slider returns a float
    creativity: float = Form(0.7)
):
    # 1. Update Stats
    try: stats.increment_usage()
    except: pass

    # 2. Process Styles
    style_string = ", ".join(styles)
    
    # 3. Run Agent
    result_data = run_agent(target, service, source, length, style_string, creativity)
    
    return templates.TemplateResponse("tool.html", {
        "request": request, 
        "result": result_data, 
        "target": target, 
        "service": service, 
        "source": source, 
        "length": length,
        "selected_styles": styles,
        "creativity": creativity
    })

@app.post("/like")
async def like_tool():
    try:
        new_count = stats.increment_likes()
        return JSONResponse(content={"likes": new_count})
    except:
        return JSONResponse(content={"likes": 0})