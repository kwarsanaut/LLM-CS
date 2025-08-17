"""
Simple web interface for IndoBERT Document Customer Service
Using FastAPI to serve HTML pages
"""

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import sys
import requests
import json
from typing import Optional

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(title="IndoBERT Document CS Web Interface")

# Setup static files and templates
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# API base URL (can be configured)
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with chat interface"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "IndoBERT Document Customer Service"
    })

@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, query: str = Form(...), conversation_id: Optional[str] = Form(None)):
    """Process chat message"""
    try:
        # Call the API
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={
                "query": query,
                "conversation_id": conversation_id,
                "include_context": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return templates.TemplateResponse("chat_response.html", {
                "request": request,
                "query": query,
                "response": result["response"],
                "intent": result["intent"],
                "conversation_id": result["conversation_id"],
                "success": True
            })
        else:
            error_msg = f"API Error: {response.status_code}"
            return templates.TemplateResponse("chat_response.html", {
                "request": request,
                "query": query,
                "error": error_msg,
                "success": False
            })
    
    except requests.exceptions.RequestException as e:
        return templates.TemplateResponse("chat_response.html", {
            "request": request,
            "query": query,
            "error": f"Connection error: {str(e)}",
            "success": False
        })

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    """About page"""
    return templates.TemplateResponse("about.html", {
        "request": request,
        "title": "About - IndoBERT Document CS"
    })

@app.get("/docs-info", response_class=HTMLResponse)
async def docs_info(request: Request):
    """Information about supported documents"""
    return templates.TemplateResponse("docs_info.html", {
        "request": request,
        "title": "Supported Documents - IndoBERT Document CS"
    })

@app.get("/health-ui", response_class=HTMLResponse)
async def health_ui(request: Request):
    """Health check UI"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
        else:
            health_data = {"status": "error", "message": f"HTTP {response.status_code}"}
    except requests.exceptions.RequestException as e:
        health_data = {"status": "error", "message": str(e)}
    
    return templates.TemplateResponse("health.html", {
        "request": request,
        "title": "System Health - IndoBERT Document CS",
        "health": health_data
    })

if __name__ == "__main__":
    import uvicorn
    
    # Create templates if they don't exist
    create_templates()
    
    uvicorn.run(
        "src.web_interface:app",
        host="0.0.0.0",
        port=8080,
        reload=True
    )

def create_templates():
    """Create HTML templates"""
    
    # Base template
    base_template = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .chat-container { max-height: 500px; overflow-y: auto; }
        .user-message { background-color: #007bff; color: white; }
        .bot-message { background-color: #f8f9fa; color: #333; }
        .intent-badge { font-size: 0.8em; }
        .footer { margin-top: 50px; padding: 20px 0; background-color: #f8f9fa; }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="/">
                <i class="fas fa-building"></i> IndoBERT Document CS
            </a>
            <div class="navbar-nav ms-auto">
                <a class="nav-link" href="/">Home</a>
                <a class="nav-link" href="/docs-info">Documents</a>
                <a class="nav-link" href="/health-ui">Health</a>
                <a class="nav-link" href="/about">About</a>
            </div>
        </div>
    </nav>
    
    <main class="container mt-4">
        {% block content %}{% endblock %}
    </main>
    
    <footer class="footer mt-auto">
        <div class="container text-center">
            <span class="text-muted">
                <i class="fas fa-robot"></i> IndoBERT Document Customer Service - 
                Powered by AI untuk membantu urusan dokumen Indonesia
            </span>
        </div>
    </footer>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    {% block scripts %}{% endblock %}
</body>
</html>
"""
    
    # Index template
    index_template = """
{% extends "base.html" %}

{% block content %}
<div class="row">
    <div class="col-lg-8 mx-auto">
        <div class="text-center mb-5">
            <h1 class="display-4"><i class="fas fa-building"></i> Asisten Dokumen Indonesia</h1>
            <p class="lead">Tanyakan apapun tentang persyaratan dokumen KTP, SIM, Passport, dan Akta Kelahiran</p>
        </div>
        
        <div class="card">
            <div class="card-header bg-primary text-white">
                <h5 class="mb-0"><i class="fas fa-comments"></i> Chat dengan Asisten</h5>
            </div>
            <div class="card-body">
                <div id="chat-messages" class="chat-container mb-3">
                    <div class="alert alert-info">
                        <i class="fas fa-info-circle"></i>
                        Selamat datang! Saya siap membantu Anda dengan informasi persyaratan dokumen. 
                        Contoh: "Syarat bikin KTP baru apa aja ya?"
                    </div>
                </div>
                
                <form method="post" action="/chat" class="d-flex gap-2">
                    <input type="text" name="query" class="form-control" 
                           placeholder="Tanyakan tentang persyaratan dokumen..." required>
                    <input type="hidden" name="conversation_id" id="conversation_id" value="">
                    <button type="submit" class="btn btn-primary">
                        <i class="fas fa-paper-plane"></i> Kirim
                    </button>
                </form>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-md-3">
                <div class="card h-100">
                    <div class="card-body text-center">
                        <i class="fas fa-id-card fa-2x text-primary mb-2"></i>
                        <h6>KTP</h6>
                        <small>Kartu Tanda Penduduk</small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card h-100">
                    <div class="card-body text-center">
                        <i class="fas fa-car fa-2x text-success mb-2"></i>
                        <h6>SIM</h6>
                        <small>Surat Izin Mengemudi</small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card h-100">
                    <div class="card-body text-center">
                        <i class="fas fa-passport fa-2x text-warning mb-2"></i>
                        <h6>Passport</h6>
                        <small>Paspor Indonesia</small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card h-100">
                    <div class="card-body text-center">
                        <i class="fas fa-baby fa-2x text-info mb-2"></i>
                        <h6>Akta Kelahiran</h6>
                        <small>Surat Kelahiran</small>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}
"""
    
    # Chat response template
    chat_response_template = """
<div class="message-pair mb-3">
    <div class="message user-message p-3 rounded mb-2">
        <strong><i class="fas fa-user"></i> Anda:</strong> {{ query }}
    </div>
    {% if success %}
    <div class="message bot-message p-3 rounded">
        <div class="d-flex justify-content-between align-items-start mb-2">
            <strong><i class="fas fa-robot"></i> Asisten:</strong>
            <span class="badge bg-secondary intent-badge">{{ intent }}</span>
        </div>
        <div>{{ response|replace("\\n", "<br>")|safe }}</div>
    </div>
    {% else %}
    <div class="message p-3 rounded bg-danger text-white">
        <strong><i class="fas fa-exclamation-triangle"></i> Error:</strong> {{ error }}
    </div>
    {% endif %}
</div>

<script>
    // Update conversation ID if provided
    {% if conversation_id %}
    document.getElementById('conversation_id').value = '{{ conversation_id }}';
    {% endif %}
    
    // Scroll to bottom
    const chatContainer = document.getElementById('chat-messages');
    chatContainer.scrollTop = chatContainer.scrollHeight;
</script>
"""
    
    # Write templates to files
    templates_dir = "templates"
    os.makedirs(templates_dir, exist_ok=True)
    
    with open(f"{templates_dir}/base.html", "w", encoding="utf-8") as f:
        f.write(base_template)
    
    with open(f"{templates_dir}/index.html", "w", encoding="utf-8") as f:
        f.write(index_template)
    
    with open(f"{templates_dir}/chat_response.html", "w", encoding="utf-8") as f:
        f.write(chat_response_template)
