"""
FastAPI server for IndoBERT Document Customer Service
RESTful API endpoints for the RAG system
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import logging
import os
import sys
import asyncio
from datetime import datetime
import uuid

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_system import RAGSystem
from src.evaluation import DocumentCSEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class ChatRequest(BaseModel):
    query: str = Field(..., description="User query about document requirements")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for tracking")
    include_context: bool = Field(False, description="Include retrieved context in response")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Generated response")
    intent: str = Field(..., description="Detected intent/document type")
    conversation_id: str = Field(..., description="Conversation ID")
    confidence: Optional[float] = Field(None, description="Response confidence score")
    context: Optional[List[str]] = Field(None, description="Retrieved context (if requested)")
    timestamp: datetime = Field(default_factory=datetime.now)

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    knowledge_base_ready: bool
    version: str
    timestamp: datetime = Field(default_factory=datetime.now)

class EvaluationRequest(BaseModel):
    test_file: str = Field("data/raw/conversations_test.json", description="Path to test file")
    save_results: bool = Field(True, description="Save evaluation results")

class EvaluationResponse(BaseModel):
    status: str
    overall_score: float
    intent_accuracy: float
    bleu_score: float
    evaluation_id: str
    timestamp: datetime = Field(default_factory=datetime.now)

# Global variables
rag_system = None
app = FastAPI(
    title="IndoBERT Document Customer Service API",
    description="API for Indonesian document requirements assistance using IndoBERT and RAG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Conversation storage (in production, use proper database)
conversation_storage = {}

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup"""
    global rag_system
    
    try:
        logger.info("Starting IndoBERT Document CS API...")
        
        # Initialize RAG system
        config_path = "config/model_config.yaml"
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        rag_system = RAGSystem(config_path)
        
        # Initialize knowledge base
        logger.info("Initializing knowledge base...")
        rag_system.initialize_knowledge_base()
        
        logger.info("✅ IndoBERT Document CS API started successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise e

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with basic information"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>IndoBERT Document Customer Service API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; text-align: center; }
            .feature { background: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 5px; }
            .endpoint { background: #e3f2fd; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { background: #4caf50; color: white; padding: 4px 8px; border-radius: 3px; font-size: 12px; }
            a { color: #3498db; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏛️ IndoBERT Document Customer Service API</h1>
            
            <div class="feature">
                <h3>📋 Supported Documents</h3>
                <ul>
                    <li><strong>KTP</strong> - Kartu Tanda Penduduk</li>
                    <li><strong>SIM</strong> - Surat Izin Mengemudi</li>
                    <li><strong>Passport</strong> - Paspor Indonesia</li>
                    <li><strong>Akta Kelahiran</strong> - Birth Certificate</li>
                </ul>
            </div>
            
            <div class="feature">
                <h3>🔌 API Endpoints</h3>
                
                <div class="endpoint">
                    <span class="method">GET</span> <strong>/health</strong> - API health check
                </div>
                
                <div class="endpoint">
                    <span class="method">POST</span> <strong>/chat</strong> - Chat with the document assistant
                </div>
                
                <div class="endpoint">
                    <span class="method">GET</span> <strong>/intents</strong> - Get supported intents
                </div>
                
                <div class="endpoint">
                    <span class="method">POST</span> <strong>/evaluate</strong> - Run model evaluation
                </div>
            </div>
            
            <div class="feature">
                <h3>📚 Documentation</h3>
                <p>
                    <a href="/docs">🔗 Interactive API Documentation (Swagger UI)</a><br>
                    <a href="/redoc">🔗 Alternative Documentation (ReDoc)</a>
                </p>
            </div>
            
            <div class="feature">
                <h3>💡 Example Usage</h3>
                <pre style="background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto;">
curl -X POST "http://localhost:8000/chat" \\
     -H "Content-Type: application/json" \\
     -d '{"query": "Syarat bikin KTP baru apa aja ya?"}'</pre>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        model_loaded = rag_system is not None and hasattr(rag_system, 'model')
        kb_ready = rag_system is not None and hasattr(rag_system, 'vector_store')
        
        status = "healthy" if model_loaded and kb_ready else "degraded"
        
        return HealthResponse(
            status=status,
            model_loaded=model_loaded,
            knowledge_base_ready=kb_ready,
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Process the query
        result = rag_system.chat(request.query)
        
        # Store conversation (in production, use proper database)
        if conversation_id not in conversation_storage:
            conversation_storage[conversation_id] = []
        
        conversation_storage[conversation_id].append({
            "timestamp": datetime.now(),
            "user_query": request.query,
            "assistant_response": result['response'],
            "intent": result['intent']
        })
        
        # Prepare response
        response = ChatResponse(
            response=result['response'],
            intent=result['intent'],
            conversation_id=conversation_id
        )
        
        # Include context if requested
        if request.include_context:
            response.context = [result.get('context', '')]
        
        return response
        
    except Exception as e:
        logger.error(f"Chat processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history"""
    if conversation_id not in conversation_storage:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "conversation_id": conversation_id,
        "messages": conversation_storage[conversation_id]
    }

@app.get("/intents")
async def get_supported_intents():
    """Get list of supported intents"""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    return {
        "intents": [
            {
                "intent": "ktp",
                "description": "Kartu Tanda Penduduk (KTP) related queries",
                "examples": [
                    "Syarat bikin KTP baru apa aja?",
                    "KTP hilang gimana cara gantinya?",
                    "Mau pindah alamat, KTP perlu diupdate ga?"
                ]
            },
            {
                "intent": "sim", 
                "description": "Surat Izin Mengemudi (SIM) related queries",
                "examples": [
                    "Cara perpanjang SIM yang expired?",
                    "Mau bikin SIM A baru, syaratnya apa?",
                    "SIM internasional untuk ke luar negeri"
                ]
            },
            {
                "intent": "passport",
                "description": "Passport related queries", 
                "examples": [
                    "Bikin passport baru persyaratannya apa?",
                    "Anak kecil perlu passport ga?",
                    "Passport hilang di luar negeri gimana?"
                ]
            },
            {
                "intent": "akta_kelahiran",
                "description": "Birth certificate related queries",
                "examples": [
                    "Cara bikin akta kelahiran untuk bayi?",
                    "Akta kelahiran hilang bisa diganti?",
                    "Belum menikah tapi punya anak, akta bisa diurus?"
                ]
            }
        ]
    }

@app.post("/evaluate", response_model=EvaluationResponse)
async def run_evaluation(request: EvaluationRequest, background_tasks: BackgroundTasks):
    """Run model evaluation"""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Check if test file exists
        if not os.path.exists(request.test_file):
            raise HTTPException(status_code=404, detail=f"Test file not found: {request.test_file}")
        
        evaluation_id = str(uuid.uuid4())
        
        # Run evaluation in background
        background_tasks.add_task(
            run_evaluation_task, 
            request.test_file, 
            evaluation_id, 
            request.save_results
        )
        
        return EvaluationResponse(
            status="started",
            overall_score=0.0,  # Will be updated when evaluation completes
            intent_accuracy=0.0,
            bleu_score=0.0,
            evaluation_id=evaluation_id
        )
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

async def run_evaluation_task(test_file: str, evaluation_id: str, save_results: bool):
    """Background task for running evaluation"""
    try:
        logger.info(f"Starting evaluation {evaluation_id}")
        
        # Initialize evaluator
        evaluator = DocumentCSEvaluator("config/model_config.yaml")
        
        # Run evaluation
        results = evaluator.run_comprehensive_evaluation(test_file)
        
        # Save results if requested
        if save_results:
            output_file = f"logs/evaluation_{evaluation_id}.json"
            evaluator.save_results(results, output_file)
        
        logger.info(f"Evaluation {evaluation_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Evaluation {evaluation_id} failed: {e}")

@app.get("/evaluation/{evaluation_id}")
async def get_evaluation_status(evaluation_id: str):
    """Get evaluation status and results"""
    result_file = f"logs/evaluation_{evaluation_id}.json"
    
    if os.path.exists(result_file):
        import json
        with open(result_file, 'r') as f:
            results = json.load(f)
        
        return {
            "evaluation_id": evaluation_id,
            "status": "completed",
            "results": results['summary']
        }
    else:
        return {
            "evaluation_id": evaluation_id,
            "status": "running",
            "message": "Evaluation still in progress"
        }

@app.get("/stats")
async def get_stats():
    """Get API usage statistics"""
    total_conversations = len(conversation_storage)
    total_messages = sum(len(conv) for conv in conversation_storage.values())
    
    return {
        "total_conversations": total_conversations,
        "total_messages": total_messages,
        "active_conversations": len([
            conv_id for conv_id, messages in conversation_storage.items()
            if messages and (datetime.now() - messages[-1]['timestamp']).seconds < 3600
        ]),
        "model_status": "loaded" if rag_system is not None else "not_loaded"
    }

@app.delete("/conversations")
async def clear_conversations():
    """Clear all conversation history"""
    global conversation_storage
    count = len(conversation_storage)
    conversation_storage.clear()
    
    return {
        "message": f"Cleared {count} conversations",
        "timestamp": datetime.now()
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "detail": str(exc)}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "detail": str(exc)}

# Main function for running the server
if __name__ == "__main__":
    import uvicorn
    
    # Configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    logger.info(f"Starting server on {HOST}:{PORT}")
    
    uvicorn.run(
        "src.api_server:app",
        host=HOST,
        port=PORT,
        reload=True,
        log_level="info"
    )
