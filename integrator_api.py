
import os
import sys
from pathlib import Path

# Add feature directories to sys.path to resolve internal imports
ROOT_DIR = Path(__file__).parent.absolute()
for folder in ["feature_8", "feature_9", "feature_10"]:
    path = str(ROOT_DIR / folder)
    if path not in sys.path:
        sys.path.append(path) # Use append to avoid shadowing root modules

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# LangGraph
from unified_graph import build_logisense_graph

# Feature Routers
from feature_8.api.routes import router as f8_router
from feature_9.api import router as f9_router
from feature_10.api.server import app as f10_app

app = FastAPI(
    title="LogiSense AI Unified Integration Hub",
    description="Unified API gateway for Features 8, 9, and 10",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
# F8: routes.py has prefix "/api/explainability", we'll mount it under /api/f8
app.include_router(f8_router, prefix="/api/f8") 
# F9: api.py has no prefix in APIRouter(), we'll wrap it
app.include_router(f9_router, prefix="/api/f9")
# F10: Already a full app, mount it
app.mount("/api/f10", f10_app)

@app.get("/")
async def root():
    return {
        "message": "LogiSense AI Unified API is active.",
        "endpoints": {
            "F8_Explainability": "/api/f8/api/explainability/all",
            "F9_Blockchain": "/api/f9/status",
            "F10_Learner": "/api/f10/health",
            "Unified_Graph": "/invoke"
        }
    }

@app.get("/health")
async def unified_health():
    """Consolidated health check for all integrated features."""
    return {
        "status": "ok",
        "gateway": "operational",
        "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
        "features": ["F8", "F9", "F10"]
    }

@app.post("/invoke")
async def invoke_graph(input_data: dict):
    """Entry point to trigger the unified LangGraph flow."""
    graph = build_logisense_graph()
    # Initialize state with input data
    state = {
        "model": None, 
        "X_df": None,
        "predictions": input_data.get("predictions", []),
        "new_decision": input_data.get("decision"),
        "pending_decisions": [],
        "blockchain_status": {},
        "tamper_alerts": [],
        "messages": [],
        "current_node": ""
    }
    result = await graph.ainvoke(state)
    return result

if __name__ == "__main__":
    uvicorn.run("integrator_api:app", host="0.0.0.0", port=8000, reload=True)
