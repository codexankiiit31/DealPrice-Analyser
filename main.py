"""
FastAPI main application - Direct replacement for Gradio interface
"""
## uvicorn api.main:app --reload --port 8000
import logging
import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import deals, websocket_handler
from src.agents.deal_agent_framework import DealAgentFramework


# Global app state
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize app state on startup"""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
    
    # Initialize the deal framework
    app_state["deal_framework"] = DealAgentFramework()
    app_state["deal_framework"].init_agents_as_needed()
    
    yield
    
    # Cleanup if needed
    app_state.clear()


# Create FastAPI app
app = FastAPI(
    title="DealSense AI API",
    description="Autonomous AI Agents Finding the Best Deals Online",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(deals.router, prefix="/api", tags=["deals"])
app.include_router(websocket_handler.router, prefix="/ws", tags=["websocket"])

# Serve static files (React build)
if os.path.exists("frontend/build"):
    app.mount("/static", StaticFiles(directory="frontend/build/static"), name="static")
    app.mount("/", StaticFiles(directory="frontend/build", html=True), name="frontend")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "DealSense AI Backend is running"}


@app.get("/health")
async def health_check():
    """Health check for the application"""
    return {
        "status": "healthy",
        "framework_initialized": "deal_framework" in app_state,
        "memory_count": len(app_state.get("deal_framework", {}).get("memory", []))
    }


def get_deal_framework() -> DealAgentFramework:
    """Get the initialized deal framework"""
    if "deal_framework" not in app_state:
        raise HTTPException(status_code=500, detail="Deal framework not initialized")
    return app_state["deal_framework"]


# Make framework accessible to other modules
app.state.get_deal_framework = get_deal_framework