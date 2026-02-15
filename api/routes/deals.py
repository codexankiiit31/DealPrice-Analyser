"""
Deal routes - Direct replacement for Gradio button clicks and data operations
"""
import asyncio
import logging
import uuid
from typing import List, Dict, Any
from datetime import datetime
import threading

from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse

from api.models.schemas import (
    CategoryRequest, SearchResponse, SearchResultsResponse, 
    AppStatus, ErrorResponse, OpportunityData
)
from src.config.feeds import CATEGORY_FEEDS

router = APIRouter()

# Global state for managing background tasks
active_jobs: Dict[str, Dict[str, Any]] = {}


def table_for(opps) -> List[List[str]]:
    """
    Format opportunities for table display - Direct from Gradio app.py
    """
    return [
        [
            opp.deal.product_description, 
            f"${opp.deal.price:.2f}", 
            f"${opp.estimate:.2f}", 
            f"${opp.discount:.2f}", 
            f'<a href="{opp.deal.url}" target="_blank" style="background-color:#007bff;color:white;padding:8px 16px;text-decoration:none;border-radius:4px;font-weight:bold;">🔗 View Deal</a>'
        ] 
        for opp in opps
    ]


def validate_categories_logic(selected_categories: List[str]) -> str:
    """
    Validate category selection - Direct from Gradio app.py
    """
    if len(selected_categories) == 0:
        return "⚠️ Please select at least one category before running."
    if len(selected_categories) > 3:
        return "⚠️ You can select up to 3 categories only."
    return None


def run_deal_search_sync(job_id: str, selected_categories: List[str], deal_framework):
    """
    Synchronous wrapper for deal search to run in thread with proper logging context
    """
    try:
        logging.info(f"Starting deal search for job {job_id} with categories: {selected_categories}")
        active_jobs[job_id]["status"] = "running"
        active_jobs[job_id]["start_time"] = datetime.now()
        
        # This mirrors the do_run function from Gradio
        new_opportunities = deal_framework.run(selected_categories)
        table_data = table_for(new_opportunities)
        
        active_jobs[job_id]["status"] = "completed"
        active_jobs[job_id]["results"] = table_data
        active_jobs[job_id]["total_count"] = len(new_opportunities)
        active_jobs[job_id]["end_time"] = datetime.now()
        
        logging.info(f"Deal search completed for job {job_id}. Found {len(new_opportunities)} opportunities.")
        
    except Exception as e:
        logging.error(f"Error in deal search for job {job_id}: {str(e)}")
        active_jobs[job_id]["status"] = "error"
        active_jobs[job_id]["error"] = str(e)


async def run_deal_search_background(job_id: str, selected_categories: List[str], deal_framework):
    """
    Background task to run deal search with proper thread handling for logging
    """
    # Run the synchronous operation in a thread executor to maintain logging context
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, 
        run_deal_search_sync,
        job_id,
        selected_categories, 
        deal_framework
    )


@router.get("/categories")
async def get_categories():
    """
    Get available categories - Direct replacement for Gradio CheckboxGroup choices
    """
    try:
        categories = [
            {"name": name, "display_name": name.replace("_", " ").title()} 
            for name in CATEGORY_FEEDS.keys()
        ]
        return {"categories": categories}
    except Exception as e:
        logging.error(f"Error getting categories: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving categories: {str(e)}")


@router.post("/search", response_model=SearchResponse)
async def start_deal_search(
    request: CategoryRequest, 
    background_tasks: BackgroundTasks,
    fastapi_request: Request
):
    """
    Start deal search - Direct replacement for Gradio button click
    """
    try:
        # Validate categories (same logic as Gradio)
        validation_error = validate_categories_logic(request.selected_categories)
        if validation_error:
            raise HTTPException(status_code=400, detail=validation_error)
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job tracking
        active_jobs[job_id] = {
            "status": "initializing",
            "selected_categories": request.selected_categories,
            "created_at": datetime.now(),
            "results": [],
            "error": None
        }
        
        # Get deal framework from app state
        deal_framework = fastapi_request.app.state.get_deal_framework()
        
        # Use asyncio.create_task instead of BackgroundTasks for better logging context
        asyncio.create_task(
            run_deal_search_background(job_id, request.selected_categories, deal_framework)
        )
        
        return SearchResponse(
            job_id=job_id,
            status="started",
            message="Deal search initiated successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error starting deal search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting search: {str(e)}")


@router.get("/results/{job_id}", response_model=SearchResultsResponse)
async def get_search_results(job_id: str):
    """
    Get search results - Direct replacement for Gradio table display
    """
    try:
        if job_id not in active_jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = active_jobs[job_id]
        
        return SearchResultsResponse(
            status=job["status"],
            results=job.get("results", []),
            error_message=job.get("error"),
            total_count=job.get("total_count", 0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting results for job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving results: {str(e)}")


@router.get("/status")
async def get_app_status(request: Request):
    """
    Get application status - New endpoint for React frontend
    """
    try:
        deal_framework = request.app.state.get_deal_framework()
        
        # Count active jobs
        active_count = sum(1 for job in active_jobs.values() if job["status"] == "running")
        current_job = None
        
        # Find most recent active job
        for job_id, job in active_jobs.items():
            if job["status"] == "running":
                current_job = job_id
                break
        
        return AppStatus(
            is_running=active_count > 0,
            current_job_id=current_job,
            total_deals_found=len(deal_framework.memory),
            categories_available=list(CATEGORY_FEEDS.keys())
        )
        
    except Exception as e:
        logging.error(f"Error getting app status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving status: {str(e)}")


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """
    Cancel a running job
    """
    try:
        if job_id not in active_jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = active_jobs[job_id]
        if job["status"] == "running":
            job["status"] = "cancelled"
            return {"message": f"Job {job_id} cancelled successfully"}
        else:
            return {"message": f"Job {job_id} is not running (status: {job['status']})"}
            
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error cancelling job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error cancelling job: {str(e)}")


@router.get("/jobs")
async def list_jobs():
    """
    List all jobs with their status
    """
    try:
        jobs_summary = []
        for job_id, job in active_jobs.items():
            jobs_summary.append({
                "job_id": job_id,
                "status": job["status"],
                "categories": job["selected_categories"],
                "created_at": job["created_at"].isoformat(),
                "total_results": job.get("total_count", 0)
            })
        
        return {"jobs": jobs_summary}
        
    except Exception as e:
        logging.error(f"Error listing jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing jobs: {str(e)}")


@router.post("/clear-results")
async def clear_all_results():
    """
    Clear all job results - useful for demo resets
    """
    try:
        global active_jobs
        completed_jobs = [job_id for job_id, job in active_jobs.items() 
                         if job["status"] in ["completed", "error", "cancelled"]]
        
        for job_id in completed_jobs:
            del active_jobs[job_id]
        
        return {"message": f"Cleared {len(completed_jobs)} completed jobs"}
        
    except Exception as e:
        logging.error(f"Error clearing results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing results: {str(e)}")