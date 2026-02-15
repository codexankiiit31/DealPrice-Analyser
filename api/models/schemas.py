"""
Pydantic schemas for FastAPI - matches Gradio data structures
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class CategoryRequest(BaseModel):
    """Request model for category selection"""
    selected_categories: List[str] = Field(
        ..., 
        description="List of selected categories", 
        min_items=1, 
        max_items=3
    )
    
    @validator('selected_categories')
    def validate_categories(cls, v):
        """Validate category selection - mirrors validate_categories() from Gradio"""
        if len(v) == 0:
            raise ValueError("Please select at least one category before running.")
        if len(v) > 3:
            raise ValueError("You can select up to 3 categories only.")
        return v


class DealData(BaseModel):
    """Deal information structure"""
    product_description: str
    price: float
    url: str


class OpportunityData(BaseModel):
    """Opportunity structure - mirrors the Opportunity class"""
    deal: DealData
    estimate: float
    discount: float
    
    def to_table_row(self) -> List[str]:
        """Convert structured data into table row format - mirrors table_for() from Gradio"""
        return [
            self.deal.product_description,
            f"${self.deal.price:.2f}",
            f"${self.estimate:.2f}",
            f"${self.discount:.2f}",
            f'<a href="{self.deal.url}" target="_blank" style="background-color:#007bff;color:white;padding:8px 16px;text-decoration:none;border-radius:4px;font-weight:bold;">🔗 View Deal</a>'
        ]


class SearchResponse(BaseModel):
    """Response model for deal search"""
    job_id: str
    status: str
    message: Optional[str] = None


class SearchResultsResponse(BaseModel):
    """Response model for search results"""
    status: str
    results: List[List[str]]  # Table format
    error_message: Optional[str] = None
    total_count: int = 0


class LogMessage(BaseModel):
    """Log message structure for WebSocket"""
    timestamp: str
    level: str
    message: str
    formatted_message: str


class CategoryInfo(BaseModel):
    """Category information"""
    name: str
    display_name: str
    description: Optional[str] = None


class AppStatus(BaseModel):
    """Application status response"""
    is_running: bool
    current_job_id: Optional[str] = None
    total_deals_found: int
    categories_available: List[str]


class WebSocketMessage(BaseModel):
    """WebSocket message structure"""
    type: str  # "log", "result", "error", "status"
    data: Dict[str, Any]
    timestamp: str


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None