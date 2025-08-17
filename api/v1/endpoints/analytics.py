"""
Analytics API endpoints for WhatsApp Task Tracker
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import structlog

from core.orchestrator import get_orchestrator

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/productivity")
async def get_productivity_analytics(
    days: int = Query(default=30, ge=1, le=365),
    group_name: Optional[str] = Query(default=None)
):
    """Get productivity analytics"""
    try:
        orchestrator = get_orchestrator()
        task_agent = orchestrator.agents.get("task_manager")
        
        if not task_agent:
            raise HTTPException(status_code=503, detail="Task management service unavailable")
        
        result = await task_agent.process({
            "command": "analyze_productivity",
            "days": days,
            "group_name": group_name
        })
        
        return result
        
    except Exception as e:
        logger.error("Productivity analytics failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/user-patterns/{user_name}")
async def get_user_patterns(user_name: str, group_name: Optional[str] = None):
    """Get user productivity patterns"""
    try:
        orchestrator = get_orchestrator()
        task_agent = orchestrator.agents.get("task_manager")
        
        if not task_agent:
            raise HTTPException(status_code=503, detail="Task management service unavailable")
        
        result = await task_agent.process({
            "command": "get_user_patterns",
            "user_name": user_name,
            "group_name": group_name
        })
        
        return result
        
    except Exception as e:
        logger.error("User patterns failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")