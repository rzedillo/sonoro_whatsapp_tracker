"""
Notifications API endpoints for WhatsApp Task Tracker
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import structlog

from core.orchestrator import get_orchestrator

logger = structlog.get_logger(__name__)
router = APIRouter()


class NotificationRequest(BaseModel):
    type: str
    title: str
    message: str
    details: Optional[Dict[str, Any]] = None
    priority: str = "medium"


@router.get("/")
async def get_notifications(limit: int = Query(default=20, ge=1, le=100)):
    """Get recent notifications"""
    try:
        orchestrator = get_orchestrator()
        notification_agent = orchestrator.agents.get("notifier")
        
        if not notification_agent:
            raise HTTPException(status_code=503, detail="Notification service unavailable")
        
        notifications = await notification_agent.get_recent_notifications(limit)
        
        return {
            "success": True,
            "data": {
                "notifications": notifications,
                "count": len(notifications)
            }
        }
        
    except Exception as e:
        logger.error("Get notifications failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/")
async def send_notification(notification: NotificationRequest):
    """Send custom notification"""
    try:
        orchestrator = get_orchestrator()
        notification_agent = orchestrator.agents.get("notifier")
        
        if not notification_agent:
            raise HTTPException(status_code=503, detail="Notification service unavailable")
        
        result = await notification_agent.process({
            "command": "send_notification",
            "type": notification.type,
            "title": notification.title,
            "message": notification.message,
            "details": notification.details,
            "priority": notification.priority
        })
        
        return result
        
    except Exception as e:
        logger.error("Send notification failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/check-reminders")
async def check_reminders():
    """Check for due date reminders"""
    try:
        orchestrator = get_orchestrator()
        notification_agent = orchestrator.agents.get("notifier")
        
        if not notification_agent:
            raise HTTPException(status_code=503, detail="Notification service unavailable")
        
        result = await notification_agent.process({
            "command": "check_reminders"
        })
        
        return result
        
    except Exception as e:
        logger.error("Check reminders failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")