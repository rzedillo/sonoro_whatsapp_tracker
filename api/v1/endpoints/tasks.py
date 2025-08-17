"""
Task API endpoints for WhatsApp Task Tracker
Enhanced Framework V3.1 Implementation
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
import structlog

from core.orchestrator import get_orchestrator
from database.models import TaskStatus, TaskPriority

logger = structlog.get_logger(__name__)
router = APIRouter()


# Pydantic models for request/response
class TaskFilter(BaseModel):
    status: Optional[str] = None
    assigned_to: Optional[str] = None
    priority: Optional[str] = None
    group: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None


class TaskCreate(BaseModel):
    descripcion: str = Field(..., min_length=1, max_length=500)
    responsable: Optional[str] = None
    fecha_limite: Optional[str] = None
    prioridad: str = Field(default=TaskPriority.MEDIUM)
    grupo_nombre: Optional[str] = None
    mensaje_original: Optional[str] = None
    autor_mensaje: Optional[str] = None


class TaskUpdate(BaseModel):
    descripcion: Optional[str] = None
    responsable: Optional[str] = None
    fecha_limite: Optional[str] = None
    prioridad: Optional[str] = None
    estado: Optional[str] = None
    notes: Optional[str] = None


class TaskResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@router.get("/", response_model=Dict[str, Any])
async def get_tasks(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status: Optional[str] = Query(default=None),
    assigned_to: Optional[str] = Query(default=None),
    priority: Optional[str] = Query(default=None),
    group: Optional[str] = Query(default=None),
):
    """
    Get tasks with optional filtering and pagination
    """
    try:
        orchestrator = get_orchestrator()
        task_agent = orchestrator.agents.get("task_manager")
        
        if not task_agent:
            raise HTTPException(status_code=503, detail="Task management service unavailable")
        
        # Build filters
        filters = {}
        if status:
            filters["status"] = status
        if assigned_to:
            filters["assigned_to"] = assigned_to
        if priority:
            filters["priority"] = priority
        if group:
            filters["group"] = group
        
        # Process request
        result = await task_agent.process({
            "command": "get_tasks",
            "filters": filters,
            "limit": limit,
            "offset": offset
        })
        
        if result.get("success"):
            return {
                "success": True,
                "data": {
                    "tasks": result.get("tasks", []),
                    "total_count": result.get("total_count", 0),
                    "limit": limit,
                    "offset": offset,
                    "filters": filters
                }
            }
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Unknown error"))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Get tasks failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{task_id}", response_model=Dict[str, Any])
async def get_task(task_id: int):
    """
    Get specific task by ID
    """
    try:
        orchestrator = get_orchestrator()
        task_agent = orchestrator.agents.get("task_manager")
        
        if not task_agent:
            raise HTTPException(status_code=503, detail="Task management service unavailable")
        
        result = await task_agent.process({
            "command": "get_tasks",
            "filters": {"id": task_id},
            "limit": 1
        })
        
        if result.get("success"):
            tasks = result.get("tasks", [])
            if tasks:
                return {
                    "success": True,
                    "data": {"task": tasks[0]}
                }
            else:
                raise HTTPException(status_code=404, detail="Task not found")
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Unknown error"))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Get task failed", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/", response_model=TaskResponse)
async def create_task(task_data: TaskCreate):
    """
    Create a new task manually
    """
    try:
        orchestrator = get_orchestrator()
        task_agent = orchestrator.agents.get("task_manager")
        
        if not task_agent:
            raise HTTPException(status_code=503, detail="Task management service unavailable")
        
        # Prepare task data
        task_info = {
            "description": task_data.descripcion,
            "assigned_to": task_data.responsable,
            "priority": task_data.prioridad,
            "due_date": task_data.fecha_limite,
        }
        
        message_data = {
            "text": task_data.mensaje_original or task_data.descripcion,
            "author": task_data.autor_mensaje or "manual",
            "chat_name": task_data.grupo_nombre or "manual",
            "message_id": f"manual_{int(datetime.utcnow().timestamp())}",
        }
        
        # Process task creation
        result = await task_agent.process({
            "task_detected": True,
            "confidence": 1.0,
            "task_info": task_info,
            "message_data": message_data,
            "analysis_method": "manual"
        })
        
        if result.get("success"):
            return TaskResponse(
                success=True,
                message="Task created successfully",
                data={
                    "task_id": result.get("task_id"),
                    "task": result.get("task_data")
                }
            )
        else:
            return TaskResponse(
                success=False,
                error=result.get("error", "Task creation failed")
            )
            
    except Exception as e:
        logger.error("Create task failed", error=str(e))
        return TaskResponse(
            success=False,
            error="Internal server error"
        )


@router.put("/{task_id}", response_model=TaskResponse)
async def update_task(task_id: int, task_update: TaskUpdate, updated_by: str = "api"):
    """
    Update an existing task
    """
    try:
        orchestrator = get_orchestrator()
        task_agent = orchestrator.agents.get("task_manager")
        
        if not task_agent:
            raise HTTPException(status_code=503, detail="Task management service unavailable")
        
        # Build updates dict (only include non-None values)
        updates = {}
        if task_update.descripcion is not None:
            updates["descripcion"] = task_update.descripcion
        if task_update.responsable is not None:
            updates["responsable"] = task_update.responsable
        if task_update.fecha_limite is not None:
            updates["fecha_limite"] = task_update.fecha_limite
        if task_update.prioridad is not None:
            updates["prioridad"] = task_update.prioridad
        if task_update.estado is not None:
            updates["estado"] = task_update.estado
        
        if not updates:
            return TaskResponse(
                success=False,
                error="No updates provided"
            )
        
        # Process update
        result = await task_agent.process({
            "command": "update_task",
            "task_id": task_id,
            "updates": updates,
            "changed_by": updated_by
        })
        
        if result.get("success"):
            return TaskResponse(
                success=True,
                message="Task updated successfully",
                data={
                    "task_id": task_id,
                    "changes": result.get("changes", {})
                }
            )
        else:
            return TaskResponse(
                success=False,
                error=result.get("error", "Task update failed")
            )
            
    except Exception as e:
        logger.error("Update task failed", task_id=task_id, error=str(e))
        return TaskResponse(
            success=False,
            error="Internal server error"
        )


@router.post("/{task_id}/complete", response_model=TaskResponse)
async def complete_task(task_id: int, completed_by: str = "api", notes: Optional[str] = None):
    """
    Mark task as completed
    """
    try:
        orchestrator = get_orchestrator()
        task_agent = orchestrator.agents.get("task_manager")
        
        if not task_agent:
            raise HTTPException(status_code=503, detail="Task management service unavailable")
        
        result = await task_agent.process({
            "command": "complete_task",
            "task_id": task_id,
            "completed_by": completed_by,
            "notes": notes or ""
        })
        
        if result.get("success"):
            return TaskResponse(
                success=True,
                message="Task completed successfully",
                data={"task_id": task_id}
            )
        else:
            return TaskResponse(
                success=False,
                error=result.get("error", "Task completion failed")
            )
            
    except Exception as e:
        logger.error("Complete task failed", task_id=task_id, error=str(e))
        return TaskResponse(
            success=False,
            error="Internal server error"
        )


@router.delete("/{task_id}", response_model=TaskResponse)
async def delete_task(task_id: int, deleted_by: str = "api"):
    """
    Delete (cancel) a task
    """
    try:
        orchestrator = get_orchestrator()
        task_agent = orchestrator.agents.get("task_manager")
        
        if not task_agent:
            raise HTTPException(status_code=503, detail="Task management service unavailable")
        
        result = await task_agent.process({
            "command": "delete_task",
            "task_id": task_id,
            "deleted_by": deleted_by
        })
        
        if result.get("success"):
            return TaskResponse(
                success=True,
                message="Task deleted successfully",
                data={"task_id": task_id}
            )
        else:
            return TaskResponse(
                success=False,
                error=result.get("error", "Task deletion failed")
            )
            
    except Exception as e:
        logger.error("Delete task failed", task_id=task_id, error=str(e))
        return TaskResponse(
            success=False,
            error="Internal server error"
        )


@router.get("/{task_id}/history", response_model=Dict[str, Any])
async def get_task_history(task_id: int):
    """
    Get task change history
    """
    try:
        # This would get task history from the database
        # For now, return placeholder
        return {
            "success": True,
            "data": {
                "task_id": task_id,
                "history": []  # Would be populated from TaskHistory model
            }
        }
        
    except Exception as e:
        logger.error("Get task history failed", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/stats/summary", response_model=Dict[str, Any])
async def get_task_summary():
    """
    Get task statistics summary
    """
    try:
        orchestrator = get_orchestrator()
        task_agent = orchestrator.agents.get("task_manager")
        
        if not task_agent:
            raise HTTPException(status_code=503, detail="Task management service unavailable")
        
        # Get overall analytics
        result = await task_agent.process({
            "command": "analyze_productivity",
            "days": 30
        })
        
        if result.get("success"):
            return {
                "success": True,
                "data": result
            }
        else:
            return {
                "success": False,
                "error": result.get("error", "Analytics failed")
            }
            
    except Exception as e:
        logger.error("Get task summary failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")