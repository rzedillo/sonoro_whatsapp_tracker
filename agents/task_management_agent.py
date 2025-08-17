"""
Task Management Agent for WhatsApp Task Tracker
Enhanced Framework V3.1 Implementation
Ported from JavaScript TaskManager class
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
from sqlalchemy import text, desc, and_, or_

from core.base_agent import BaseAgent
from core.redis_client import cache_manager
from database.connection import get_db_context
from database.models import Task, TaskHistory, UserPattern, TaskStatus, TaskPriority


class TaskManagementAgent(BaseAgent):
    """
    Task management agent handling task CRUD operations and analytics
    
    Features:
    - Task creation from message analysis
    - Task querying and filtering
    - User productivity analysis
    - Task history tracking
    - Duplicate task detection
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        
        # Configuration
        self.auto_assign = config.get("auto_assign", True)
        self.priority_detection = config.get("priority_detection", True)
        self.duplicate_threshold = config.get("duplicate_threshold", 0.8)
        
        # Analytics cache
        self.user_patterns_cache = {}
        self.pattern_cache_duration = 1800  # 30 minutes
    
    async def _initialize_agent(self):
        """Initialize task management agent"""
        self.logger.info("Initializing task management agent")
        
        # Initialize analytics patterns
        await self._initialize_user_patterns()
        
        self.logger.info("Task management agent initialized")
    
    async def _cleanup_agent(self):
        """Cleanup task management agent"""
        self.logger.info("Task management agent cleanup completed")
    
    async def _process_data(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process task management commands"""
        command = data.get("command", "")
        action = data.get("action", "")
        
        try:
            # Handle different types of requests
            if "task_detected" in data:
                return await self._create_task_from_analysis(data)
            elif command == "get_tasks":
                return await self._get_tasks(data)
            elif command == "update_task":
                return await self._update_task(data)
            elif command == "complete_task":
                return await self._complete_task(data)
            elif command == "delete_task":
                return await self._delete_task(data)
            elif command == "get_user_patterns":
                return await self._get_user_patterns(data)
            elif command == "analyze_productivity":
                return await self._analyze_productivity(data)
            elif action == "task_update":
                return await self._handle_task_update(data)
            else:
                return {
                    "success": False,
                    "error": f"Unknown command: {command}"
                }
                
        except Exception as e:
            self.logger.error("Task management processing failed", error=str(e))
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _health_check(self) -> Dict[str, Any]:
        """Task management agent health check"""
        try:
            with get_db_context() as db:
                # Check database connectivity
                total_tasks = db.execute(text("SELECT COUNT(*) FROM tasks")).scalar()
                pending_tasks = db.execute(
                    text("SELECT COUNT(*) FROM tasks WHERE estado = :status"),
                    {"status": TaskStatus.PENDING}
                ).scalar()
                
                return {
                    "database_connected": True,
                    "total_tasks": total_tasks,
                    "pending_tasks": pending_tasks,
                    "cache_patterns": len(self.user_patterns_cache),
                }
        except Exception as e:
            return {
                "database_connected": False,
                "error": str(e)
            }
    
    async def _create_task_from_analysis(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create task from message analysis"""
        try:
            # Extract task information
            task_info = analysis_data.get("task_info", {})
            message_data = analysis_data.get("message_data", {})
            
            # Check for duplicate tasks
            duplicate_task = await self._find_duplicate_task(task_info)
            if duplicate_task:
                self.logger.info("Duplicate task detected", duplicate_id=duplicate_task["id"])
                return {
                    "success": False,
                    "error": "Duplicate task detected",
                    "duplicate_task": duplicate_task
                }
            
            # Create new task
            task_data = {
                "descripcion": task_info.get("description", ""),
                "responsable": task_info.get("assigned_to"),
                "fecha_limite": task_info.get("due_date"),
                "prioridad": task_info.get("priority", TaskPriority.MEDIUM),
                "estado": TaskStatus.PENDING,
                "mensaje_original": message_data.get("text", ""),
                "autor_mensaje": message_data.get("author", ""),
                "grupo_id": message_data.get("chat_name", ""),
                "grupo_nombre": message_data.get("chat_name", ""),
                "mensaje_id": message_data.get("message_id", ""),
                "confidence_score": analysis_data.get("confidence", 0.8),
                "analysis_metadata": json.dumps(analysis_data),
            }
            
            # Auto-assign if enabled
            if self.auto_assign and not task_data["responsable"]:
                task_data["responsable"] = await self._auto_assign_task(task_data)
            
            # Save to database
            task_id = await self._save_task(task_data)
            
            # Update user patterns
            await self._update_user_patterns(task_data)
            
            # Cache for quick access
            await cache_manager.set(
                f"task:{task_id}",
                task_data,
                expire=3600
            )
            
            self.logger.info("Task created successfully", task_id=task_id)
            
            return {
                "success": True,
                "task_id": task_id,
                "task_data": task_data,
                "message": "Task created successfully"
            }
            
        except Exception as e:
            self.logger.error("Task creation failed", error=str(e))
            raise
    
    async def _save_task(self, task_data: Dict[str, Any]) -> int:
        """Save task to database"""
        with get_db_context() as db:
            task = Task(**task_data)
            db.add(task)
            db.flush()  # Get the ID
            
            # Create initial history entry
            history = TaskHistory(
                task_id=task.id,
                action="created",
                previous_state=None,
                new_state=TaskStatus.PENDING,
                changed_by="system",
                notes="Task created from WhatsApp message"
            )
            db.add(history)
            
            return task.id
    
    async def _find_duplicate_task(self, task_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find potentially duplicate tasks"""
        try:
            description = task_info.get("description", "").lower()
            assigned_to = task_info.get("assigned_to", "")
            
            if not description:
                return None
            
            with get_db_context() as db:
                # Look for similar tasks created recently
                recent_cutoff = datetime.utcnow() - timedelta(days=7)
                
                query = text("""
                    SELECT id, descripcion, responsable, estado, confidence_score
                    FROM tasks 
                    WHERE timestamp > :cutoff 
                    AND estado != :completed
                    AND LOWER(descripcion) LIKE :description
                """)
                
                similar_tasks = db.execute(query, {
                    "cutoff": recent_cutoff,
                    "completed": TaskStatus.COMPLETED,
                    "description": f"%{description[:50]}%"
                }).fetchall()
                
                for task in similar_tasks:
                    # Simple similarity check
                    similarity = self._calculate_similarity(description, task[1].lower())
                    if similarity > self.duplicate_threshold:
                        return {
                            "id": task[0],
                            "description": task[1],
                            "assigned_to": task[2],
                            "status": task[3],
                            "similarity": similarity
                        }
                
                return None
                
        except Exception as e:
            self.logger.error("Duplicate detection failed", error=str(e))
            return None
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity"""
        # Simple word overlap similarity
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    async def _auto_assign_task(self, task_data: Dict[str, Any]) -> Optional[str]:
        """Auto-assign task based on user patterns"""
        try:
            grupo_nombre = task_data.get("grupo_nombre", "")
            
            # Get user patterns for this group
            patterns = await self._get_group_user_patterns(grupo_nombre)
            
            if not patterns:
                return None
            
            # Find best candidate based on workload and expertise
            best_candidate = None
            best_score = 0
            
            for user_name, pattern in patterns.items():
                # Calculate assignment score
                score = self._calculate_assignment_score(pattern, task_data)
                
                if score > best_score:
                    best_score = score
                    best_candidate = user_name
            
            return best_candidate
            
        except Exception as e:
            self.logger.error("Auto-assignment failed", error=str(e))
            return None
    
    def _calculate_assignment_score(self, user_pattern: Dict[str, Any], task_data: Dict[str, Any]) -> float:
        """Calculate assignment score for user"""
        score = 0.5  # Base score
        
        # Factor in productivity score
        productivity = user_pattern.get("productivity_score", 0.5)
        score += productivity * 0.3
        
        # Factor in current workload (inverse)
        pending_tasks = user_pattern.get("pending_tasks", 0)
        workload_factor = max(0, 1 - (pending_tasks / 10))  # Penalty for high workload
        score += workload_factor * 0.2
        
        return min(1.0, score)
    
    async def _get_tasks(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get tasks with filtering"""
        try:
            # Extract filters
            filters = data.get("filters", {})
            limit = data.get("limit", 50)
            offset = data.get("offset", 0)
            
            with get_db_context() as db:
                query = db.query(Task)
                
                # Apply filters
                if filters.get("status"):
                    query = query.filter(Task.estado == filters["status"])
                
                if filters.get("assigned_to"):
                    query = query.filter(Task.responsable == filters["assigned_to"])
                
                if filters.get("priority"):
                    query = query.filter(Task.prioridad == filters["priority"])
                
                if filters.get("group"):
                    query = query.filter(Task.grupo_nombre == filters["group"])
                
                if filters.get("date_from"):
                    query = query.filter(Task.timestamp >= filters["date_from"])
                
                if filters.get("date_to"):
                    query = query.filter(Task.timestamp <= filters["date_to"])
                
                # Order by timestamp (newest first)
                query = query.order_by(desc(Task.timestamp))
                
                # Pagination
                total_count = query.count()
                tasks = query.offset(offset).limit(limit).all()
                
                # Convert to dictionaries
                task_list = [task.to_dict() for task in tasks]
                
                return {
                    "success": True,
                    "tasks": task_list,
                    "total_count": total_count,
                    "limit": limit,
                    "offset": offset,
                }
                
        except Exception as e:
            self.logger.error("Get tasks failed", error=str(e))
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _update_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update task"""
        try:
            task_id = data.get("task_id")
            updates = data.get("updates", {})
            changed_by = data.get("changed_by", "system")
            
            with get_db_context() as db:
                task = db.query(Task).filter(Task.id == task_id).first()
                
                if not task:
                    return {
                        "success": False,
                        "error": "Task not found"
                    }
                
                # Track changes for history
                changes = {}
                for field, new_value in updates.items():
                    if hasattr(task, field):
                        old_value = getattr(task, field)
                        if old_value != new_value:
                            changes[field] = {"old": old_value, "new": new_value}
                            setattr(task, field, new_value)
                
                # Update timestamp
                task.last_updated = datetime.utcnow()
                
                # Create history entries
                for field, change in changes.items():
                    history = TaskHistory(
                        task_id=task_id,
                        action=f"updated_{field}",
                        previous_state=str(change["old"]),
                        new_state=str(change["new"]),
                        changed_by=changed_by,
                        notes=f"Updated {field}"
                    )
                    db.add(history)
                
                # Update cache
                await cache_manager.set(
                    f"task:{task_id}",
                    task.to_dict(),
                    expire=3600
                )
                
                return {
                    "success": True,
                    "task_id": task_id,
                    "changes": changes,
                    "message": "Task updated successfully"
                }
                
        except Exception as e:
            self.logger.error("Task update failed", error=str(e))
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _complete_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mark task as completed"""
        try:
            task_id = data.get("task_id")
            completed_by = data.get("completed_by", "system")
            notes = data.get("notes", "")
            
            updates = {
                "estado": TaskStatus.COMPLETED,
                "completion_date": datetime.utcnow().isoformat()
            }
            
            return await self._update_task({
                "task_id": task_id,
                "updates": updates,
                "changed_by": completed_by
            })
            
        except Exception as e:
            self.logger.error("Task completion failed", error=str(e))
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _delete_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Delete task (soft delete by marking as cancelled)"""
        try:
            task_id = data.get("task_id")
            deleted_by = data.get("deleted_by", "system")
            
            updates = {
                "estado": TaskStatus.CANCELLED
            }
            
            return await self._update_task({
                "task_id": task_id,
                "updates": updates,
                "changed_by": deleted_by
            })
            
        except Exception as e:
            self.logger.error("Task deletion failed", error=str(e))
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _get_user_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get user productivity patterns"""
        try:
            user_name = data.get("user_name")
            group_name = data.get("group_name")
            
            patterns = await self._analyze_user_patterns(user_name, group_name)
            
            return {
                "success": True,
                "user_name": user_name,
                "group_name": group_name,
                "patterns": patterns
            }
            
        except Exception as e:
            self.logger.error("Get user patterns failed", error=str(e))
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _analyze_user_patterns(self, user_name: str, group_name: Optional[str] = None) -> Dict[str, Any]:
        """Analyze user productivity patterns"""
        try:
            # Check cache first
            cache_key = f"user_patterns:{user_name}:{group_name or 'all'}"
            cached_patterns = await cache_manager.get_user_patterns(cache_key)
            
            if cached_patterns:
                return cached_patterns
            
            with get_db_context() as db:
                # Base query
                query = db.query(Task).filter(Task.responsable == user_name)
                
                if group_name:
                    query = query.filter(Task.grupo_nombre == group_name)
                
                # Get all tasks for user
                tasks = query.all()
                
                if not tasks:
                    return {
                        "total_tasks": 0,
                        "completed_tasks": 0,
                        "completion_rate": 0.0,
                        "average_completion_time": 0.0,
                        "most_common_priority": None,
                        "productivity_score": 0.5
                    }
                
                # Calculate patterns
                total_tasks = len(tasks)
                completed_tasks = len([t for t in tasks if t.estado == TaskStatus.COMPLETED])
                completion_rate = completed_tasks / total_tasks if total_tasks > 0 else 0
                
                # Calculate average completion time
                completion_times = []
                for task in tasks:
                    if task.estado == TaskStatus.COMPLETED and task.completion_date:
                        try:
                            created = task.timestamp
                            completed = datetime.fromisoformat(task.completion_date)
                            completion_time = (completed - created).total_seconds() / 3600  # hours
                            completion_times.append(completion_time)
                        except:
                            continue
                
                avg_completion_time = sum(completion_times) / len(completion_times) if completion_times else 0
                
                # Most common priority
                priorities = [t.prioridad for t in tasks]
                most_common_priority = max(set(priorities), key=priorities.count) if priorities else None
                
                # Calculate productivity score
                productivity_score = self._calculate_productivity_score(
                    completion_rate, avg_completion_time, total_tasks
                )
                
                patterns = {
                    "total_tasks": total_tasks,
                    "completed_tasks": completed_tasks,
                    "completion_rate": completion_rate,
                    "average_completion_time": avg_completion_time,
                    "most_common_priority": most_common_priority,
                    "productivity_score": productivity_score,
                    "analysis_date": datetime.utcnow().isoformat()
                }
                
                # Cache patterns
                await cache_manager.cache_user_patterns(cache_key, patterns)
                
                return patterns
                
        except Exception as e:
            self.logger.error("User pattern analysis failed", error=str(e))
            return {}
    
    def _calculate_productivity_score(self, completion_rate: float, avg_completion_time: float, total_tasks: int) -> float:
        """Calculate productivity score (0.0 to 1.0)"""
        score = 0.5  # Base score
        
        # Completion rate factor (0.4 weight)
        score += (completion_rate - 0.5) * 0.4
        
        # Task volume factor (0.2 weight)
        volume_factor = min(1.0, total_tasks / 20)  # Normalize to 20 tasks
        score += volume_factor * 0.2
        
        # Completion time factor (0.2 weight) - faster is better up to a point
        if avg_completion_time > 0:
            time_factor = max(0, 1 - (avg_completion_time / 72))  # 72 hours as baseline
            score += time_factor * 0.2
        
        return max(0.0, min(1.0, score))
    
    async def _initialize_user_patterns(self):
        """Initialize user patterns on startup"""
        try:
            with get_db_context() as db:
                # Get active users
                users_query = text("""
                    SELECT DISTINCT responsable 
                    FROM tasks 
                    WHERE responsable IS NOT NULL 
                    AND timestamp > :cutoff
                """)
                
                cutoff = datetime.utcnow() - timedelta(days=30)
                users = db.execute(users_query, {"cutoff": cutoff}).fetchall()
                
                # Pre-compute patterns for active users
                for (user_name,) in users:
                    await self._analyze_user_patterns(user_name)
                
                self.logger.info("User patterns initialized", user_count=len(users))
                
        except Exception as e:
            self.logger.error("User patterns initialization failed", error=str(e))
    
    async def _update_user_patterns(self, task_data: Dict[str, Any]):
        """Update user patterns when new task is created"""
        try:
            user_name = task_data.get("responsable")
            group_name = task_data.get("grupo_nombre")
            
            if user_name:
                # Invalidate cache to force recalculation
                cache_key = f"user_patterns:{user_name}:{group_name or 'all'}"
                await cache_manager.delete(cache_key)
                
                # Recalculate patterns
                await self._analyze_user_patterns(user_name, group_name)
                
        except Exception as e:
            self.logger.error("User pattern update failed", error=str(e))
    
    async def _get_group_user_patterns(self, group_name: str) -> Dict[str, Dict[str, Any]]:
        """Get user patterns for a specific group"""
        try:
            with get_db_context() as db:
                users_query = text("""
                    SELECT DISTINCT responsable 
                    FROM tasks 
                    WHERE grupo_nome = :group_name 
                    AND responsable IS NOT NULL
                """)
                
                users = db.execute(users_query, {"group_name": group_name}).fetchall()
                
                patterns = {}
                for (user_name,) in users:
                    user_patterns = await self._analyze_user_patterns(user_name, group_name)
                    patterns[user_name] = user_patterns
                
                return patterns
                
        except Exception as e:
            self.logger.error("Group user patterns failed", error=str(e))
            return {}
    
    async def _handle_task_update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task update from external sources"""
        return await self._update_task(data)
    
    async def _analyze_productivity(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall productivity metrics"""
        try:
            group_name = data.get("group_name")
            days = data.get("days", 30)
            
            cutoff = datetime.utcnow() - timedelta(days=days)
            
            with get_db_context() as db:
                # Base query
                query = db.query(Task).filter(Task.timestamp >= cutoff)
                
                if group_name:
                    query = query.filter(Task.grupo_nome == group_name)
                
                tasks = query.all()
                
                # Calculate metrics
                total_tasks = len(tasks)
                completed_tasks = len([t for t in tasks if t.estado == TaskStatus.COMPLETED])
                pending_tasks = len([t for t in tasks if t.estado == TaskStatus.PENDING])
                
                # Group by priority
                priority_breakdown = {}
                for priority in TaskPriority:
                    priority_tasks = [t for t in tasks if t.prioridad == priority]
                    priority_breakdown[priority] = len(priority_tasks)
                
                # Group by user
                user_breakdown = {}
                for task in tasks:
                    if task.responsable:
                        if task.responsable not in user_breakdown:
                            user_breakdown[task.responsable] = {"total": 0, "completed": 0}
                        user_breakdown[task.responsable]["total"] += 1
                        if task.estado == TaskStatus.COMPLETED:
                            user_breakdown[task.responsable]["completed"] += 1
                
                return {
                    "success": True,
                    "period_days": days,
                    "group_name": group_name,
                    "total_tasks": total_tasks,
                    "completed_tasks": completed_tasks,
                    "pending_tasks": pending_tasks,
                    "completion_rate": completed_tasks / total_tasks if total_tasks > 0 else 0,
                    "priority_breakdown": priority_breakdown,
                    "user_breakdown": user_breakdown,
                    "analysis_date": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            self.logger.error("Productivity analysis failed", error=str(e))
            return {
                "success": False,
                "error": str(e)
            }