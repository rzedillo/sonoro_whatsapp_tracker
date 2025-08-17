"""
Notification Agent for WhatsApp Task Tracker
Enhanced Framework V3.1 Implementation
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

from core.base_agent import BaseAgent
from core.redis_client import cache_manager
from database.connection import get_db_context
from database.models import Task, TaskStatus


class NotificationAgent(BaseAgent):
    """
    Notification agent for sending alerts and updates
    
    Features:
    - Task creation notifications
    - Due date reminders
    - Assignment notifications
    - Status update alerts
    - Escalation notifications
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        
        # Configuration
        self.enabled = config.get("enabled", True)
        self.notification_channels = config.get("channels", ["console", "cache", "whatsapp"])
        self.reminder_hours = config.get("reminder_hours", [24, 2])  # Hours before due date
        
        # WhatsApp integration
        self.whatsapp_agent = None
        
        # Notification queue
        self.notification_queue = []
        self.max_queue_size = config.get("max_queue_size", 100)
        
        # Notification types
        self.notification_types = {
            "task_created": self._notify_task_created,
            "task_assigned": self._notify_task_assigned,
            "task_completed": self._notify_task_completed,
            "task_overdue": self._notify_task_overdue,
            "due_reminder": self._notify_due_reminder,
            "status_update": self._notify_status_update,
        }
    
    async def _initialize_agent(self):
        """Initialize notification agent"""
        self.logger.info("Initializing notification agent")
        
        if not self.enabled:
            self.logger.info("Notification agent disabled")
            return
        
        # Start reminder checking
        await self._start_reminder_service()
        
        self.logger.info("Notification agent initialized")
    
    async def _cleanup_agent(self):
        """Cleanup notification agent"""
        self.logger.info("Notification agent cleanup completed")
    
    async def _process_data(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process notification requests"""
        try:
            if not self.enabled:
                return {
                    "success": True,
                    "message": "Notifications disabled"
                }
            
            # Handle different notification triggers
            if "task_detected" in data and data.get("task_detected"):
                return await self._handle_task_creation_notification(data)
            elif "task_updated" in data:
                return await self._handle_task_update_notification(data)
            elif data.get("command") == "send_notification":
                return await self._send_custom_notification(data)
            elif data.get("command") == "check_reminders":
                return await self._check_due_reminders()
            else:
                return {
                    "success": True,
                    "message": "No notification action required"
                }
                
        except Exception as e:
            self.logger.error("Notification processing failed", error=str(e))
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _health_check(self) -> Dict[str, Any]:
        """Notification agent health check"""
        return {
            "enabled": self.enabled,
            "channels": self.notification_channels,
            "queue_size": len(self.notification_queue),
            "max_queue_size": self.max_queue_size,
        }
    
    async def _start_reminder_service(self):
        """Start background reminder service"""
        # This would typically be a background task
        # For now, we'll implement on-demand checking
        self.logger.info("Reminder service ready (on-demand)")
    
    async def _handle_task_creation_notification(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task creation notification"""
        try:
            task_info = data.get("task_info", {})
            message_data = data.get("message_data", {})
            
            notification = {
                "type": "task_created",
                "title": "Nueva Tarea Detectada",
                "message": f"Se ha detectado una nueva tarea: {task_info.get('description', 'Sin descripción')}",
                "details": {
                    "task": task_info,
                    "source": message_data.get("chat_name", "Unknown"),
                    "author": message_data.get("author", "Unknown"),
                    "confidence": data.get("confidence", 0.0),
                },
                "timestamp": datetime.utcnow().isoformat(),
                "priority": self._get_notification_priority(task_info),
            }
            
            return await self._send_notification(notification)
            
        except Exception as e:
            self.logger.error("Task creation notification failed", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def _handle_task_update_notification(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task update notification"""
        try:
            task_id = data.get("task_id")
            changes = data.get("changes", {})
            
            if not changes:
                return {"success": True, "message": "No changes to notify"}
            
            # Get task details
            task_details = await self._get_task_details(task_id)
            
            notification = {
                "type": "status_update",
                "title": "Tarea Actualizada",
                "message": f"La tarea '{task_details.get('descripcion', 'Unknown')}' ha sido actualizada",
                "details": {
                    "task_id": task_id,
                    "changes": changes,
                    "task": task_details,
                },
                "timestamp": datetime.utcnow().isoformat(),
                "priority": "medium",
            }
            
            return await self._send_notification(notification)
            
        except Exception as e:
            self.logger.error("Task update notification failed", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def _send_custom_notification(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send custom notification"""
        try:
            notification = {
                "type": data.get("type", "custom"),
                "title": data.get("title", "Notification"),
                "message": data.get("message", ""),
                "details": data.get("details", {}),
                "timestamp": datetime.utcnow().isoformat(),
                "priority": data.get("priority", "medium"),
            }
            
            return await self._send_notification(notification)
            
        except Exception as e:
            self.logger.error("Custom notification failed", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def _check_due_reminders(self) -> Dict[str, Any]:
        """Check for tasks due soon and send reminders"""
        try:
            reminders_sent = 0
            
            for hours in self.reminder_hours:
                due_tasks = await self._get_tasks_due_in(hours)
                
                for task in due_tasks:
                    # Check if reminder already sent
                    reminder_key = f"reminder:{task['id']}:{hours}h"
                    if await cache_manager.exists(reminder_key):
                        continue
                    
                    # Send reminder
                    notification = {
                        "type": "due_reminder",
                        "title": f"Recordatorio: Tarea vence en {hours} horas",
                        "message": f"La tarea '{task['descripcion']}' vence pronto",
                        "details": {
                            "task": task,
                            "hours_remaining": hours,
                        },
                        "timestamp": datetime.utcnow().isoformat(),
                        "priority": "high" if hours <= 2 else "medium",
                    }
                    
                    await self._send_notification(notification)
                    
                    # Mark reminder as sent
                    await cache_manager.set(reminder_key, True, expire=hours * 3600)
                    reminders_sent += 1
            
            return {
                "success": True,
                "reminders_sent": reminders_sent
            }
            
        except Exception as e:
            self.logger.error("Due reminders check failed", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def _get_tasks_due_in(self, hours: int) -> List[Dict[str, Any]]:
        """Get tasks due in specified hours"""
        try:
            with get_db_context() as db:
                # Calculate time window
                now = datetime.utcnow()
                start_time = now + timedelta(hours=hours-1)
                end_time = now + timedelta(hours=hours+1)
                
                # Query tasks with due dates in the window
                tasks = db.query(Task).filter(
                    Task.estado == TaskStatus.PENDING,
                    Task.fecha_limite.isnot(None)
                ).all()
                
                # Filter by actual due date (fecha_limite is stored as string)
                due_tasks = []
                for task in tasks:
                    if task.fecha_limite:
                        try:
                            # Parse due date (assuming ISO format or simple date)
                            due_date = datetime.fromisoformat(task.fecha_limite)
                            if start_time <= due_date <= end_time:
                                due_tasks.append(task.to_dict())
                        except ValueError:
                            # Skip if date format is not recognized
                            continue
                
                return due_tasks
                
        except Exception as e:
            self.logger.error("Tasks due query failed", error=str(e))
            return []
    
    async def _send_notification(self, notification: Dict[str, Any]) -> Dict[str, Any]:
        """Send notification through configured channels"""
        try:
            results = {}
            
            for channel in self.notification_channels:
                if channel == "console":
                    results["console"] = await self._send_console_notification(notification)
                elif channel == "cache":
                    results["cache"] = await self._send_cache_notification(notification)
                elif channel == "webhook":
                    results["webhook"] = await self._send_webhook_notification(notification)
                elif channel == "email":
                    results["email"] = await self._send_email_notification(notification)
                elif channel == "whatsapp":
                    results["whatsapp"] = await self._send_whatsapp_notification(notification)
            
            # Add to queue for web interface
            await self._add_to_notification_queue(notification)
            
            self.logger.info(
                "Notification sent",
                type=notification["type"],
                channels=list(results.keys())
            )
            
            return {
                "success": True,
                "notification_id": notification.get("id"),
                "channels": results
            }
            
        except Exception as e:
            self.logger.error("Notification sending failed", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def _send_console_notification(self, notification: Dict[str, Any]) -> Dict[str, Any]:
        """Send notification to console/logs"""
        try:
            self.logger.info(
                "NOTIFICATION",
                title=notification["title"],
                message=notification["message"],
                type=notification["type"],
                priority=notification["priority"]
            )
            
            return {"success": True, "method": "console"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _send_cache_notification(self, notification: Dict[str, Any]) -> Dict[str, Any]:
        """Send notification to Redis cache for real-time access"""
        try:
            # Store in notifications list
            await cache_manager.lpush("notifications", notification)
            
            # Keep only recent notifications
            await cache_manager.ltrim("notifications", 0, 99)  # Keep last 100
            
            # Store with expiration for specific access
            notification_id = f"notification:{notification['type']}:{int(datetime.utcnow().timestamp())}"
            await cache_manager.set(notification_id, notification, expire=86400)  # 24 hours
            
            return {"success": True, "method": "cache", "id": notification_id}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _send_webhook_notification(self, notification: Dict[str, Any]) -> Dict[str, Any]:
        """Send notification via webhook (placeholder)"""
        # Placeholder for webhook implementation
        return {"success": False, "error": "Webhook not implemented"}
    
    async def _send_email_notification(self, notification: Dict[str, Any]) -> Dict[str, Any]:
        """Send notification via email (placeholder)"""
        # Placeholder for email implementation
        return {"success": False, "error": "Email not implemented"}
    
    async def _send_whatsapp_notification(self, notification: Dict[str, Any]) -> Dict[str, Any]:
        """Send notification via WhatsApp"""
        try:
            if not self.whatsapp_agent:
                return {"success": False, "error": "WhatsApp agent not available"}
            
            # Format notification for WhatsApp
            whatsapp_message = await self._format_whatsapp_message(notification)
            
            # Determine chat to send to
            chat_name = notification.get("details", {}).get("source")
            if not chat_name:
                # For general notifications, get from task details
                task_details = notification.get("details", {}).get("task", {})
                chat_name = task_details.get("grupo_nome")
            
            if not chat_name:
                return {"success": False, "error": "No chat specified for WhatsApp notification"}
            
            # Send message via WhatsApp agent
            result = await self.whatsapp_agent.process({
                "command": "send_message",
                "chat_name": chat_name,
                "message": whatsapp_message
            })
            
            return {
                "success": result.get("success", False),
                "method": "whatsapp",
                "chat_name": chat_name,
                "error": result.get("error") if not result.get("success") else None
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _format_whatsapp_message(self, notification: Dict[str, Any]) -> str:
        """Format notification for WhatsApp display"""
        try:
            notification_type = notification.get("type", "")
            title = notification.get("title", "")
            message = notification.get("message", "")
            priority = notification.get("priority", "medium")
            
            # Choose emoji based on type and priority
            emoji_map = {
                "task_created": "📝",
                "task_assigned": "👤",
                "task_completed": "✅",
                "task_overdue": "🚨",
                "due_reminder": "⏰",
                "status_update": "📊",
                "custom": "🔔"
            }
            
            priority_emoji = {
                "high": "🔴",
                "medium": "🟡",
                "low": "🟢"
            }
            
            # Build formatted message
            emoji = emoji_map.get(notification_type, "🔔")
            p_emoji = priority_emoji.get(priority, "")
            
            formatted_message = f"{emoji} **{title}** {p_emoji}\n\n{message}"
            
            # Add task details if available
            task_details = notification.get("details", {}).get("task", {})
            if task_details and notification_type in ["task_created", "task_assigned", "task_completed"]:
                formatted_message += f"\n\n📋 **Detalles:**"
                if task_details.get("responsable"):
                    formatted_message += f"\n👤 Responsable: {task_details['responsable']}"
                if task_details.get("prioridad"):
                    formatted_message += f"\n⭐ Prioridad: {task_details['prioridad']}"
                if task_details.get("fecha_limite"):
                    formatted_message += f"\n📅 Vence: {task_details['fecha_limite']}"
            
            # Add reminder details
            if notification_type == "due_reminder":
                hours = notification.get("details", {}).get("hours_remaining", 0)
                formatted_message += f"\n\n⏰ **Vence en {hours} horas**"
            
            # Add confidence for task detection
            if notification_type == "task_created":
                confidence = notification.get("details", {}).get("confidence", 0)
                if confidence > 0:
                    formatted_message += f"\n\n🎯 Confianza: {confidence:.1%}"
            
            return formatted_message
            
        except Exception as e:
            self.logger.error("WhatsApp message formatting failed", error=str(e))
            return f"🔔 {notification.get('title', 'Notification')}: {notification.get('message', '')}"
    
    def set_whatsapp_agent(self, whatsapp_agent):
        """Set WhatsApp agent for sending notifications"""
        self.whatsapp_agent = whatsapp_agent
        self.logger.info("WhatsApp agent connected to notification system")
    
    async def _add_to_notification_queue(self, notification: Dict[str, Any]):
        """Add notification to internal queue"""
        try:
            # Add to queue
            self.notification_queue.append(notification)
            
            # Maintain queue size
            if len(self.notification_queue) > self.max_queue_size:
                self.notification_queue = self.notification_queue[-self.max_queue_size:]
            
        except Exception as e:
            self.logger.error("Queue addition failed", error=str(e))
    
    def _get_notification_priority(self, task_info: Dict[str, Any]) -> str:
        """Determine notification priority based on task info"""
        priority = task_info.get("priority", "media").lower()
        
        if priority == "alta" or priority == "urgente":
            return "high"
        elif priority == "media":
            return "medium"
        else:
            return "low"
    
    async def _get_task_details(self, task_id: int) -> Dict[str, Any]:
        """Get task details from database"""
        try:
            with get_db_context() as db:
                task = db.query(Task).filter(Task.id == task_id).first()
                
                if task:
                    return task.to_dict()
                else:
                    return {}
                    
        except Exception as e:
            self.logger.error("Task details retrieval failed", error=str(e))
            return {}
    
    async def get_recent_notifications(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent notifications from queue and cache"""
        try:
            # Get from cache first (more persistent)
            cached_notifications = await cache_manager.lrange("notifications", 0, limit - 1)
            
            # Combine with queue notifications
            all_notifications = cached_notifications + self.notification_queue
            
            # Sort by timestamp and limit
            sorted_notifications = sorted(
                all_notifications,
                key=lambda x: x.get("timestamp", ""),
                reverse=True
            )
            
            return sorted_notifications[:limit]
            
        except Exception as e:
            self.logger.error("Recent notifications retrieval failed", error=str(e))
            return []
    
    async def mark_notification_read(self, notification_id: str) -> bool:
        """Mark notification as read"""
        try:
            # This would update the notification status
            # For now, just log the action
            self.logger.info("Notification marked as read", id=notification_id)
            return True
            
        except Exception as e:
            self.logger.error("Mark notification read failed", error=str(e))
            return False
    
    # Specific notification methods
    async def _notify_task_created(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send task creation notification"""
        return await self._handle_task_creation_notification(task_data)
    
    async def _notify_task_assigned(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send task assignment notification"""
        notification = {
            "type": "task_assigned",
            "title": "Tarea Asignada",
            "message": f"Se te ha asignado una nueva tarea: {task_data.get('descripcion', 'Sin descripción')}",
            "details": task_data,
            "timestamp": datetime.utcnow().isoformat(),
            "priority": self._get_notification_priority(task_data),
        }
        
        return await self._send_notification(notification)
    
    async def _notify_task_completed(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send task completion notification"""
        notification = {
            "type": "task_completed",
            "title": "Tarea Completada",
            "message": f"La tarea '{task_data.get('descripcion', 'Unknown')}' ha sido completada",
            "details": task_data,
            "timestamp": datetime.utcnow().isoformat(),
            "priority": "medium",
        }
        
        return await self._send_notification(notification)
    
    async def _notify_task_overdue(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send overdue task notification"""
        notification = {
            "type": "task_overdue",
            "title": "Tarea Vencida",
            "message": f"La tarea '{task_data.get('descripcion', 'Unknown')}' está vencida",
            "details": task_data,
            "timestamp": datetime.utcnow().isoformat(),
            "priority": "high",
        }
        
        return await self._send_notification(notification)
    
    async def _notify_due_reminder(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send due date reminder notification"""
        hours = task_data.get("hours_remaining", 0)
        
        notification = {
            "type": "due_reminder",
            "title": f"Recordatorio: Tarea vence en {hours} horas",
            "message": f"La tarea '{task_data.get('descripcion', 'Unknown')}' vence pronto",
            "details": task_data,
            "timestamp": datetime.utcnow().isoformat(),
            "priority": "high" if hours <= 2 else "medium",
        }
        
        return await self._send_notification(notification)
    
    async def _notify_status_update(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send status update notification"""
        return await self._handle_task_update_notification(task_data)