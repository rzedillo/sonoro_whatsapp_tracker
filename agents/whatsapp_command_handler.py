"""
WhatsApp Command Handler for WhatsApp Task Tracker
Implements the missing command system from v1
"""

import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import structlog

from core.base_agent import BaseAgent
from database.connection import get_db_context
from database.models import Task, TaskStatus, TaskPriority


class WhatsAppCommandHandler:
    """
    Command handler for WhatsApp interactions
    Ported from v1 command system
    """
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.logger = structlog.get_logger(__name__)
        
        # Command mappings from v1
        self.commands = {
            '!help': self._handle_help,
            '!tasks': self._handle_list_tasks,
            '!list': self._handle_list_tasks,
            '!create': self._handle_create_task,
            '!complete': self._handle_complete_task,
            '!status': self._handle_status,
            '!stats': self._handle_stats,
            '!assign': self._handle_assign_task,
            '!delete': self._handle_delete_task,
            '!update': self._handle_update_task,
        }
    
    async def is_command(self, message_text: str) -> bool:
        """Check if message is a command"""
        return message_text.strip().startswith('!')
    
    async def handle_command(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle WhatsApp command - main entry point
        Ported from v1 handleCommand method
        """
        try:
            message_text = message_data.get("text", "").strip()
            author = message_data.get("author", "")
            chat_name = message_data.get("chat_name", "")
            
            self.logger.info("Processing command", command=message_text, author=author, chat=chat_name)
            
            # Parse command and arguments
            parts = message_text.split()
            command = parts[0].lower()
            args = parts[1:] if len(parts) > 1 else []
            
            # Find and execute command handler
            if command in self.commands:
                response = await self.commands[command](args, message_data)
                
                return {
                    "success": True,
                    "command": command,
                    "response": response,
                    "send_reply": True,
                    "reply_text": response["message"]
                }
            else:
                return {
                    "success": False,
                    "command": command,
                    "error": f"Unknown command: {command}",
                    "send_reply": True,
                    "reply_text": f"❌ Unknown command: {command}\nSend !help to see available commands."
                }
                
        except Exception as e:
            self.logger.error("Command handling failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "send_reply": True,
                "reply_text": f"❌ Error processing command: {str(e)}"
            }
    
    async def _handle_help(self, args: List[str], message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle !help command"""
        help_text = """🤖 **WhatsApp Task Tracker - Comandos Disponibles**

📋 **Gestión de Tareas:**
• !tasks - Ver todas las tareas
• !create [descripción] - Crear nueva tarea
• !complete [id] - Marcar tarea como completada
• !assign [id] [usuario] - Asignar tarea a usuario
• !update [id] [campo] [valor] - Actualizar tarea
• !delete [id] - Eliminar tarea

📊 **Información:**
• !status - Estado del sistema
• !stats - Estadísticas de tareas
• !help - Mostrar esta ayuda

💡 **Ejemplos:**
• !create Revisar el reporte semanal
• !complete 5
• !assign 3 @juan
• !tasks pendiente

🔧 **Sistema activo y monitoreando mensajes**"""
        
        return {
            "success": True,
            "message": help_text
        }
    
    async def _handle_list_tasks(self, args: List[str], message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle !tasks and !list commands"""
        try:
            # Build filters from arguments
            filters = {}
            if args:
                if args[0] in ['pendiente', 'completado', 'en_progreso', 'cancelado']:
                    filters['status'] = args[0]
                elif args[0] in ['alta', 'media', 'baja', 'urgente']:
                    filters['priority'] = args[0]
                else:
                    filters['assigned_to'] = args[0]
            
            # Add group filter
            chat_name = message_data.get("chat_name")
            if chat_name:
                filters['group'] = chat_name
            
            # Get tasks from task management agent
            task_agent = self.orchestrator.agents.get("task_manager")
            if not task_agent:
                return {"success": False, "message": "❌ Task management service unavailable"}
            
            result = await task_agent.process({
                "command": "get_tasks",
                "filters": filters,
                "limit": 10
            })
            
            if not result.get("success"):
                return {"success": False, "message": "❌ Error retrieving tasks"}
            
            tasks = result.get("tasks", [])
            
            if not tasks:
                filter_text = f" with filters: {filters}" if filters else ""
                return {
                    "success": True,
                    "message": f"📋 No tasks found{filter_text}"
                }
            
            # Format tasks for display
            task_text = f"📋 **Tasks ({len(tasks)} found):**\n\n"
            
            for task in tasks[:10]:  # Limit to 10 for WhatsApp
                status_emoji = {
                    'pendiente': '⏳',
                    'en_progreso': '🔄',
                    'completado': '✅',
                    'cancelado': '❌'
                }.get(task['estado'], '📋')
                
                priority_emoji = {
                    'alta': '🔴',
                    'urgente': '🚨',
                    'media': '🟡',
                    'baja': '🟢'
                }.get(task['prioridad'], '')
                
                task_text += f"{status_emoji} **#{task['id']}** {task['descripcion'][:50]}...\n"
                task_text += f"   👤 {task.get('responsable', 'Sin asignar')} {priority_emoji}\n"
                if task.get('fecha_limite'):
                    task_text += f"   📅 {task['fecha_limite']}\n"
                task_text += "\n"
            
            if len(tasks) > 10:
                task_text += f"... and {len(tasks) - 10} more tasks\n"
                task_text += "Use !tasks [filter] to narrow results"
            
            return {
                "success": True,
                "message": task_text
            }
            
        except Exception as e:
            self.logger.error("List tasks failed", error=str(e))
            return {"success": False, "message": f"❌ Error listing tasks: {str(e)}"}
    
    async def _handle_create_task(self, args: List[str], message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle !create command"""
        try:
            if not args:
                return {
                    "success": False,
                    "message": "❌ Usage: !create [task description]\nExample: !create Review weekly report"
                }
            
            description = " ".join(args)
            author = message_data.get("author", "")
            chat_name = message_data.get("chat_name", "")
            
            # Create task through task management agent
            task_agent = self.orchestrator.agents.get("task_manager")
            if not task_agent:
                return {"success": False, "message": "❌ Task management service unavailable"}
            
            # Prepare task data similar to v1 format
            task_info = {
                "description": description,
                "assigned_to": author,  # Auto-assign to creator
                "priority": "media",
                "due_date": None,
            }
            
            message_data_for_task = {
                "text": description,
                "author": author,
                "chat_name": chat_name,
                "message_id": f"manual_create_{int(datetime.utcnow().timestamp())}",
            }
            
            result = await task_agent.process({
                "task_detected": True,
                "confidence": 1.0,
                "task_info": task_info,
                "message_data": message_data_for_task,
                "analysis_method": "manual_command"
            })
            
            if result.get("success"):
                task_id = result.get("task_id")
                return {
                    "success": True,
                    "message": f"✅ **Task created successfully!**\n📋 Task #{task_id}: {description}\n👤 Assigned to: {author}"
                }
            else:
                return {
                    "success": False,
                    "message": f"❌ Failed to create task: {result.get('error', 'Unknown error')}"
                }
                
        except Exception as e:
            self.logger.error("Create task failed", error=str(e))
            return {"success": False, "message": f"❌ Error creating task: {str(e)}"}
    
    async def _handle_complete_task(self, args: List[str], message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle !complete command"""
        try:
            if not args:
                return {
                    "success": False,
                    "message": "❌ Usage: !complete [task_id]\nExample: !complete 5"
                }
            
            try:
                task_id = int(args[0])
            except ValueError:
                return {"success": False, "message": "❌ Task ID must be a number"}
            
            author = message_data.get("author", "")
            
            # Complete task through task management agent
            task_agent = self.orchestrator.agents.get("task_manager")
            if not task_agent:
                return {"success": False, "message": "❌ Task management service unavailable"}
            
            result = await task_agent.process({
                "command": "complete_task",
                "task_id": task_id,
                "completed_by": author
            })
            
            if result.get("success"):
                return {
                    "success": True,
                    "message": f"🎉 **Task #{task_id} completed!**\n✅ Marked as completed by {author}"
                }
            else:
                return {
                    "success": False,
                    "message": f"❌ Failed to complete task: {result.get('error', 'Task not found')}"
                }
                
        except Exception as e:
            self.logger.error("Complete task failed", error=str(e))
            return {"success": False, "message": f"❌ Error completing task: {str(e)}"}
    
    async def _handle_assign_task(self, args: List[str], message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle !assign command"""
        try:
            if len(args) < 2:
                return {
                    "success": False,
                    "message": "❌ Usage: !assign [task_id] [user]\nExample: !assign 3 @juan"
                }
            
            try:
                task_id = int(args[0])
            except ValueError:
                return {"success": False, "message": "❌ Task ID must be a number"}
            
            assignee = args[1].replace('@', '')  # Remove @ if present
            author = message_data.get("author", "")
            
            # Update task assignment
            task_agent = self.orchestrator.agents.get("task_manager")
            if not task_agent:
                return {"success": False, "message": "❌ Task management service unavailable"}
            
            result = await task_agent.process({
                "command": "update_task",
                "task_id": task_id,
                "updates": {"responsable": assignee},
                "changed_by": author
            })
            
            if result.get("success"):
                return {
                    "success": True,
                    "message": f"👤 **Task #{task_id} assigned!**\n✅ Now assigned to: {assignee}"
                }
            else:
                return {
                    "success": False,
                    "message": f"❌ Failed to assign task: {result.get('error', 'Task not found')}"
                }
                
        except Exception as e:
            self.logger.error("Assign task failed", error=str(e))
            return {"success": False, "message": f"❌ Error assigning task: {str(e)}"}
    
    async def _handle_status(self, args: List[str], message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle !status command"""
        try:
            # Get system status from orchestrator
            orchestrator_status = self.orchestrator.get_orchestrator_status()
            agent_health = await self.orchestrator.health_check()
            
            status_text = "🤖 **System Status**\n\n"
            status_text += f"📊 **Orchestrator**: {orchestrator_status.get('status', 'unknown')}\n"
            status_text += f"⏰ **Uptime**: {orchestrator_status.get('uptime_seconds', 0):.0f}s\n"
            status_text += f"📝 **Queue Size**: {orchestrator_status.get('queue_size', 0)}\n"
            status_text += f"🔧 **Agents**: {len(orchestrator_status.get('agents', {}))}\n\n"
            
            status_text += "**Agent Status:**\n"
            for agent_name, status in orchestrator_status.get('agents', {}).items():
                emoji = "✅" if status == "idle" else "🔄" if status == "processing" else "❌"
                status_text += f"{emoji} {agent_name.title()}: {status}\n"
            
            # Add WhatsApp specific status
            whatsapp_health = agent_health.get("whatsapp", {})
            if whatsapp_health:
                status_text += f"\n📱 **WhatsApp**: "
                if whatsapp_health.get("healthy"):
                    status_text += "Connected ✅"
                else:
                    status_text += "Disconnected ❌"
            
            return {
                "success": True,
                "message": status_text
            }
            
        except Exception as e:
            self.logger.error("Status check failed", error=str(e))
            return {"success": False, "message": f"❌ Error getting status: {str(e)}"}
    
    async def _handle_stats(self, args: List[str], message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle !stats command"""
        try:
            chat_name = message_data.get("chat_name")
            
            # Get analytics from task management agent
            task_agent = self.orchestrator.agents.get("task_manager")
            if not task_agent:
                return {"success": False, "message": "❌ Task management service unavailable"}
            
            result = await task_agent.process({
                "command": "analyze_productivity",
                "days": 30,
                "group_name": chat_name
            })
            
            if not result.get("success"):
                return {"success": False, "message": "❌ Error getting statistics"}
            
            stats = result
            
            stats_text = f"📊 **Statistics (Last 30 days)**\n"
            if chat_name:
                stats_text += f"📱 **Group**: {chat_name}\n\n"
            
            stats_text += f"📋 **Total Tasks**: {stats.get('total_tasks', 0)}\n"
            stats_text += f"✅ **Completed**: {stats.get('completed_tasks', 0)}\n"
            stats_text += f"⏳ **Pending**: {stats.get('pending_tasks', 0)}\n"
            stats_text += f"📈 **Completion Rate**: {stats.get('completion_rate', 0):.1%}\n\n"
            
            # User breakdown
            user_breakdown = stats.get('user_breakdown', {})
            if user_breakdown:
                stats_text += "👥 **Top Contributors**:\n"
                sorted_users = sorted(user_breakdown.items(), key=lambda x: x[1]['total'], reverse=True)
                for user, user_stats in sorted_users[:5]:
                    completion = user_stats['completed'] / user_stats['total'] if user_stats['total'] > 0 else 0
                    stats_text += f"• {user}: {user_stats['total']} tasks ({completion:.1%} completed)\n"
            
            return {
                "success": True,
                "message": stats_text
            }
            
        except Exception as e:
            self.logger.error("Stats failed", error=str(e))
            return {"success": False, "message": f"❌ Error getting stats: {str(e)}"}
    
    async def _handle_delete_task(self, args: List[str], message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle !delete command"""
        try:
            if not args:
                return {
                    "success": False,
                    "message": "❌ Usage: !delete [task_id]\nExample: !delete 5"
                }
            
            try:
                task_id = int(args[0])
            except ValueError:
                return {"success": False, "message": "❌ Task ID must be a number"}
            
            author = message_data.get("author", "")
            
            # Delete task through task management agent
            task_agent = self.orchestrator.agents.get("task_manager")
            if not task_agent:
                return {"success": False, "message": "❌ Task management service unavailable"}
            
            result = await task_agent.process({
                "command": "delete_task",
                "task_id": task_id,
                "deleted_by": author
            })
            
            if result.get("success"):
                return {
                    "success": True,
                    "message": f"🗑️ **Task #{task_id} deleted!**\n❌ Marked as cancelled by {author}"
                }
            else:
                return {
                    "success": False,
                    "message": f"❌ Failed to delete task: {result.get('error', 'Task not found')}"
                }
                
        except Exception as e:
            self.logger.error("Delete task failed", error=str(e))
            return {"success": False, "message": f"❌ Error deleting task: {str(e)}"}
    
    async def _handle_update_task(self, args: List[str], message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle !update command"""
        try:
            if len(args) < 3:
                return {
                    "success": False,
                    "message": "❌ Usage: !update [task_id] [field] [value]\nExample: !update 3 priority alta"
                }
            
            try:
                task_id = int(args[0])
            except ValueError:
                return {"success": False, "message": "❌ Task ID must be a number"}
            
            field = args[1].lower()
            value = " ".join(args[2:])
            author = message_data.get("author", "")
            
            # Map field names
            field_mapping = {
                'priority': 'prioridad',
                'status': 'estado',
                'responsible': 'responsable',
                'assignee': 'responsable',
                'description': 'descripcion',
                'due': 'fecha_limite',
                'deadline': 'fecha_limite'
            }
            
            db_field = field_mapping.get(field, field)
            
            # Validate field values
            if db_field == 'prioridad' and value not in ['alta', 'media', 'baja', 'urgente']:
                return {"success": False, "message": "❌ Priority must be: alta, media, baja, urgente"}
            
            if db_field == 'estado' and value not in ['pendiente', 'en_progreso', 'completado', 'cancelado']:
                return {"success": False, "message": "❌ Status must be: pendiente, en_progreso, completado, cancelado"}
            
            # Update task
            task_agent = self.orchestrator.agents.get("task_manager")
            if not task_agent:
                return {"success": False, "message": "❌ Task management service unavailable"}
            
            result = await task_agent.process({
                "command": "update_task",
                "task_id": task_id,
                "updates": {db_field: value},
                "changed_by": author
            })
            
            if result.get("success"):
                return {
                    "success": True,
                    "message": f"✏️ **Task #{task_id} updated!**\n📝 {field.title()}: {value}"
                }
            else:
                return {
                    "success": False,
                    "message": f"❌ Failed to update task: {result.get('error', 'Task not found')}"
                }
                
        except Exception as e:
            self.logger.error("Update task failed", error=str(e))
            return {"success": False, "message": f"❌ Error updating task: {str(e)}"}