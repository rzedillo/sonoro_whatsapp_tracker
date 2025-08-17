"""
Message Analysis Agent for WhatsApp Task Tracker
Enhanced Framework V3.1 Implementation
Ported from Claude AI integration logic
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import re

from anthropic import Anthropic

from core.base_agent import BaseAgent
from core.redis_client import cache_manager
from database.connection import get_db_context
from database.models import Task, Conversation, UserPattern


class MessageAnalysisAgent(BaseAgent):
    """
    Message analysis agent using Claude AI for task detection
    
    Features:
    - Natural language task detection
    - Context-aware analysis with user patterns
    - Confidence scoring
    - Duplicate detection
    - Priority and assignment suggestions
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        
        # Configuration
        self.anthropic_api_key = config.get("anthropic_api_key")
        self.analysis_timeout = config.get("analysis_timeout", 30)
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        
        # AI client
        self.anthropic_client = None
        if self.anthropic_api_key:
            self.anthropic_client = Anthropic(api_key=self.anthropic_api_key)
        
        # Analysis patterns
        self.task_indicators = [
            r'\b(necesit[ao]|requiere|debe|hay que|tiene que)\b',
            r'\b(hacer|realizar|completar|entregar|enviar)\b',
            r'\b(pendiente|urgente|importante)\b',
            r'\b(deadline|fecha|plazo|entregar)\b',
            r'\b(asignar|responsable|encargado)\b',
        ]
        
        # Context cache
        self.context_cache = {}
        self.context_cache_duration = 3600  # 1 hour
    
    async def _initialize_agent(self):
        """Initialize message analysis agent"""
        self.logger.info("Initializing message analysis agent")
        
        if not self.anthropic_client:
            self.logger.warning("Anthropic API key not provided - using pattern-based analysis only")
        
        # Compile regex patterns
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.task_indicators]
        
        self.logger.info("Message analysis agent initialized")
    
    async def _cleanup_agent(self):
        """Cleanup message analysis agent"""
        self.logger.info("Message analysis agent cleanup completed")
    
    async def _process_data(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process message for task detection"""
        try:
            message_data = data
            
            # Quick pattern check first
            quick_check = await self._quick_task_check(message_data)
            if not quick_check["likely_task"]:
                return {
                    "success": True,
                    "task_detected": False,
                    "reason": "No task indicators found",
                    "confidence": quick_check["confidence"]
                }
            
            # Full AI analysis
            if self.anthropic_client:
                analysis_result = await self._analyze_with_ai(message_data)
            else:
                analysis_result = await self._analyze_with_patterns(message_data)
            
            # Cache analysis result
            if message_data.get("message_id"):
                await cache_manager.cache_task_analysis(
                    message_data["message_id"],
                    analysis_result
                )
            
            return analysis_result
            
        except Exception as e:
            self.logger.error("Message analysis failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "task_detected": False
            }
    
    async def _health_check(self) -> Dict[str, Any]:
        """Message analysis agent health check"""
        return {
            "anthropic_available": self.anthropic_client is not None,
            "patterns_loaded": len(self.compiled_patterns),
            "context_cache_size": len(self.context_cache),
        }
    
    async def _quick_task_check(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Quick pattern-based task detection"""
        text = message_data.get("text", "").lower()
        
        # Check for task indicators
        indicator_matches = 0
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                indicator_matches += 1
        
        # Calculate likelihood
        likelihood = min(1.0, indicator_matches / 3)  # 3+ indicators = high likelihood
        
        return {
            "likely_task": likelihood > 0.3,
            "confidence": likelihood,
            "indicator_matches": indicator_matches
        }
    
    async def _analyze_with_ai(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze message using Claude AI"""
        try:
            # Get context for better analysis
            context = await self._get_analysis_context(message_data)
            
            # Prepare prompt
            prompt = await self._build_analysis_prompt(message_data, context)
            
            # Call Claude API
            response = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                temperature=0.1,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # Parse response
            analysis_result = await self._parse_ai_response(response.content[0].text, message_data)
            
            return analysis_result
            
        except Exception as e:
            self.logger.error("AI analysis failed", error=str(e))
            # Fallback to pattern analysis
            return await self._analyze_with_patterns(message_data)
    
    async def _analyze_with_patterns(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback pattern-based analysis"""
        text = message_data.get("text", "")
        author = message_data.get("author", "")
        chat_name = message_data.get("chat_name", "")
        
        # Quick check result
        quick_check = await self._quick_task_check(message_data)
        
        if not quick_check["likely_task"]:
            return {
                "success": True,
                "task_detected": False,
                "confidence": quick_check["confidence"],
                "analysis_method": "pattern_based"
            }
        
        # Extract basic task information
        task_info = {
            "description": self._extract_description(text),
            "assigned_to": self._extract_assignee(text, author),
            "priority": self._extract_priority(text),
            "due_date": self._extract_due_date(text),
        }
        
        return {
            "success": True,
            "task_detected": True,
            "confidence": quick_check["confidence"],
            "task_info": task_info,
            "message_data": message_data,
            "analysis_method": "pattern_based",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _extract_description(self, text: str) -> str:
        """Extract task description from text"""
        # Clean up text
        cleaned = re.sub(r'[^\w\s]', ' ', text)
        words = cleaned.split()
        
        # Take first reasonable chunk as description
        if len(words) > 10:
            return ' '.join(words[:10]) + "..."
        else:
            return ' '.join(words)
    
    def _extract_assignee(self, text: str, default_author: str) -> Optional[str]:
        """Extract task assignee from text"""
        # Look for assignment patterns
        assignment_patterns = [
            r'@(\w+)',  # @username
            r'para (\w+)',  # para usuario
            r'(?:asignado a|assign to|for) (\w+)',
        ]
        
        for pattern in assignment_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Default to message author
        return default_author
    
    def _extract_priority(self, text: str) -> str:
        """Extract task priority from text"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['urgente', 'urgent', 'asap', 'ya']):
            return "alta"
        elif any(word in text_lower for word in ['importante', 'important', 'priority']):
            return "media"
        else:
            return "baja"
    
    def _extract_due_date(self, text: str) -> Optional[str]:
        """Extract due date from text"""
        # Look for date patterns
        date_patterns = [
            r'(?:para|by|antes del?) (\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(?:para|by) (hoy|today|mañana|tomorrow)',
            r'(?:para|by) (?:el )?(\w+ \d{1,2})',  # "para el lunes 15"
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    async def _get_analysis_context(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get contextual information for better analysis"""
        try:
            chat_name = message_data.get("chat_name", "")
            author = message_data.get("author", "")
            
            # Check cache first
            cache_key = f"context:{chat_name}:{author}"
            cached_context = self.context_cache.get(cache_key)
            
            if cached_context and cached_context["timestamp"] > datetime.utcnow() - timedelta(seconds=self.context_cache_duration):
                return cached_context["data"]
            
            context = {
                "recent_tasks": await self._get_recent_tasks(chat_name),
                "user_patterns": await self._get_user_patterns(author, chat_name),
                "chat_history": await self._get_recent_messages(chat_name),
            }
            
            # Cache context
            self.context_cache[cache_key] = {
                "data": context,
                "timestamp": datetime.utcnow()
            }
            
            return context
            
        except Exception as e:
            self.logger.error("Context retrieval failed", error=str(e))
            return {}
    
    async def _get_recent_tasks(self, chat_name: str) -> List[Dict[str, Any]]:
        """Get recent tasks from the same chat"""
        try:
            with get_db_context() as db:
                recent_cutoff = datetime.utcnow() - timedelta(days=7)
                
                tasks = db.query(Task).filter(
                    Task.grupo_nome == chat_name,
                    Task.timestamp >= recent_cutoff
                ).order_by(Task.timestamp.desc()).limit(5).all()
                
                return [task.to_dict() for task in tasks]
                
        except Exception as e:
            self.logger.error("Recent tasks retrieval failed", error=str(e))
            return []
    
    async def _get_user_patterns(self, author: str, chat_name: str) -> Dict[str, Any]:
        """Get user productivity patterns"""
        try:
            # Get from cache first
            cached_patterns = await cache_manager.get_user_patterns(f"{author}:{chat_name}")
            
            if cached_patterns:
                return cached_patterns
            
            # Calculate patterns
            with get_db_context() as db:
                user_tasks = db.query(Task).filter(
                    Task.responsable == author,
                    Task.grupo_nome == chat_name
                ).all()
                
                if not user_tasks:
                    return {}
                
                total_tasks = len(user_tasks)
                completed_tasks = len([t for t in user_tasks if t.estado == "completado"])
                
                patterns = {
                    "total_tasks": total_tasks,
                    "completion_rate": completed_tasks / total_tasks if total_tasks > 0 else 0,
                    "most_common_priority": "media",  # Simplified
                }
                
                # Cache patterns
                await cache_manager.cache_user_patterns(f"{author}:{chat_name}", patterns)
                
                return patterns
                
        except Exception as e:
            self.logger.error("User patterns retrieval failed", error=str(e))
            return {}
    
    async def _get_recent_messages(self, chat_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent messages from the same chat"""
        try:
            with get_db_context() as db:
                recent_cutoff = datetime.utcnow() - timedelta(hours=24)
                
                messages = db.query(Conversation).filter(
                    Conversation.grupo_nome == chat_name,
                    Conversation.timestamp >= recent_cutoff
                ).order_by(Conversation.timestamp.desc()).limit(limit).all()
                
                return [msg.to_dict() for msg in messages]
                
        except Exception as e:
            self.logger.error("Recent messages retrieval failed", error=str(e))
            return []
    
    async def _build_analysis_prompt(self, message_data: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Build analysis prompt for Claude AI"""
        message_text = message_data.get("text", "")
        author = message_data.get("author", "")
        chat_name = message_data.get("chat_name", "")
        
        prompt = f"""
Analiza el siguiente mensaje de WhatsApp para determinar si contiene una tarea o asignación:

MENSAJE: "{message_text}"
AUTOR: {author}
GRUPO: {chat_name}

CONTEXTO:
- Tareas recientes en este grupo: {len(context.get('recent_tasks', []))}
- Patrones del usuario: {context.get('user_patterns', {})}

INSTRUCCIONES:
1. Determina si el mensaje contiene una tarea real (no solo comentarios o preguntas)
2. Si es una tarea, extrae:
   - Descripción clara de la tarea
   - Persona asignada (si se menciona)
   - Prioridad (baja/media/alta)
   - Fecha límite (si se menciona)
   - Nivel de confianza (0.0-1.0)

RESPONDE EN FORMATO JSON:
{{
    "es_tarea": true/false,
    "confianza": 0.0-1.0,
    "descripcion": "descripción de la tarea",
    "asignado_a": "nombre de persona o null",
    "prioridad": "baja/media/alta",
    "fecha_limite": "fecha o null",
    "razon": "explicación breve"
}}
"""
        
        return prompt
    
    async def _parse_ai_response(self, response_text: str, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Claude AI response"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")
            
            response_json = json.loads(json_match.group())
            
            # Convert to standard format
            task_detected = response_json.get("es_tarea", False)
            confidence = response_json.get("confianza", 0.0)
            
            if task_detected and confidence >= self.confidence_threshold:
                task_info = {
                    "description": response_json.get("descripcion", ""),
                    "assigned_to": response_json.get("asignado_a"),
                    "priority": response_json.get("prioridad", "media"),
                    "due_date": response_json.get("fecha_limite"),
                }
                
                return {
                    "success": True,
                    "task_detected": True,
                    "confidence": confidence,
                    "task_info": task_info,
                    "message_data": message_data,
                    "analysis_method": "ai_claude",
                    "reason": response_json.get("razon", ""),
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                return {
                    "success": True,
                    "task_detected": False,
                    "confidence": confidence,
                    "reason": response_json.get("razon", "Confidence below threshold"),
                    "analysis_method": "ai_claude",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            self.logger.error("AI response parsing failed", error=str(e))
            # Fallback to pattern analysis
            return await self._analyze_with_patterns(message_data)