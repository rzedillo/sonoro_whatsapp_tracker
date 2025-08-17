"""
Redis client for caching and session management
Enhanced Framework V3.1 Implementation
"""

import redis.asyncio as redis
import json
import structlog
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta

from core.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class RedisClient:
    """Enhanced Redis client with async support"""
    
    def __init__(self):
        self.client: Optional[redis.Redis] = None
        self.connected = False
    
    async def connect(self):
        """Connect to Redis"""
        try:
            self.client = redis.from_url(
                settings.redis_url,
                password=settings.redis_password,
                decode_responses=True,
                health_check_interval=30
            )
            
            # Test connection
            await self.client.ping()
            self.connected = True
            logger.info("Redis connection established")
            
        except Exception as e:
            logger.error("Redis connection failed", error=str(e))
            self.connected = False
            raise
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.client:
            await self.client.close()
            self.connected = False
            logger.info("Redis connection closed")
    
    async def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """Set a value with optional expiration"""
        try:
            if not self.client:
                await self.connect()
            
            # Serialize complex objects
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            result = await self.client.set(key, value, ex=expire)
            return result
            
        except Exception as e:
            logger.error("Redis set failed", key=key, error=str(e))
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value and attempt to deserialize JSON"""
        try:
            if not self.client:
                await self.connect()
            
            value = await self.client.get(key)
            if value is None:
                return None
            
            # Try to deserialize JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
                
        except Exception as e:
            logger.error("Redis get failed", key=key, error=str(e))
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete a key"""
        try:
            if not self.client:
                await self.connect()
            
            result = await self.client.delete(key)
            return bool(result)
            
        except Exception as e:
            logger.error("Redis delete failed", key=key, error=str(e))
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            if not self.client:
                await self.connect()
            
            result = await self.client.exists(key)
            return bool(result)
            
        except Exception as e:
            logger.error("Redis exists failed", key=key, error=str(e))
            return False
    
    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration for a key"""
        try:
            if not self.client:
                await self.connect()
            
            result = await self.client.expire(key, seconds)
            return bool(result)
            
        except Exception as e:
            logger.error("Redis expire failed", key=key, error=str(e))
            return False
    
    async def incr(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment a counter"""
        try:
            if not self.client:
                await self.connect()
            
            result = await self.client.incrby(key, amount)
            return result
            
        except Exception as e:
            logger.error("Redis incr failed", key=key, error=str(e))
            return None
    
    async def lpush(self, key: str, *values) -> Optional[int]:
        """Push values to the left of a list"""
        try:
            if not self.client:
                await self.connect()
            
            # Serialize complex objects
            serialized_values = []
            for value in values:
                if isinstance(value, (dict, list)):
                    serialized_values.append(json.dumps(value))
                else:
                    serialized_values.append(str(value))
            
            result = await self.client.lpush(key, *serialized_values)
            return result
            
        except Exception as e:
            logger.error("Redis lpush failed", key=key, error=str(e))
            return None
    
    async def lrange(self, key: str, start: int = 0, end: int = -1) -> List[Any]:
        """Get range of values from list"""
        try:
            if not self.client:
                await self.connect()
            
            values = await self.client.lrange(key, start, end)
            
            # Try to deserialize JSON values
            result = []
            for value in values:
                try:
                    result.append(json.loads(value))
                except (json.JSONDecodeError, TypeError):
                    result.append(value)
            
            return result
            
        except Exception as e:
            logger.error("Redis lrange failed", key=key, error=str(e))
            return []
    
    async def ltrim(self, key: str, start: int, end: int) -> bool:
        """Trim list to specified range"""
        try:
            if not self.client:
                await self.connect()
            
            result = await self.client.ltrim(key, start, end)
            return bool(result)
            
        except Exception as e:
            logger.error("Redis ltrim failed", key=key, error=str(e))
            return False


class CacheManager:
    """High-level cache management"""
    
    def __init__(self, redis_client: RedisClient):
        self.redis = redis_client
    
    async def cache_task_analysis(self, message_id: str, analysis: Dict[str, Any], ttl: int = 3600):
        """Cache task analysis results"""
        key = f"task_analysis:{message_id}"
        await self.redis.set(key, analysis, expire=ttl)
    
    async def get_task_analysis(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Get cached task analysis"""
        key = f"task_analysis:{message_id}"
        return await self.redis.get(key)
    
    async def cache_user_patterns(self, user_name: str, patterns: Dict[str, Any], ttl: int = 1800):
        """Cache user productivity patterns"""
        key = f"user_patterns:{user_name}"
        await self.redis.set(key, patterns, expire=ttl)
    
    async def get_user_patterns(self, user_name: str) -> Optional[Dict[str, Any]]:
        """Get cached user patterns"""
        key = f"user_patterns:{user_name}"
        return await self.redis.get(key)
    
    async def cache_whatsapp_session(self, session_id: str, session_data: Dict[str, Any], ttl: int = 7200):
        """Cache WhatsApp session data"""
        key = f"whatsapp_session:{session_id}"
        await self.redis.set(key, session_data, expire=ttl)
    
    async def get_whatsapp_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get cached WhatsApp session"""
        key = f"whatsapp_session:{session_id}"
        return await self.redis.get(key)
    
    async def log_agent_activity(self, agent_name: str, activity: Dict[str, Any], max_entries: int = 100):
        """Log agent activity with rotation"""
        key = f"agent_activity:{agent_name}"
        
        # Add timestamp to activity
        activity["timestamp"] = datetime.utcnow().isoformat()
        
        # Push to list and trim to max_entries
        await self.redis.lpush(key, activity)
        await self.redis.ltrim(key, 0, max_entries - 1)
    
    async def get_agent_activity(self, agent_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent agent activity"""
        key = f"agent_activity:{agent_name}"
        return await self.redis.lrange(key, 0, limit - 1)
    
    async def increment_metric(self, metric_key: str, amount: int = 1) -> Optional[int]:
        """Increment a metric counter"""
        key = f"metrics:{metric_key}"
        return await self.redis.incr(key, amount)
    
    async def get_metric(self, metric_key: str) -> Optional[int]:
        """Get metric value"""
        key = f"metrics:{metric_key}"
        result = await self.redis.get(key)
        return int(result) if result is not None else None


# Global instances
redis_client = RedisClient()
cache_manager = CacheManager(redis_client)


async def get_redis_health() -> bool:
    """Check Redis health"""
    try:
        if not redis_client.connected:
            await redis_client.connect()
        
        await redis_client.client.ping()
        return True
        
    except Exception as e:
        logger.error("Redis health check failed", error=str(e))
        return False


async def initialize_redis():
    """Initialize Redis connection"""
    await redis_client.connect()


async def shutdown_redis():
    """Shutdown Redis connection"""
    await redis_client.disconnect()