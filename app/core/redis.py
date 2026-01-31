"""
Redis Connection Manager
Provides Redis connection pool and utilities for RQ workers.
"""

import logging
from typing import Optional
from redis import Redis, ConnectionPool
from redis.exceptions import ConnectionError, TimeoutError
from functools import lru_cache

from app.core.config import settings

logger = logging.getLogger(__name__)


class RedisManager:
    """
    Manages Redis connections with connection pooling.
    
    Features:
    - Connection pooling for efficient resource usage
    - Health checks
    - Automatic reconnection handling
    - Singleton pattern via get_redis_manager()
    """
    
    _instance: Optional["RedisManager"] = None
    _pool: Optional[ConnectionPool] = None
    
    def __init__(self):
        self._pool = None
        self._client: Optional[Redis] = None
    
    @classmethod
    def get_instance(cls) -> "RedisManager":
        """Get singleton instance of RedisManager."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _create_pool(self) -> ConnectionPool:
        """Create Redis connection pool."""
        return ConnectionPool.from_url(
            settings.REDIS_URL,
            max_connections=10,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            decode_responses=False  # RQ needs bytes
        )
    
    def get_connection(self) -> Redis:
        """
        Get a Redis connection from the pool.
        
        Returns:
            Redis client instance
        """
        if self._pool is None:
            self._pool = self._create_pool()
            logger.info(f"Created Redis connection pool for {settings.REDIS_URL}")
        
        if self._client is None:
            self._client = Redis(connection_pool=self._pool)
        
        return self._client
    
    def health_check(self) -> dict:
        """
        Check Redis connection health.
        
        Returns:
            dict with status and info
        """
        try:
            client = self.get_connection()
            ping_result = client.ping()
            info = client.info("server")
            
            return {
                "status": "healthy" if ping_result else "unhealthy",
                "connected": True,
                "redis_version": info.get("redis_version", "unknown"),
                "url": self._mask_url(settings.REDIS_URL)
            }
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Redis health check failed: {e}")
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e),
                "url": self._mask_url(settings.REDIS_URL)
            }
    
    def _mask_url(self, url: str) -> str:
        """Mask password in Redis URL for logging."""
        if "@" in url:
            # redis://:password@host:port -> redis://***@host:port
            parts = url.split("@")
            return f"redis://***@{parts[-1]}"
        return url
    
    def close(self):
        """Close all connections in the pool."""
        if self._pool:
            self._pool.disconnect()
            self._pool = None
            self._client = None
            logger.info("Redis connection pool closed")


# Convenience functions
@lru_cache()
def get_redis_manager() -> RedisManager:
    """Get the singleton Redis manager instance."""
    return RedisManager.get_instance()


def get_redis() -> Redis:
    """Get a Redis connection (convenience function)."""
    return get_redis_manager().get_connection()


def redis_health_check() -> dict:
    """Check Redis health (convenience function)."""
    return get_redis_manager().health_check()


# Queue names
class Queues:
    """Standard queue names for the application."""
    DEFAULT = "default"
    GENERATION = "generation"
    REFINEMENT = "refinement"
    EXTRACTION = "extraction"
    HIGH_PRIORITY = "high"
    LOW_PRIORITY = "low"


# Export all
__all__ = [
    "RedisManager",
    "get_redis_manager", 
    "get_redis",
    "redis_health_check",
    "Queues"
]
