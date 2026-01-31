"""
Base Worker Classes
Provides base classes for RQ workers with retry logic, error handling, and monitoring.
"""

import logging
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, TypeVar
from datetime import datetime
from functools import wraps
from enum import Enum

from rq import get_current_job
from rq.job import Job

from app.core.config import settings

logger = logging.getLogger(__name__)

T = TypeVar('T')


class JobStatus(str, Enum):
    """Job status enumeration."""
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"


class WorkerException(Exception):
    """Base exception for worker errors."""
    
    def __init__(self, message: str, retryable: bool = True, details: Optional[dict] = None):
        super().__init__(message)
        self.retryable = retryable
        self.details = details or {}


class NonRetryableError(WorkerException):
    """Error that should NOT be retried (e.g., invalid input)."""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, retryable=False, details=details)


class RetryableError(WorkerException):
    """Error that SHOULD be retried (e.g., API timeout)."""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, retryable=True, details=details)


def with_retry(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    exponential_backoff: bool = True,
    retryable_exceptions: tuple = (RetryableError, TimeoutError, ConnectionError)
):
    """
    Decorator to add retry logic to worker tasks.
    
    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Base delay between retries (seconds)
        exponential_backoff: Whether to use exponential backoff
        retryable_exceptions: Tuple of exception types that should trigger retry
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                    
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        delay = retry_delay * (2 ** attempt if exponential_backoff else 1)
                        logger.warning(
                            f"[Retry {attempt + 1}/{max_retries}] {func.__name__} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"[Failed] {func.__name__} exhausted all {max_retries} retries: {e}"
                        )
                        
                except NonRetryableError as e:
                    logger.error(f"[Non-Retryable] {func.__name__}: {e}")
                    raise
                    
                except Exception as e:
                    logger.error(f"[Unexpected] {func.__name__}: {e}\n{traceback.format_exc()}")
                    raise
            
            # All retries exhausted
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        delay = retry_delay * (2 ** attempt if exponential_backoff else 1)
                        logger.warning(
                            f"[Retry {attempt + 1}/{max_retries}] {func.__name__} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"[Failed] {func.__name__} exhausted all {max_retries} retries: {e}"
                        )
                        
                except NonRetryableError as e:
                    logger.error(f"[Non-Retryable] {func.__name__}: {e}")
                    raise
                    
                except Exception as e:
                    logger.error(f"[Unexpected] {func.__name__}: {e}\n{traceback.format_exc()}")
                    raise
            
            raise last_exception
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


class BaseWorker(ABC):
    """
    Abstract base class for RQ workers.
    
    Features:
    - Automatic job progress tracking
    - Structured logging
    - Metrics collection
    - Error handling with context
    """
    
    def __init__(self):
        self.start_time: Optional[datetime] = None
        self.job: Optional[Job] = None
    
    def _get_current_job(self) -> Optional[Job]:
        """Get the current RQ job context."""
        try:
            return get_current_job()
        except Exception:
            return None
    
    def _update_progress(self, progress: float, message: str = ""):
        """
        Update job progress (0.0 to 1.0).
        
        Args:
            progress: Progress value between 0 and 1
            message: Optional status message
        """
        job = self._get_current_job()
        if job:
            job.meta["progress"] = min(max(progress, 0), 1)
            job.meta["progress_message"] = message
            job.meta["updated_at"] = datetime.utcnow().isoformat()
            job.save_meta()
            
        logger.debug(f"Progress: {progress:.0%} - {message}")
    
    def _set_status(self, status: JobStatus, details: Optional[dict] = None):
        """Set job status with optional details."""
        job = self._get_current_job()
        if job:
            job.meta["worker_status"] = status.value
            job.meta["status_details"] = details or {}
            job.meta["updated_at"] = datetime.utcnow().isoformat()
            job.save_meta()
    
    def _log_start(self, task_name: str, **context):
        """Log task start with context."""
        self.start_time = datetime.utcnow()
        self._set_status(JobStatus.RUNNING)
        logger.info(f"[START] {task_name} | Context: {context}")
    
    def _log_complete(self, task_name: str, result_summary: str = ""):
        """Log task completion with timing."""
        duration = (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0
        self._set_status(JobStatus.SUCCESS)
        self._update_progress(1.0, "Complete")
        logger.info(f"[COMPLETE] {task_name} | Duration: {duration:.2f}s | {result_summary}")
    
    def _log_error(self, task_name: str, error: Exception):
        """Log task error with details."""
        duration = (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0
        self._set_status(JobStatus.FAILED, {"error": str(error)})
        logger.error(f"[ERROR] {task_name} | Duration: {duration:.2f}s | Error: {error}")
    
    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """
        Execute the worker task. Must be implemented by subclasses.
        
        Returns:
            Task result
        """
        pass


class ImageGenerationWorker(BaseWorker):
    """Worker for image generation tasks."""
    
    TASK_NAME = "image_generation"
    
    async def execute(
        self,
        job_id: str,
        character_id: str,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute image generation pipeline.
        
        This is called by the RQ task wrapper.
        """
        self._log_start(self.TASK_NAME, job_id=job_id, character_id=character_id)
        
        try:
            self._update_progress(0.1, "Loading character data...")
            # Actual generation logic will be imported from services
            
            self._update_progress(0.3, "Generating prompt...")
            # Groq LLM call
            
            self._update_progress(0.5, "Generating image...")
            # Gemini call
            
            self._update_progress(0.7, "Extracting state...")
            # State extraction
            
            self._update_progress(0.9, "Validating identity...")
            # IDR check
            
            result = {
                "job_id": job_id,
                "status": "success",
                "result_url": None,  # Will be set by actual implementation
            }
            
            self._log_complete(self.TASK_NAME, f"Job {job_id} completed")
            return result
            
        except Exception as e:
            self._log_error(self.TASK_NAME, e)
            raise


class RefinementWorker(BaseWorker):
    """Worker for face refinement tasks."""
    
    TASK_NAME = "face_refinement"
    
    async def execute(
        self,
        job_id: str,
        image_url: str,
        character_id: str,
        current_idr: float,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute face refinement pipeline.
        """
        self._log_start(
            self.TASK_NAME, 
            job_id=job_id, 
            character_id=character_id,
            current_idr=current_idr
        )
        
        try:
            self._update_progress(0.2, "Loading image...")
            
            self._update_progress(0.4, "Detecting face region...")
            
            self._update_progress(0.6, "Refining face...")
            
            self._update_progress(0.8, "Blending result...")
            
            result = {
                "job_id": job_id,
                "refined": True,
                "new_idr": 0.0,  # Will be computed
            }
            
            self._log_complete(self.TASK_NAME, f"Job {job_id} refined")
            return result
            
        except Exception as e:
            self._log_error(self.TASK_NAME, e)
            raise


# Export all
__all__ = [
    "JobStatus",
    "WorkerException",
    "NonRetryableError", 
    "RetryableError",
    "with_retry",
    "BaseWorker",
    "ImageGenerationWorker",
    "RefinementWorker"
]
