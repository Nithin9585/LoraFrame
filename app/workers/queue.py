"""
Queue Management Utilities
Provides RQ queue wrappers for job enqueueing and management.
"""

import logging
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum

from rq import Queue, Retry
from rq.job import Job, JobStatus as RQJobStatus

from app.core.redis import get_redis, Queues
from app.core.config import settings

logger = logging.getLogger(__name__)


class JobPriority(str, Enum):
    """Job priority levels."""
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class QueueManager:
    """
    Manages RQ queues for different job types.
    
    Features:
    - Named queues for different job types
    - Priority-based enqueueing
    - Job status tracking
    - Retry configuration
    """
    
    def __init__(self):
        self._queues: Dict[str, Queue] = {}
        self._redis = None
    
    @property
    def redis(self):
        """Lazy Redis connection."""
        if self._redis is None:
            self._redis = get_redis()
        return self._redis
    
    def get_queue(self, queue_name: str = Queues.DEFAULT) -> Queue:
        """
        Get or create a queue by name.
        
        Args:
            queue_name: Name of the queue
            
        Returns:
            RQ Queue instance
        """
        if queue_name not in self._queues:
            self._queues[queue_name] = Queue(
                name=queue_name,
                connection=self.redis,
                default_timeout=settings.JOB_TIMEOUT_GENERATION
            )
            logger.debug(f"Created queue: {queue_name}")
        
        return self._queues[queue_name]
    
    def enqueue_generation(
        self,
        job_id: str,
        character_id: str,
        prompt: str,
        priority: JobPriority = JobPriority.NORMAL,
        **kwargs
    ) -> Job:
        """
        Enqueue an image generation job.
        
        Args:
            job_id: Unique job identifier
            character_id: Character to generate for
            prompt: User prompt for generation
            priority: Job priority level
            **kwargs: Additional generation parameters
            
        Returns:
            RQ Job instance
        """
        from app.workers.tasks import run_image_generation_task
        
        queue = self._get_priority_queue(priority, Queues.GENERATION)
        
        job = queue.enqueue(
            run_image_generation_task,
            job_id=job_id,
            character_id=character_id,
            prompt=prompt,
            **kwargs,
            job_timeout=settings.JOB_TIMEOUT_GENERATION,
            retry=Retry(max=2, interval=[10, 30]),
            meta={
                "type": "image_generation",
                "character_id": character_id,
                "created_at": datetime.utcnow().isoformat(),
                "priority": priority.value
            }
        )
        
        logger.info(f"Enqueued generation job: {job_id} (priority: {priority.value})")
        return job
    
    def enqueue_refinement(
        self,
        job_id: str,
        image_url: str,
        character_id: str,
        current_idr: float,
        character_data: dict,
        priority: JobPriority = JobPriority.HIGH,
        **kwargs
    ) -> Job:
        """
        Enqueue a face refinement job.
        
        Args:
            job_id: Original generation job ID
            image_url: URL of image to refine
            character_id: Character ID
            current_idr: Current IDR score
            character_data: Character metadata
            priority: Job priority (default HIGH for refinement)
            
        Returns:
            RQ Job instance
        """
        from app.workers.tasks import run_refinement_task
        
        queue = self._get_priority_queue(priority, Queues.REFINEMENT)
        
        job = queue.enqueue(
            run_refinement_task,
            job_id=job_id,
            image_url=image_url,
            character_id=character_id,
            current_idr=current_idr,
            character_data=character_data,
            **kwargs,
            job_timeout=settings.JOB_TIMEOUT_REFINE,
            retry=Retry(max=1, interval=[5]),
            meta={
                "type": "refinement",
                "character_id": character_id,
                "original_idr": current_idr,
                "created_at": datetime.utcnow().isoformat()
            }
        )
        
        logger.info(f"Enqueued refinement job: {job_id} (IDR: {current_idr:.3f})")
        return job
    
    def enqueue_extraction(
        self,
        job_id: str,
        image_url: str,
        character_id: str,
        scene_index: int,
        prompt_used: str = "",
        **kwargs
    ) -> Job:
        """
        Enqueue a state extraction job.
        
        Args:
            job_id: Generation job ID
            image_url: URL of generated image
            character_id: Character ID
            scene_index: Scene number in narrative
            prompt_used: The prompt that generated the image
            
        Returns:
            RQ Job instance
        """
        from app.workers.tasks import run_state_extraction_task
        
        queue = self.get_queue(Queues.EXTRACTION)
        
        job = queue.enqueue(
            run_state_extraction_task,
            job_id=job_id,
            image_url=image_url,
            character_id=character_id,
            scene_index=scene_index,
            prompt_used=prompt_used,
            **kwargs,
            job_timeout=settings.JOB_TIMEOUT_EXTRACTION,
            meta={
                "type": "extraction",
                "character_id": character_id,
                "scene_index": scene_index,
                "created_at": datetime.utcnow().isoformat()
            }
        )
        
        logger.info(f"Enqueued extraction job: {job_id} (scene: {scene_index})")
        return job
    
    def _get_priority_queue(self, priority: JobPriority, default_queue: str) -> Queue:
        """Get queue based on priority."""
        if priority == JobPriority.HIGH:
            return self.get_queue(Queues.HIGH_PRIORITY)
        elif priority == JobPriority.LOW:
            return self.get_queue(Queues.LOW_PRIORITY)
        return self.get_queue(default_queue)
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """
        Get a job by ID from any queue.
        
        Args:
            job_id: The RQ job ID (e.g., "gen_job_xxx")
            
        Returns:
            Job instance or None
        """
        try:
            return Job.fetch(job_id, connection=self.redis)
        except Exception as e:
            logger.debug(f"Job not found: {job_id} - {e}")
            return None
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get comprehensive job status.
        
        Args:
            job_id: The RQ job ID
            
        Returns:
            Dict with status information
        """
        job = self.get_job(job_id)
        
        if not job:
            return {
                "found": False,
                "status": "unknown",
                "message": f"Job {job_id} not found"
            }
        
        status = {
            "found": True,
            "job_id": job.id,
            "status": job.get_status(),
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "ended_at": job.ended_at.isoformat() if job.ended_at else None,
            "meta": job.meta or {}
        }
        
        # Add result if completed
        if job.get_status() == RQJobStatus.FINISHED:
            status["result"] = job.result
        
        # Add error if failed
        if job.get_status() == RQJobStatus.FAILED:
            status["error"] = job.exc_info
        
        return status
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pending job.
        
        Args:
            job_id: The RQ job ID
            
        Returns:
            True if cancelled, False if not found or already running
        """
        job = self.get_job(job_id)
        
        if not job:
            return False
        
        status = job.get_status()
        if status in [RQJobStatus.QUEUED, RQJobStatus.DEFERRED]:
            job.cancel()
            logger.info(f"Cancelled job: {job_id}")
            return True
        
        logger.warning(f"Cannot cancel job {job_id} with status: {status}")
        return False
    
    def get_queue_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get statistics for all queues.
        
        Returns:
            Dict mapping queue names to their stats
        """
        stats = {}
        
        queue_names = [
            Queues.DEFAULT,
            Queues.GENERATION,
            Queues.REFINEMENT,
            Queues.EXTRACTION,
            Queues.HIGH_PRIORITY,
            Queues.LOW_PRIORITY
        ]
        
        for name in queue_names:
            try:
                queue = self.get_queue(name)
                stats[name] = {
                    "queued": len(queue),
                    "started": queue.started_job_registry.count,
                    "finished": queue.finished_job_registry.count,
                    "failed": queue.failed_job_registry.count,
                    "deferred": queue.deferred_job_registry.count
                }
            except Exception as e:
                stats[name] = {"error": str(e)}
        
        return stats


# Singleton instance
_queue_manager: Optional[QueueManager] = None


def get_queue_manager() -> QueueManager:
    """Get singleton QueueManager instance."""
    global _queue_manager
    if _queue_manager is None:
        _queue_manager = QueueManager()
    return _queue_manager


# Convenience functions
def enqueue_generation(job_id: str, character_id: str, prompt: str, **kwargs) -> Job:
    """Enqueue an image generation job (convenience function)."""
    return get_queue_manager().enqueue_generation(job_id, character_id, prompt, **kwargs)


def enqueue_refinement(job_id: str, image_url: str, character_id: str, current_idr: float, character_data: dict, **kwargs) -> Job:
    """Enqueue a face refinement job (convenience function)."""
    return get_queue_manager().enqueue_refinement(job_id, image_url, character_id, current_idr, character_data, **kwargs)


def enqueue_extraction(job_id: str, image_url: str, character_id: str, scene_index: int, **kwargs) -> Job:
    """Enqueue a state extraction job (convenience function)."""
    return get_queue_manager().enqueue_extraction(job_id, image_url, character_id, scene_index, **kwargs)


def get_job_status(job_id: str) -> Dict[str, Any]:
    """Get job status (convenience function)."""
    return get_queue_manager().get_job_status(job_id)


# Export all
__all__ = [
    "JobPriority",
    "QueueManager",
    "get_queue_manager",
    "enqueue_generation",
    "enqueue_refinement",
    "enqueue_extraction",
    "get_job_status"
]
