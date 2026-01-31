"""
LoRA Cloud Training Service
Triggers LoRA training on Vertex AI (GPU-enabled) from Cloud Run.

Architecture:
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   Cloud Run     │ ──▶  │   Vertex AI      │ ──▶  │   GCS Bucket    │
│  (collects      │      │  (trains with    │      │  (stores        │
│   golden imgs)  │      │   GPU)           │      │   LoRA weights) │
└─────────────────┘      └──────────────────┘      └─────────────────┘
"""

import os
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from app.core.config import settings

logger = logging.getLogger(__name__)

# Check if Vertex AI is available
try:
    from google.cloud import aiplatform
    VERTEX_AVAILABLE = True
except ImportError:
    VERTEX_AVAILABLE = False
    logger.warning("Vertex AI SDK not available. Install with: pip install google-cloud-aiplatform")


class LoraCloudTrainer:
    """
    Manages LoRA training jobs on Vertex AI.
    
    This allows Cloud Run to trigger GPU training without having GPU itself.
    """
    
    def __init__(self):
        self.project_id = settings.GCP_PROJECT_ID
        self.region = os.getenv("VERTEX_REGION", "us-central1")
        self.bucket = settings.GCS_BUCKET_OUTPUTS
        self.use_cloud = settings.USE_GCS and VERTEX_AVAILABLE
        
        if self.use_cloud:
            aiplatform.init(
                project=self.project_id,
                location=self.region
            )
            logger.info(f"[LoraCloudTrainer] Initialized for Vertex AI in {self.region}")
        else:
            logger.info("[LoraCloudTrainer] Running in local/mock mode")
    
    def submit_training_job(
        self,
        character_id: str,
        lora_model_id: str,
        dataset_gcs_path: str,
        base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        training_steps: int = 500,
        learning_rate: float = 1e-4,
        rank: int = 16
    ) -> Dict[str, Any]:
        """
        Submit a LoRA training job to Vertex AI.
        
        Args:
            character_id: Character being trained
            lora_model_id: LoRA model ID in database
            dataset_gcs_path: GCS path to training images (gs://bucket/path)
            base_model: Base model to fine-tune
            training_steps: Number of training steps
            learning_rate: Learning rate
            rank: LoRA rank (lower = smaller model)
            
        Returns:
            Dict with job info
        """
        if not self.use_cloud:
            return self._mock_training(character_id, lora_model_id)
        
        job_display_name = f"lora-{character_id}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        output_path = f"gs://{self.bucket}/lora_models/{lora_model_id}"
        
        # Training script arguments
        training_args = [
            f"--character_id={character_id}",
            f"--dataset_path={dataset_gcs_path}",
            f"--output_path={output_path}",
            f"--base_model={base_model}",
            f"--steps={training_steps}",
            f"--lr={learning_rate}",
            f"--rank={rank}",
            f"--lora_model_id={lora_model_id}",
        ]
        
        try:
            # Create custom training job
            job = aiplatform.CustomJob(
                display_name=job_display_name,
                worker_pool_specs=[
                    {
                        "machine_spec": {
                            "machine_type": "n1-standard-8",
                            "accelerator_type": "NVIDIA_TESLA_T4",
                            "accelerator_count": 1,
                        },
                        "replica_count": 1,
                        "container_spec": {
                            "image_uri": f"gcr.io/{self.project_id}/cineai-lora-trainer:latest",
                            "args": training_args,
                        },
                    }
                ],
                staging_bucket=f"gs://{self.bucket}/vertex_staging",
            )
            
            # Submit job (non-blocking)
            job.submit()
            
            logger.info(f"[LoraCloudTrainer] Submitted training job: {job.resource_name}")
            
            return {
                "success": True,
                "job_id": job.resource_name,
                "job_name": job_display_name,
                "status": "submitted",
                "output_path": output_path,
                "message": "Training job submitted to Vertex AI"
            }
            
        except Exception as e:
            logger.error(f"[LoraCloudTrainer] Failed to submit job: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to submit training job"
            }
    
    def get_job_status(self, job_resource_name: str) -> Dict[str, Any]:
        """Get status of a Vertex AI training job."""
        if not self.use_cloud:
            return {"status": "mock_completed", "state": "JOB_STATE_SUCCEEDED"}
        
        try:
            job = aiplatform.CustomJob.get(job_resource_name)
            
            return {
                "job_id": job.resource_name,
                "state": job.state.name,
                "status": self._map_state(job.state.name),
                "create_time": str(job.create_time) if job.create_time else None,
                "end_time": str(job.end_time) if job.end_time else None,
                "error": job.error.message if job.error else None
            }
        except Exception as e:
            logger.error(f"[LoraCloudTrainer] Failed to get job status: {e}")
            return {"error": str(e)}
    
    def _map_state(self, vertex_state: str) -> str:
        """Map Vertex AI state to simple status."""
        state_map = {
            "JOB_STATE_QUEUED": "queued",
            "JOB_STATE_PENDING": "pending",
            "JOB_STATE_RUNNING": "running",
            "JOB_STATE_SUCCEEDED": "completed",
            "JOB_STATE_FAILED": "failed",
            "JOB_STATE_CANCELLED": "cancelled",
        }
        return state_map.get(vertex_state, "unknown")
    
    def _mock_training(self, character_id: str, lora_model_id: str) -> Dict[str, Any]:
        """Mock training for local development."""
        logger.info(f"[LoraCloudTrainer] Mock training for {character_id}")
        return {
            "success": True,
            "job_id": f"mock-job-{lora_model_id}",
            "status": "mock_completed",
            "message": "Mock training (Vertex AI not available). Use local trainer instead."
        }
    
    def check_training_readiness(self, character_id: str) -> Dict[str, Any]:
        """
        Check if character has enough golden images for training.
        
        Returns readiness status and recommendations.
        """
        from app.services.golden_collector import GoldenImageCollector
        from app.core.database import SessionLocal
        
        db = SessionLocal()
        try:
            collector = GoldenImageCollector(db)
            stats = collector.get_collection_stats(character_id)
            
            min_images = 10  # Minimum for decent LoRA
            recommended_images = 25  # Recommended for good quality
            
            count = stats.get("total_images", 0)
            
            return {
                "character_id": character_id,
                "golden_images": count,
                "minimum_required": min_images,
                "recommended": recommended_images,
                "ready_for_training": count >= min_images,
                "quality": "excellent" if count >= recommended_images else "good" if count >= min_images else "insufficient",
                "message": (
                    f"Ready for training with {count} golden images!" if count >= min_images
                    else f"Need {min_images - count} more high-quality generations (IDR > 0.85)"
                )
            }
        finally:
            db.close()


# Convenience function
def trigger_cloud_training(
    character_id: str,
    lora_model_id: str,
    dataset_gcs_path: str,
    **kwargs
) -> Dict[str, Any]:
    """Trigger LoRA training on Vertex AI."""
    trainer = LoraCloudTrainer()
    return trainer.submit_training_job(
        character_id=character_id,
        lora_model_id=lora_model_id,
        dataset_gcs_path=dataset_gcs_path,
        **kwargs
    )
