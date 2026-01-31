"""
Jobs API Routes
Handles job status queries and results retrieval.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.job import Job
from app.schemas.job import JobResponse, JobStatus

router = APIRouter()


@router.get("/{job_id}", response_model=JobResponse)
async def get_job_status(
    job_id: str,
    db: Session = Depends(get_db),
):
    """Get job status and result."""
    job = db.query(Job).filter(Job.id == job_id).first()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    return job


@router.get("", response_model=List[JobResponse])
async def list_jobs(
    character_id: Optional[str] = None,
    job_status: Optional[JobStatus] = None,
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """List jobs with optional filters."""
    query = db.query(Job)
    
    if character_id:
        query = query.filter(Job.character_id == character_id)
    
    if job_status:
        query = query.filter(Job.status == job_status.value)
    
    jobs = query.order_by(Job.created_at.desc()).offset(offset).limit(limit).all()
    
    return jobs
