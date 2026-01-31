"""
IDLock API - Persistent Character Memory System
FastAPI Backend Entry Point
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import init_db
from app.api import characters, generate, jobs, video


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup: Create database tables
    print("Starting IDLock API...")
    init_db()
    print("Database tables created")
    yield
    # Shutdown
    print("Shutting down IDLock API...")


app = FastAPI(
    title="IDLock API",
    description="Persistent Character Memory & Consistency System with AI Video Generation",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(characters.router, prefix="/api/v1/characters", tags=["Characters"])
app.include_router(generate.router, prefix="/api/v1", tags=["Image Generation"])
app.include_router(video.router, prefix="/api/v1", tags=["Video Generation"])
app.include_router(jobs.router, prefix="/api/v1/jobs", tags=["Jobs"])


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint for Cloud Run and monitoring.
    Returns detailed status of critical services.
    """
    import os
    from app.core.config import settings
    
    status = {
        "status": "healthy",
        "version": "0.2.0",
        "environment": {
            "storage": "gcs" if settings.USE_GCS else "local",
            "database": "cloud_sql" if settings.CLOUD_SQL_CONNECTION_NAME else "sqlite",
        },
        "services": {}
    }
    
    # Check database connection
    try:
        from app.core.database import SessionLocal
        from sqlalchemy import text
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        status["services"]["database"] = "ok"
    except Exception as e:
        status["services"]["database"] = f"error: {str(e)}"
        status["status"] = "degraded"
    
    # Check Redis connection
    try:
        from app.core.redis import redis_health_check
        redis_status = redis_health_check()
        if redis_status.get("connected"):
            status["services"]["redis"] = "ok"
            status["services"]["redis_version"] = redis_status.get("redis_version")
        else:
            status["services"]["redis"] = f"error: {redis_status.get('error', 'not connected')}"
            status["status"] = "degraded"
    except Exception as e:
        status["services"]["redis"] = f"error: {str(e)}"
        status["status"] = "degraded"
    
    # Check storage availability
    try:
        if settings.USE_GCS:
            from google.cloud import storage
            client = storage.Client(project=settings.GCP_PROJECT_ID)
            bucket = client.bucket(settings.GCS_BUCKET_UPLOADS)
            bucket.exists()
            status["services"]["storage"] = "ok"
        else:
            import os
            os.path.exists(settings.LOCAL_STORAGE_PATH)
            status["services"]["storage"] = "ok"
    except Exception as e:
        status["services"]["storage"] = f"error: {str(e)}"
        status["status"] = "degraded"
    
    return status


@app.get("/files/{file_path:path}", tags=["Files"])
async def serve_file(file_path: str):
    """
    Serve uploaded files (images, videos) from storage.
    This proxies files from GCS/S3/local storage to the frontend.
    """
    from fastapi.responses import StreamingResponse
    from app.services.storage import StorageService
    import io
    
    try:
        storage = StorageService()
        file_bytes = await storage.get_file(file_path)
        
        # Determine content type based on file extension
        content_type = "image/jpeg"
        if file_path.endswith(".png"):
            content_type = "image/png"
        elif file_path.endswith(".mp4"):
            content_type = "video/mp4"
        elif file_path.endswith(".gif"):
            content_type = "image/gif"
        elif file_path.endswith(".webp"):
            content_type = "image/webp"
        
        return StreamingResponse(
            io.BytesIO(file_bytes),
            media_type=content_type,
            headers={
                "Cache-Control": "public, max-age=3600",
                "Access-Control-Allow-Origin": "*"
            }
        )
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "IDLock API - Persistent Character Memory System",
        "docs": "/docs",
        "health": "/health",
    }
