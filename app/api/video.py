"""
Video Generation API Routes
Handles video generation requests using Veo 3.0.
"""

import os
import time
import uuid
import requests
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

from app.core.config import settings

router = APIRouter()

PUBLIC_VIDEOS_DIR = Path("uploads/videos")
PUBLIC_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)


class VideoRequest(BaseModel):
    prompt: str


class ImageToVideoRequest(BaseModel):
    prompt: str
    image_url: str


class VideoResponse(BaseModel):
    status: str
    video_path: str


def _get_public_video_url(filename: str) -> str:
    """
    Returns a public URL for the generated video.
    """
    if getattr(settings, "BASE_URL", None):
        return f"{settings.BASE_URL}/videos/{filename}"
    return f"/videos/{filename}"


def _wait_for_operation(client, operation, message: str):
    while not operation.done:
        print(message)
        time.sleep(10)
        operation = client.operations.get(operation)
    return operation

async def generate_video(prompt: str) -> str:
    api_key = settings.GEMINI_API_KEY
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    client = genai.Client(api_key=api_key)

    operation = client.models.generate_videos(
        model="veo-3.0-generate-001",
        prompt=prompt,
        config={
            "aspect_ratio": "16:9",
            "negative_prompt": "cartoon, drawing, low quality",
        },
    )

    operation = _wait_for_operation(
        client,
        operation,
        "Waiting for video generation to complete...",
    )

    generated_video = operation.response.generated_videos[0]

    # Download video bytes from Veo
    print(f"[Video] Downloading generated video...")
    
    video_file = generated_video.video
    
    # Download the file to get the video bytes
    if hasattr(video_file, 'uri'):
        # If it's a URI, download it
        import httpx
        async with httpx.AsyncClient() as http_client:
            response = await http_client.get(video_file.uri, timeout=300.0)
            response.raise_for_status()
            video_bytes = response.content
    elif hasattr(video_file, 'data'):
        video_bytes = video_file.data
    else:
        video_bytes = bytes(video_file)
    
    print(f"[Video] Downloaded {len(video_bytes)} bytes")
    
    # Upload to storage (GCS, S3, or local) using StorageService
    from app.services.storage import StorageService
    storage = StorageService()
    
    filename = f"{uuid.uuid4()}.mp4"
    video_path = f"videos/generated/{filename}"
    
    video_url = await storage.upload_bytes(video_bytes, video_path, content_type="video/mp4")
    print(f"[Video] Uploaded to storage: {video_url}")
    
    return video_url


async def generate_video_from_image(prompt: str, image_url: str) -> str:
    api_key = settings.GEMINI_API_KEY
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    client = genai.Client(api_key=api_key)

    response = requests.get(image_url, timeout=30)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "image/jpeg")
    if "png" in content_type:
        mime_type = "image/png"
    elif "webp" in content_type:
        mime_type = "image/webp"
    else:
        mime_type = "image/jpeg"

    image = types.Image(
        image_bytes=response.content,
        mime_type=mime_type,
    )

    operation = client.models.generate_videos(
        model="veo-3.0-generate-001",
        prompt=prompt,
        image=image,
        config={
            "aspect_ratio": "16:9",
            "negative_prompt": "cartoon, drawing, low quality",
        },
    )

    operation = _wait_for_operation(
        client,
        operation,
        "Waiting for image-to-video generation to complete...",
    )

    generated_video = operation.response.generated_videos[0]

    # Download video bytes from Veo
    print(f"[Video] Downloading generated video...")
    
    video_file = generated_video.video
    
    # Download the file to get the video bytes
    if hasattr(video_file, 'uri'):
        import httpx
        async with httpx.AsyncClient() as http_client:
            response = await http_client.get(video_file.uri, timeout=300.0)
            response.raise_for_status()
            video_bytes = response.content
    elif hasattr(video_file, 'data'):
        video_bytes = video_file.data
    else:
        video_bytes = bytes(video_file)
    
    print(f"[Video] Downloaded {len(video_bytes)} bytes")
    
    # Upload to storage (GCS, S3, or local) using StorageService
    from app.services.storage import StorageService
    storage = StorageService()
    
    filename = f"{uuid.uuid4()}.mp4"
    video_path = f"videos/generated/{filename}"
    
    video_url = await storage.upload_bytes(video_bytes, video_path, content_type="video/mp4")
    print(f"[Video] Uploaded to storage: {video_url}")
    
    return video_url


@router.post("/generate-video", response_model=VideoResponse)
async def generate_video_endpoint(request: VideoRequest):
    try:
        video_url = await generate_video(request.prompt)
        return {
            "status": "success",
            "video_path": video_url,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-video-from-image", response_model=VideoResponse)
async def generate_video_from_image_endpoint(request: ImageToVideoRequest):
    try:
        video_url = await generate_video_from_image(
            prompt=request.prompt,
            image_url=request.image_url,
        )
        return {
            "status": "success",
            "video_path": video_url,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
