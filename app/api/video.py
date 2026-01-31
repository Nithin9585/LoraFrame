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

def generate_video(prompt: str) -> str:
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

    filename = f"{uuid.uuid4()}.mp4"
    output_path = PUBLIC_VIDEOS_DIR / filename

    client.files.download(file=generated_video.video)
    generated_video.video.save(str(output_path))

    return _get_public_video_url(filename)


def generate_video_from_image(prompt: str, image_url: str) -> str:
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

    filename = f"{uuid.uuid4()}.mp4"
    output_path = PUBLIC_VIDEOS_DIR / filename

    client.files.download(file=generated_video.video)
    generated_video.video.save(str(output_path))

    return _get_public_video_url(filename)


@router.post("/generate-video", response_model=VideoResponse)
def generate_video_endpoint(request: VideoRequest):
    try:
        video_url = generate_video(request.prompt)
        return {
            "status": "success",
            "video_path": video_url,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-video-from-image", response_model=VideoResponse)
def generate_video_from_image_endpoint(request: ImageToVideoRequest):
    try:
        video_url = generate_video_from_image(
            prompt=request.prompt,
            image_url=request.image_url,
        )
        return {
            "status": "success",
            "video_path": video_url,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
