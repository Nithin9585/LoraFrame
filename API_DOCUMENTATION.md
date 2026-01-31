# CineAI API - Quick Reference

**Base URL:** `http://localhost:8000/api`

## Quick Links
- [Characters](#characters) - Create, list, get, update, delete
- [Generate](#generate) - Image and video generation
- [Jobs](#jobs) - Check generation status
- [Validation](#validation) - Memory health check

---

## Characters

### `POST /characters` - Create Character
**Content-Type:** `multipart/form-data`

**Send:**
```
name: string (required)
description: string (optional)
consent: true (required)
files: File[] (1-5 images)
```

**Receive:**
```json
{
  "id": "char_a1b2c3d4",
  "name": "Alex Morgan",
  "semantic_vector_id": "sem_char_a1b2c3d4",
  "base_image_url": "uploads/characters/char_a1b2c3d4/ref_0.jpg",
  "char_metadata": {
    "face": "Oval face, defined jawline...",
    "hair": "Dark brown, wavy...",
    "eyes": "Deep brown...",
    "reference_images": ["url1", "url2"]
  }
}
```

---

### `GET /characters` - List All Characters

**Send:** Query params `?limit=20&offset=0`

**Receive:** Array of character objects

---

### `GET /characters/{id}` - Get Character

**Send:** Character ID in URL

**Receive:** Single character object

---

### `PUT /characters/{id}` - Update Character

**Send:**
```json
{
  "name": "New Name",
  "description": "New description"
}
```

**Receive:** Updated character object

---

### `DELETE /characters/{id}` - Delete Character

**Send:** Character ID in URL

**Receive:** 204 No Content

---

### `GET /characters/{id}/history` - Get Scene History

**Send:** Character ID in URL

**Receive:**
```json
{
  "character": {...},
  "episodic_states": [
    {
      "scene_index": 1,
      "tags": ["tag1", "tag2"],
      "image_url": "..."
    }
  ],
  "total_scenes": 5
}
```

---

### `GET /characters/{id}/memory-status` - Check Memory Health

**Send:** Character ID in URL

**Receive:**
```json
{
  "semantic_memory": {"status": "OK", "has_vector": true},
  "character_metadata": {"status": "OK"},
  "health_score": 100,
  "health_status": "HEALTHY"
}
```

---

### `POST /characters/{id}/reextract-identity` - Fix Memory

**Send:** Character ID in URL

**Receive:** Updated character with new semantic vector

---

## Generate

### `POST /generate` - Generate Image

**Send:**
```json
{
  "character_id": "char_a1b2c3d4",
  "prompt": "Standing at sunset looking determined",
  "options": {
    "aspect_ratio": "16:9",
    "refine_face": true
  }
}
```

**Aspect Ratios:** `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9`

**Receive:**
```json
{
  "job_id": "job_1a2b3c4d",
  "status": "success",
  "result_url": "uploads/outputs/jobs/job_1a2b3c4d/result.jpg",
  "scene_index": 1,
  "idr_score": 0.782,
  "message": "Generation completed in 8.3s"
}
```

---

### `POST /video/generate` - Generate Video

**Send:**
```json
{
  "character_id": "char_a1b2c3d4",
  "prompt": "Walking into warehouse nervously",
  "options": {
    "aspect_ratio": "16:9",
    "resolution": "1080p",
    "duration_seconds": 8,
    "dialogue": [
      {
        "speaker": "Alex",
        "line": "Anyone here?",
        "emotion": "nervously"
      }
    ]
  }
}
```

**Video Options:**
- `aspect_ratio`: `16:9` or `9:16`
- `resolution`: `720p`, `1080p`, `4k`
- `duration_seconds`: `4`, `5`, `6`, `8`
- `person_generation`: `allow_all`, `allow_adult`, `dont_allow`

**Receive:**
```json
{
  "job_id": "video_abc123",
  "status": "success",
  "result_url": "uploads/outputs/videos/video_abc123/result.mp4",
  "video_duration_seconds": 8,
  "has_audio": true
}
```

---

### `POST /video/extend` - Extend Video

**Send:**
```json
{
  "job_id": "video_abc123",
  "continuation_prompt": "Spins around quickly",
  "duration_seconds": 8
}
```

**Receive:** Same as video/generate

---

### `POST /video/transition` - Frame-to-Frame Video

**Send:**
```json
{
  "character_id": "char_a1b2c3d4",
  "transition_prompt": "Turning from left to right",
  "first_frame_prompt": "Looking left with concern",
  "last_frame_prompt": "Looking right with determination",
  "options": {
    "duration_seconds": 5
  }
}
```

**Receive:** Same as video/generate

---

## Jobs

### `GET /jobs/{job_id}` - Get Job Status

**Send:** Job ID in URL

**Receive:**
```json
{
  "id": "job_1a2b3c4d",
  "status": "success",
  "result_url": "uploads/outputs/jobs/job_1a2b3c4d/result.jpg",
  "metrics": {
    "generation_time_seconds": 8.3,
    "scene_index": 1,
    "idr_score": 0.782
  }
}
```

---

### `GET /jobs` - List All Jobs

**Send:** Query params `?character_id=char_xxx&status=success&limit=20`

**Receive:** Array of job objects

---

## Data Types

### Character Object
```typescript
{
  id: string;              // "char_a1b2c3d4"
  name: string;
  description: string | null;
  semantic_vector_id: string | null;
  base_image_url: string;
  char_metadata: {
    reference_images: string[];
    face: string;
    hair: string;
    eyes: string;
    // ... more traits
  };
}
```

### Job Object
```typescript
{
  id: string;              // "job_xxx" or "video_xxx"
  character_id: string;
  prompt: string;
  status: "queued" | "running" | "success" | "failed";
  result_url: string | null;
  metrics: {
    generation_time_seconds: number;
    scene_index: number;
    idr_score: number;
  };
}
```

---

## Error Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 201 | Created |
| 204 | Deleted |
| 400 | Bad Request |
| 404 | Not Found |
| 500 | Server Error |

**Error Format:**
```json
{
  "detail": "Error message"
}
```

---

## Quick Start

```javascript
// 1. Create character
const formData = new FormData();
formData.append('name', 'Alex');
formData.append('consent', 'true');
formData.append('files', imageFile);

const char = await fetch('/api/characters', {
  method: 'POST',
  body: formData
}).then(r => r.json());

// 2. Generate image
const job = await fetch('/api/generate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    character_id: char.id,
    prompt: "Standing at sunset",
    options: { aspect_ratio: "16:9" }
  })
}).then(r => r.json());

// 3. Use result
console.log(job.result_url);

// 4. Generate video
const video = await fetch('/api/video/generate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    character_id: char.id,
    prompt: "Walking forward confidently",
    options: {
      duration_seconds: 8,
      resolution: "1080p"
    }
  })
}).then(r => r.json());

console.log(video.result_url);
```

---

**That's it! Keep it simple.**
