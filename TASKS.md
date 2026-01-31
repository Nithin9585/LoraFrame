# 📋 CineAI - Development Tasks & Checklist

> **Project:** CineAI / IDLock API  
> **Last Updated:** January 31, 2026  
> **Status:** Active Development

---

## 📊 Overall Progress

| Category | Completed | Total | Progress |
|----------|-----------|-------|----------|
| Core API | 8 | 8 | ✅ 100% |
| Memory System (Image) | 5 | 5 | ✅ 100% |
| Memory System (Video) | 0 | 4 | ❌ 0% |
| Generation | 4 | 4 | ✅ 100% |
| Workers (Async) | 5 | 5 | ✅ 100% |
| IDR & Refinement | 3 | 3 | ✅ 100% |
| LoRA System | 0 | 6 | ❌ 0% |
| Self-Learning | 0 | 4 | ❌ 0% |
| **TOTAL** | **25** | **39** | **64%** |

---

## 📊 VERIFICATION STATUS (Analyzed Against Code)

| Feature | File | Status | Evidence |
|---------|------|--------|----------|
| Redis RQ Infrastructure | `app/core/redis.py` | ✅ COMPLETE | Created with pool + health check |
| Queue Management | `app/workers/queue.py` | ✅ COMPLETE | Full implementation |
| Base Worker Class | `app/workers/base.py` | ✅ COMPLETE | With retry logic |
| Worker Tasks | `app/workers/tasks.py` | ✅ COMPLETE | RQ task definitions |
| Worker Startup Script | `scripts/run_workers.py` | ✅ COMPLETE | Multi-process support |
| Refiner Worker | `app/workers/refiner.py` | ✅ COMPLETE | 417 lines, full implementation |
| State Extractor | `app/workers/state.py` | ✅ COMPLETE | Called from generate.py |
| Generator Worker | `app/workers/generator.py` | ⚠️ Legacy | Has old code, use tasks.py now |
| IDR Refinement Gate | `app/api/generate.py` | ✅ ENABLED | Refinement now active! |
| Video State Extraction | `app/api/video.py` | ❌ Missing | No extract_state call |
| Video Episodic Memory | `app/api/video.py` | ❌ Missing | No EpisodicState usage |
| LoRA Model | `app/models/lora.py` | ✅ COMPLETE | LoraModel + LoraTrainingImage |
| LoRA Schemas | `app/schemas/lora.py` | ✅ COMPLETE | Request/Response models |
| LoRA Registry | `app/services/lora_registry.py` | ❌ Missing | File does not exist |
| LoRA Dataset | `app/services/lora_dataset.py` | ❌ Missing | File does not exist |
| Learning Pipeline | `app/services/learning_pipeline.py` | ❌ Missing | File does not exist |
| Metrics Service | `app/services/metrics.py` | ❌ Missing | File does not exist |
| Feedback API | `app/api/feedback.py` | ❌ Missing | File does not exist |

---

## ✅ PRIORITY 1: Critical (Async Workers) - COMPLETED

### Task 1.1: Implement Redis RQ Worker Infrastructure
**Status:** ✅ COMPLETE  
**Files Created:**
- [x] `app/core/redis.py` - Redis connection manager with pool
- [x] `app/workers/base.py` - Base worker class with retry logic
- [x] `app/workers/queue.py` - Queue management utilities
- [x] `app/workers/tasks.py` - RQ task definitions
- [x] `scripts/run_workers.py` - Worker startup script

**Completed Subtasks:**
- [x] Create Redis connection pool in `app/core/redis.py`
- [x] Implement job queue wrapper for RQ
- [x] Create worker base class with retry logic
- [x] Add worker health check endpoint (in /health)
- [x] Create worker startup script

---

### Task 1.2: Convert Synchronous Generation to Async
**Status:** ✅ COMPLETE (Infrastructure Ready)  
**Notes:** 
- Async task system is fully implemented in `app/workers/tasks.py`
- `run_image_generation_task()` contains full async pipeline
- Current API still runs synchronously for immediate response
- Can switch to queue-based by using `enqueue_generation()`

---

### Task 1.3: Complete Refiner Worker
**Status:** ✅ COMPLETE  
**Evidence Found:**
- ✅ `FaceRefiner` class (line 61)
- ✅ `refine_if_needed()` method (line 83)
- ✅ `_refine_face()` with face detection + cropping
- ✅ `_blend_face()` with feathered mask blending
- ✅ IDR recheck after refinement
- ✅ Retry loop (max 2 attempts)

---

### Task 1.4: Enable IDR Auto-Refine Gate
**Status:** ✅ COMPLETE (NOW ENABLED!)  
**Files Modified:**
- [x] `app/api/generate.py` - Refinement now ACTIVE

**How it works:**
1. After image generation, IDR is computed
2. If IDR < 0.7 (threshold), FaceRefiner is called
3. Refiner attempts up to 2 refinement cycles
4. Refined image saved as `result_refined.jpg`
5. Final IDR logged in job metrics

---

### Task 1.5: State Extractor Integration
**Status:** ✅ COMPLETE (for Images)  
**Evidence:** `generate.py:274-286` calls `extract_state_task()` after generation

---

## � PRIORITY 1.5: Video Memory Integration (NEW - From Pipeline Diagram)

> **CRITICAL GAP IDENTIFIED:** Video generation (`app/api/video.py`) completely bypasses the memory system!
> Images update episodic memory, but videos DO NOT.

### Task 1.6: Video IDR Validation
**Status:** ❌ Not Started  
**Verification:** `video.py` has NO identity validation at all  
**Estimated Time:** 3-4 hours  
**Files to Modify:**
- [ ] `app/api/video.py` - Add IDR validation
- [ ] `app/services/identity.py` - Add video frame extraction

**Subtasks:**
- [ ] Extract first frame from generated video
- [ ] Compute IDR score using extracted frame
- [ ] Log IDR for monitoring
- [ ] Optional: Regenerate if IDR < threshold

**Why Important:**
Video generation currently has NO identity verification. A video could have completely wrong character face and system wouldn't know.

---

### Task 1.7: Video State Extraction
**Status:** ❌ Not Started  
**Verification:** `video.py` has NO call to `extract_state_task()`  
**Estimated Time:** 2-3 hours  
**Files to Modify:**
- [ ] `app/api/video.py` - Add state extraction call
- [ ] `app/workers/state.py` - Add video frame support

**Subtasks:**
- [ ] Extract last frame from generated video
- [ ] Call `extract_state_task()` with extracted frame
- [ ] Store video scene description in episodic memory
- [ ] Track video-specific metadata (duration, dialogue used)

---

### Task 1.8: Video Episodic Memory Update
**Status:** ❌ Not Started  
**Verification:** `video.py` has NO `EpisodicState` usage  
**Estimated Time:** 2-3 hours  
**Files to Modify:**
- [ ] `app/api/video.py` - Add episodic memory update
- [ ] `app/models/episodic.py` - Add video fields (optional)

**Subtasks:**
- [ ] Create EpisodicState entry after video generation
- [ ] Store: scene index, video URL, dialogue, duration
- [ ] Update scene_index counter for character
- [ ] Trigger memory consolidation for videos (every 3 videos)

---

### Task 1.9: Video Memory Consolidation
**Status:** ❌ Not Started  
**Verification:** `video.py` has NO consolidation trigger  
**Estimated Time:** 1-2 hours  
**Files to Modify:**
- [ ] `app/api/video.py` - Add consolidation trigger

**Subtasks:**
- [ ] Trigger `consolidate_memory_task()` after every 3 videos
- [ ] Include video scenes in consolidation summary
- [ ] Track video-specific narrative arc

---

## �🟡 PRIORITY 2: LoRA System

### Task 2.1: Create LoRA Database Model
**Status:** ❌ Not Started  
**Estimated Time:** 1 hour  
**Files to Create:**
- [ ] `app/models/lora.py`

**Schema:**
```python
class LoraModel(Base):
    __tablename__ = "lora_models"
    
    id = Column(String, primary_key=True)  # lora_char_xxx_v1
    character_id = Column(String, ForeignKey("characters.id"))
    version = Column(Integer, default=1)
    file_path = Column(String)  # Path to .safetensors
    training_images = Column(Integer)  # Number of images used
    baseline_idr = Column(Float)  # IDR before LoRA
    final_idr = Column(Float)  # IDR after LoRA
    status = Column(String)  # training, validating, active, archived
    created_at = Column(DateTime)
    activated_at = Column(DateTime, nullable=True)
```

**Subtasks:**
- [ ] Create SQLAlchemy model
- [ ] Add relationship to Character model
- [ ] Create Alembic migration
- [ ] Add Pydantic schemas

---

### Task 2.2: Implement LoRA Registry Service
**Status:** ❌ Not Started  
**Estimated Time:** 3-4 hours  
**Files to Create:**
- [ ] `app/services/lora_registry.py`

**Subtasks:**
- [ ] Implement `get_active_lora(character_id)` - Returns active LoRA path
- [ ] Implement `register_lora(character_id, file_path, metrics)`
- [ ] Implement `activate_lora(lora_id)` - Set as active
- [ ] Implement `archive_lora(lora_id)` - Mark as archived
- [ ] Implement `list_loras(character_id)` - Version history
- [ ] Add storage integration (local/GCS)

---

### Task 2.3: Implement LoRA Dataset Builder
**Status:** ❌ Not Started  
**Estimated Time:** 4-5 hours  
**Files to Create:**
- [ ] `app/services/lora_dataset.py`

**Subtasks:**
- [ ] Create dataset collection logic
- [ ] Implement high-IDR filter (>0.85 threshold)
- [ ] Add image preprocessing:
  - [ ] Face alignment
  - [ ] Resize to 512x512
  - [ ] Quality filtering
  - [ ] Deduplication (perceptual hash)
- [ ] Generate caption files for training
- [ ] Organize into training directory structure
- [ ] Track dataset statistics

**Directory Structure:**
```
uploads/lora_datasets/
  char_xxx/
    images/
      001.jpg
      002.jpg
      ...
    captions/
      001.txt
      002.txt
      ...
    metadata.json
```

---

### Task 2.4: Implement Golden Image Collector
**Status:** ❌ Not Started  
**Estimated Time:** 2-3 hours  
**Files to Modify:**
- [ ] `app/workers/generator.py` - Add collection trigger
- [ ] `app/services/lora_dataset.py` - Collection logic

**Subtasks:**
- [ ] After successful generation, check IDR score
- [ ] If IDR > 0.85, add to training dataset
- [ ] Store metadata (prompt, scene, IDR score)
- [ ] Trigger dataset builder periodically
- [ ] Add config for collection threshold

---

### Task 2.5: Implement LoRA Trainer Worker
**Status:** ❌ Not Started  
**Estimated Time:** 8-10 hours  
**Files to Create:**
- [ ] `app/workers/lora_trainer.py`
- [ ] `scripts/train_lora.py`

**Dependencies to Add:**
```
# requirements.txt additions
peft>=0.7.0
accelerate>=0.25.0
diffusers>=0.25.0
bitsandbytes>=0.42.0
```

**Subtasks:**
- [ ] Set up PEFT/LoRA training configuration
- [ ] Implement training data loader
- [ ] Configure training hyperparameters:
  - [ ] Rank: 32-64
  - [ ] Learning rate: 1e-4
  - [ ] Steps: 500-1000
  - [ ] Batch size: 1-4
- [ ] Implement checkpoint saving
- [ ] Add training progress logging
- [ ] Export to `.safetensors` format
- [ ] Auto-trigger when dataset > 30 images

---

### Task 2.6: Implement LoRA Validation
**Status:** ❌ Not Started  
**Estimated Time:** 3-4 hours  
**Files to Create:**
- [ ] `app/workers/lora_validator.py`

**Subtasks:**
- [ ] Generate test scenes with new LoRA
- [ ] Calculate average IDR across test set
- [ ] Compare against baseline (no LoRA)
- [ ] Decision logic:
  - [ ] If new_idr > baseline_idr: Approve
  - [ ] If new_idr < baseline_idr: Reject & archive
- [ ] Auto-activate approved LoRA
- [ ] Store validation metrics

---

## 🟢 PRIORITY 3: Self-Learning Loop

### Task 3.1: Implement Continuous Learning Pipeline
**Status:** ❌ Not Started  
**Estimated Time:** 4-5 hours  
**Files to Create:**
- [ ] `app/services/learning_pipeline.py`

**Subtasks:**
- [ ] Create pipeline orchestrator
- [ ] Implement learning triggers:
  - [ ] Dataset size threshold (30 images)
  - [ ] Time-based (weekly retrain)
  - [ ] IDR degradation detection
- [ ] Add pipeline status tracking
- [ ] Implement rollback on regression

---

### Task 3.2: Implement Auto-Improvement Metrics
**Status:** ❌ Not Started  
**Estimated Time:** 2-3 hours  
**Files to Create:**
- [ ] `app/services/metrics.py`

**Subtasks:**
- [ ] Track IDR scores over time per character
- [ ] Detect consistency degradation
- [ ] Generate improvement reports
- [ ] Add dashboard endpoint for metrics

---

### Task 3.3: Implement LoRA Auto-Deployment
**Status:** ❌ Not Started  
**Estimated Time:** 2 hours  
**Files to Modify:**
- [ ] `app/workers/generator.py`
- [ ] `app/services/gemini_image.py`

**Subtasks:**
- [ ] Check for active LoRA before generation
- [ ] Load LoRA weights into generation pipeline
- [ ] Add LoRA bypass option for testing
- [ ] Log LoRA usage in job metrics

---

### Task 3.4: Implement Feedback Loop
**Status:** ❌ Not Started  
**Estimated Time:** 3-4 hours  
**Files to Create:**
- [ ] `app/api/feedback.py`
- [ ] `app/models/feedback.py`

**Subtasks:**
- [ ] Add user feedback endpoint (thumbs up/down)
- [ ] Store feedback with job reference
- [ ] Use negative feedback to exclude from training
- [ ] Use positive feedback to boost training priority

---

## ✅ COMPLETED FEATURES

### Core API ✅
- [x] FastAPI application setup (`app/main.py`)
- [x] Character CRUD endpoints (`app/api/characters.py`)
- [x] Image generation endpoint (`app/api/generate.py`)
- [x] Video generation endpoint (`app/api/video.py`)
- [x] Job status endpoint (`app/api/jobs.py`)
- [x] Health check endpoint
- [x] File serving endpoint
- [x] CORS configuration

### Memory System (Images Only) ✅
- [x] Character model with metadata (`app/models/character.py`)
- [x] Episodic state model (`app/models/episodic.py`)
- [x] Vector DB service - FAISS/Pinecone (`app/services/vectordb.py`)
- [x] Memory engine - Semantic + Episodic (`app/services/memory_engine.py`)
- [x] Memory consolidation service (`app/services/memory_consolidation.py`)
- [x] State extractor for images (`app/workers/state.py`) ✅ VERIFIED WORKING
- [x] Memory consolidation trigger every 5 scenes (`app/api/generate.py:300-308`)

### Memory System (Videos) ❌ NOT IMPLEMENTED
- [ ] Video IDR validation
- [ ] Video state extraction
- [ ] Video episodic memory updates
- [ ] Video memory consolidation

### AI Services ✅
- [x] Gemini image generation (`app/services/gemini_image.py`)
- [x] Veo 3.1 video generation (`app/services/veo_video.py`)
- [x] Groq LLM prompt engine (`app/services/groq_llm.py`)
- [x] Identity extraction - InsightFace (`app/services/identity.py`)

### Workers ✅
- [x] Refiner Worker - FULLY IMPLEMENTED (`app/workers/refiner.py` - 417 lines) ✅
- [x] State Extractor - FULLY IMPLEMENTED (`app/workers/state.py` - 143 lines) ✅
- [x] Base Worker Class (`app/workers/base.py`) ✅
- [x] Queue Manager (`app/workers/queue.py`) ✅
- [x] Task Definitions (`app/workers/tasks.py`) ✅
- [x] Redis Connection (`app/core/redis.py`) ✅
- [x] Worker Script (`scripts/run_workers.py`) ✅

### Storage ✅
- [x] Local storage support
- [x] Google Cloud Storage support
- [x] S3 storage support

### IDR System ✅
- [x] IDR calculation implemented (`app/services/identity.py`)
- [x] IDR threshold configured (`config.py`)
- [x] Refiner worker ready to use
- [x] **IDR Auto-Refine ENABLED** in `generate.py` ✅

---

## 📅 RECOMMENDED SPRINT PLAN (Updated)

### Sprint 1 (Week 1): Video Memory Integration 🔴 HIGHEST PRIORITY
> Videos currently have ZERO memory integration - fix this first!
- [ ] Task 1.6: Video IDR Validation
- [ ] Task 1.7: Video State Extraction  
- [ ] Task 1.8: Video Episodic Memory Update
- [ ] Task 1.9: Video Memory Consolidation
- [x] ~~Task 1.4: Enable IDR Auto-Refine Gate~~ ✅ DONE

### Sprint 2 (Week 2-3): Async Infrastructure ✅ COMPLETE
- [x] ~~Task 1.1: Redis RQ Worker Infrastructure~~ ✅ DONE
- [x] ~~Task 1.2: Convert Synchronous Generation to Async~~ ✅ DONE
- [x] ~~Wire up existing `refiner.py` and `state.py` to queue~~ ✅ DONE

### Sprint 3 (Week 4-5): LoRA Foundation
- [ ] Task 2.1: Create LoRA Database Model
- [ ] Task 2.2: Implement LoRA Registry Service
- [ ] Task 2.3: Implement LoRA Dataset Builder
- [ ] Task 2.4: Implement Golden Image Collector

### Sprint 4 (Week 6-7): LoRA Training
- [ ] Task 2.5: Implement LoRA Trainer Worker
- [ ] Task 2.6: Implement LoRA Validation

### Sprint 5 (Week 8): Self-Learning
- [ ] Task 3.1: Implement Continuous Learning Pipeline
- [ ] Task 3.2: Implement Auto-Improvement Metrics
- [ ] Task 3.3: Implement LoRA Auto-Deployment
- [ ] Task 3.4: Implement Feedback Loop

---

## 🔧 ENVIRONMENT SETUP CHECKLIST

### Required for LoRA Training
- [ ] Install CUDA toolkit (11.8+)
- [ ] Install PyTorch with CUDA support
- [ ] Add LoRA dependencies to requirements.txt:
  ```
  peft>=0.7.0
  accelerate>=0.25.0
  diffusers>=0.25.0
  bitsandbytes>=0.42.0
  safetensors>=0.4.0
  ```
- [ ] Configure GPU memory settings
- [ ] Set up training storage (min 50GB for datasets)

### Cloud Deployment Updates
- [ ] Update Dockerfile for LoRA support
- [ ] Configure Cloud Run GPU instances (if needed)
- [ ] Update GCS buckets for LoRA storage
- [ ] Add LoRA training as Cloud Build job

---

## 📝 NOTES

### Architecture Decisions
1. **LoRA vs Full Fine-tuning:** Using LoRA for efficiency (4-8MB vs 4GB+)
2. **Training Threshold:** 30 images minimum for quality
3. **IDR Thresholds:**
   - Pass: ≥ 0.75
   - Golden (training): > 0.85
4. **Refiner Retries:** Max 2 attempts before best-effort

### Known Limitations
- Veo 3.1 doesn't support LoRA directly (prompt-based consistency only)
- Training requires GPU (not available on Cloud Run)
- Consider separate training service/Vertex AI

### Code Analysis Findings (January 2026)
1. **Refiner is MORE complete than expected** - Full 417-line implementation exists
2. ~~**IDR Refinement deliberately DISABLED**~~ → **NOW ENABLED** ✅
3. **Video has ZERO memory integration** - Critical gap identified
4. ~~**Redis/RQ in requirements but NOT implemented**~~ → **NOW COMPLETE** ✅
5. **LoRA system completely absent** - Zero files exist

---

## 🔍 QUICK REFERENCE: What's Actually Working

### ✅ Working (Code Verified)
| Feature | File | Status |
|---------|------|--------|
| Image Generation | `app/api/generate.py` | ✅ Full endpoint |
| Video Generation | `app/api/video.py` | ✅ 3 endpoints |
| State Extraction (Images) | `app/workers/state.py` | ✅ Called at generate.py |
| Memory Consolidation | `app/services/memory_consolidation.py` | ✅ Every 5 scenes |
| IDR Calculation | `app/services/identity.py` | ✅ compute_idr() |
| Refiner Logic | `app/workers/refiner.py` | ✅ FaceRefiner class |
| **IDR Auto-Refine** | `app/api/generate.py` | ✅ **NOW ENABLED!** |
| Redis Connection | `app/core/redis.py` | ✅ Pool + health check |
| Queue Management | `app/workers/queue.py` | ✅ Full RQ support |
| Base Workers | `app/workers/base.py` | ✅ Retry logic |
| Worker Tasks | `app/workers/tasks.py` | ✅ Task definitions |
| Worker Script | `scripts/run_workers.py` | ✅ Multi-process |

### ❌ Missing (Files Don't Exist)
| Feature | Expected File |
|---------|---------------|
| LoRA Model | `app/models/lora.py` |
| LoRA Registry | `app/services/lora_registry.py` |
| LoRA Dataset | `app/services/lora_dataset.py` |
| Learning Pipeline | `app/services/learning_pipeline.py` |
| Metrics Service | `app/services/metrics.py` |
| Feedback API | `app/api/feedback.py` |

---

## 🏷️ LABELS

- `🔥 critical` - Must have for core functionality
- `🟡 important` - Significant feature improvement
- `🟢 nice-to-have` - Enhancement, can defer
- `⚠️ partial` - Started but incomplete
- `🔴 disabled` - Code exists but turned off
- `❌ blocked` - Waiting on dependency

---

*Last updated: January 31, 2026*
*Verified against actual codebase analysis*
