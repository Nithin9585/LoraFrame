# `tasks.md` — Backend Pipeline (Persistent Character Memory System / Nano-Banana)

> Project: **IDLock / Persistent Character Memory & Consistency System**
> Engine: **Nano-Banana 2.5 Flash Image API** (rendering)
> Purpose: Extract a character's identity from reference image(s), store semantic/episodic memory, produce consistent images across scenes via RAG + LLM prompt engineering, and update memory after generation.

---

## Table of contents

1. [High-level architecture (overview)](#1-high-level-architecture-overview)
2. [Component list & responsibilities](#2-component-list--responsibilities)
3. [Data models & storage schema](#3-data-models--storage-schema)
4. [Core API endpoints (contracts)](#4-core-api-endpoints-contracts)
5. [Detailed backend flows (step-by-step)](#5-detailed-backend-flows-step-by-step)
6. [Workers, queues, and async tasks](#6-workers-queues-and-async-tasks)
7. [Prompt engineering & LLM integration details](#7-prompt-engineering--llm-integration-details)
8. [Vector DB / RAG retrieval & merging algorithm](#8-vector-db--rag-retrieval--merging-algorithm)
9. [Image postprocessing & refiner loop](#9-image-postprocessing--refiner-loop)
10. [Security, privacy & abuse prevention](#10-security-privacy--abuse-prevention)
11. [Monitoring, logging & metrics](#11-monitoring-logging--metrics)
12. [Testing & evaluation plan](#12-testing--evaluation-plan)
13. [Deployment & infra suggestions (dev → prod)](#13-deployment--infra-suggestions-dev-→-prod)
14. [Dev tasks & timeline (prioritized)](#14-dev-tasks--timeline-prioritized)
15. [Appendix: sample code snippets & env vars](#15-appendix-sample-code-snippets--env-vars)

---

## 1 — High-level architecture (overview)

Mermaid overview (paste into Markdown capable of mermaid):

```mermaid
graph TD
  subgraph Frontend
    UI[Web UI / CLI / Mobile]
  end

  subgraph API
    API[FastAPI / Node API]
  end

  subgraph Orchestration
    Orch[Orchestrator / Job Scheduler]
    Queue[(Job Queue: Redis / RQ / Celery)]
    WorkerPool[Worker Pool]
  end

  subgraph Memory
    VDB[(Vector DB: Pinecone/Milvus/FAISS)]
    PG[(Postgres: metadata)]
    S3[(Object Storage)]
    Cache[(Redis Cache)]
  end

  subgraph LLM
    LLM[LLM: GPT-4o / Claude]
  end

  subgraph Render
    NB[Nano-Banana 2.5 Flash API]
  end

  UI --> API
  API --> Orch
  Orch --> Queue
  Queue --> WorkerPool
  WorkerPool --> VDB
  WorkerPool --> PG
  WorkerPool --> S3
  WorkerPool --> LLM
  WorkerPool --> NB
  NB --> WorkerPool
  WorkerPool --> API
  Cache --> API
  PG --> API
```

**Short description:**
User uploads reference image(s) → API extracts identity embedding(s) → stores in Vector DB and metadata DB → user requests scene generation → orchestrator places job in queue → worker retrieves memory + calls LLM to produce a context-locked prompt → calls Nano-Banana API with that prompt (and optional image conditioning) → receives generated output → runs state extractor + optional face-refiner → stores output and updates episodic memory → returns result to user.

---

## 2 — Component list & responsibilities

* **Frontend (UI)**

  * Upload references, create Character, add tags, request generations, view history, and consent UI.

* **API Layer (FastAPI / Node)**

  * Auth, input validation, file upload, orchestration calls, job submission, job status endpoints.

* **Orchestrator / Scheduler**

  * Accepts generate requests, schedules jobs to queue, enforces rate limits/quotas, selects model endpoint (Nano-Banana or fallback), retries, and fallback logic.

* **Job Queue & Workers**

  * Redis + RQ or Celery for async tasks. Workers run the pipeline steps (embedding extraction, LLM prompt generation, Nano-Banana call, postprocessing, memory update).

* **Identity Extractor**

  * Face encoder (InsightFace / ArcFace) + CLIP (or hybrid) to produce identity and style embeddings.

* **Vector DB**

  * Pinecone/Milvus/FAISS to store semantic/episodic vectors and support similarity queries.

* **Metadata DB**

  * Postgres for character registry, scene records, job records, user data, policy/consent info.

* **Object Storage**

  * S3-compatible (AWS S3, MinIO) for images, logs, and model artifacts.

* **LLM Prompt Engine**

  * GPT-4o / Claude wrapper for generating the final Kontext prompt adhering to a template.

* **Renderer**

  * Nano-Banana 2.5 Flash API client wrapper.

* **State Extractor**

  * Vision + LLM combo that reads generated image and outputs episodic metadata (clothing tags, injuries, props).

* **Refiner Loop** (optional)

  * Crop & upscale face → re-render with high-identity emphasis (if using secondary rendering or local re-rendering).

* **Cache**

  * Redis for frequent retrieval, rate-limit counters, ephemeral staging.

* **Auth & Billing**

  * JWT-based auth, API keys, usage metering for credits model if you monetize.

* **Monitoring & Logging**

  * Prometheus/Grafana for metrics, ELK/Datadog for logs.

---

## 3 — Data models & storage schema

### Postgres schemas (simplified)

**users**

```sql
id UUID PRIMARY KEY
email TEXT UNIQUE
name TEXT
created_at TIMESTAMP
```

**characters**

```sql
id UUID PRIMARY KEY
user_id UUID REFERENCES users(id)
name TEXT
description TEXT
semantic_vector_id TEXT -- pointer to VDB ID
semantic_dim int
base_image_url TEXT -- S3 path
created_at TIMESTAMP
metadata JSONB -- canonical sheet (hair, eye color, tags)
```

**episodic_states**

```sql
id UUID PRIMARY KEY
character_id UUID REFERENCES characters(id)
vdb_vector_id TEXT
scene_index INT
tags TEXT[] -- ["battle_armor", "muddy"]
image_url TEXT -- S3
created_at TIMESTAMP
notes TEXT
```

**jobs**

```sql
id UUID PRIMARY KEY
user_id UUID
character_id UUID
prompt TEXT
pose_image_url TEXT
status TEXT -- queued, running, success, failed
result_url TEXT -- s3 url
metrics JSONB -- time, idr etc
created_at TIMESTAMP
updated_at TIMESTAMP
```

**usage_log**

```sql
id UUID PRIMARY KEY
user_id UUID
job_id UUID
credits_used INT
cost_usd NUMERIC
timestamp TIMESTAMP
```

### Vector DB schema (Pinecone / Milvus)

* Index name: `idlock_characters`
* Namespace: `semantic` and `episodic`
* Vector dimension: `512` or `1024` (match your encoder)
* Metadata fields: `character_id`, `type`, `scene`, `tags`, `image_url`, `created_at`, `confidence`

Example vector entry metadata:

```json
{
  "character_id": "char_abc123",
  "type": "semantic",
  "image_url": "s3://.../ref.jpg",
  "created_at": "2026-01-27T10:00:00Z"
}
```

---

## 4 — Core API endpoints (contracts)

Design with FastAPI; JSON contract examples:

### Authentication

* `POST /auth/register` — register user
* `POST /auth/login` — login returns JWT

### Character management

* `POST /api/v1/characters` — create character

  * form-data: `name`, `description`, `image[]` (1-5 files), `consent` boolean
  * returns: `{character_id, semantic_vector_id}`

* `GET /api/v1/characters/{id}` — get character metadata

* `DELETE /api/v1/characters/{id}` — delete character (GDPR/forget)

### Generation

* `POST /api/v1/generate` — create generation job

  * body:

    ```json
    {
      "character_id": "char_...",
      "prompt": "Arya fights a dragon on a cliff at sunset",
      "pose_image_url": "s3://.../pose.jpg",
      "style_overrides": ["worn leather armor"],
      "options": {"video": false, "refine_face": true}
    }
    ```
  * returns: `{job_id, status: queued}`

* `GET /api/v1/job/{job_id}` — job status and result info

* `GET /api/v1/character/{id}/history` — list episodic states and generated images

### Admin / monitoring

* `GET /api/v1/admin/health` — service health
* `GET /api/v1/admin/metrics` — consumption metrics

---

## 5 — Detailed backend flows (step-by-step)

Below are the major end-to-end flows with step-by-step actions.

### 5.1 Create Character (user flow)

1. User uploads 1–5 reference images + fills `character sheet` (name, must-have traits).
2. API stores raw images in S3 with `uploads/characters/<character_id>/ref_<n>.jpg`.
3. API enqueues `extract_identity` job (fast response returns `character_id`).
4. Worker runs identity extraction:

   * Run face detection & alignment (MTCNN/dlib/InsightFace util).
   * Compute face embedding (ArcFace/InsightFace). Also compute CLIP embedding for style.
   * Aggregate multi-reference embeddings into canonical semantic embedding (methods below).
5. Upsert semantic vector into Vector DB (namespace `semantic`) and save `vector_id` in Postgres `characters` record.
6. Return success to user and show character sheet with preview + editable canonical attributes.

**Multi-reference aggregation:**

* Normalize each embedding; semantic_embedding = mean(normalized_embeddings). Store also per-view embeddings for diagnostics.

### 5.2 Generate Scene (user requests generation)

1. API validates request, checks quotas, checks consent.
2. Create `job` record (status `queued`) and push job to Queue.
3. Orchestrator picks job, moves status to `running`.
4. Worker retrieves `semantic` vector and top-K episodic vectors (see merging algorithm).
5. Worker calls LLM prompt-engine with:

   * Master style guide (persisted), character sheet, latest episodic entries (sliding window), user prompt, negative tokens.
   * Template-based fill to produce deterministic Kontext prompt.
6. LLM returns final prompt (or prompt-block). Worker sanitizes & enforces constraints (no banned content, no personal data).
7. Worker calls Nano-Banana API with prompt. If Nano-Banana supports image-conditioning, include pose image. Set parameters (temperature, guidance equivalents).
8. Receive image(s). Store raw output to S3 (job results).
9. Run State Extractor:

   * Vision model or LLM-based classifier identifies clothing, injuries, props; compute generated face embedding.
10. Compute IDR: cosine_similarity(semantic_face_embedding, generated_face_embedding). Save in `job.metrics`.
11. Optionally run Refiner Loop (face crop → upscale → re-generate or re-sample). Blend back.
12. Upsert new episodic vector in Vector DB namespace `episodic` with `scene_index` and tags.
13. Update job status with result URL and metrics.
14. Notify user (websocket / webhook / polling).

**Failure/retry rules:**

* On Nano-Banana 5xx: retry 1-2 times with backoff, then mark failed; return error with reason.
* On IDR below threshold (e.g., <0.7): mark as `needs_more_refs` and optionally auto-retry (ask user for another ref).

---

## 6 — Workers, queues, and async tasks

### Queue & worker system

* **Queue:** Redis + RQ or Celery (Redis backend). Redis simpler for hackathon; Celery if you need robust scheduling.
* **Worker types:**

  * `extractor-worker` — identity extraction (CPU + small GPU).
  * `generator-worker` — LLM prompt generation + Nano-Banana calls (stateless).
  * `refiner-worker` — face crop/upscale/refine (may use separate GPU).
  * `state-update-worker` — state extraction & DB upsert.
* **Worker scaling:** horizontal. Use k8s HPA or autoscaling groups.

### Job timeouts and concurrency

* Long-running jobs should stream logs. Set timeouts: extraction 60s, generation 120s, refine 90s. Provide cancellation endpoint.

---

## 7 — Prompt engineering & LLM integration details

### Prompt template (stable)

Use a deterministic template — LLM fills slots. Example:

```
[MASTER_STYLE_GUIDE]
{master_style_guide_text}

[CHARACTER_SHEET]
Name: {name}
Face: {face_description} -- do not alter facial proportions, eye color, or placement
Hair: {hair}
Distinctives: {distinctive_marks}
SemanticTags: {semantic_tags}

[RECENT_STATE] (only include up to N recent episodic entries)
{scene_1_text}
{scene_2_text}

[USER_SCENE]
{user_prompt}

[CONSTRAINTS]
- Never change facial structure or eye color.
- Do not merge identity with background characters.
- Negative tokens: {negative_tokens}
- Final output: one photorealistic image, 1024x1024 (or requested)

Produce a cleaned, concise prompt string only — no commentary.
```

**Negative tokens** (mandatory): `changing face, morphing identity, different clothes, watermark, text, extra limbs, distorted eyes`.

### LLM handling

* Use LLM to `format` prompt. Avoid asking LLM to invent new identity properties.
* Enforce `temperature=0` deterministic or low temperature for consistent prompts.
* Cache LLM outputs for identical (character_id, prompt_hash) to reduce cost.

---

## 8 — Vector DB / RAG retrieval & merging algorithm

### Retrieval

* Query: retrieve top `K_semantic` (usually 1 semantic) + top `K_episodic` (e.g., K=3) by similarity.
* Use metadata filters (character_id).

### Merging algorithm (recommended)

1. Let `S` = semantic vector (v_s).
2. Let episodic vectors `E = {e1, e2, ..., ek}` sorted by timestamp desc.
3. Weighted merge:

   ```
   merged = normalize( w_s * v_s + Σ_i (w_e * decay(i) * e_i) )
   ```

   * `w_s = 0.6` (semantic weight)
   * `w_e = 0.4` (episodic total)
   * `decay(i) = alpha^(i-1)` (alpha ~ 0.6) or linear decay
4. For discrete tags (clothing), prefer most recent episodic entries.
5. Provide both `merged` and the list of `E` to LLM as textual episodic history (sliding window).

**Alternative:** attention merge (trainable) — not needed for MVP.

---

## 9 — Image postprocessing & refiner loop

### FaceDetailer loop (optional / fallback)

* Detect face bbox (MTCNN / FaceNet)
* Upscale (Real-ESRGAN or GFPGAN) to 1024x1024
* Re-run Nano-Banana for the face region only with an identity-strong prompt (or run local model if available)
* Alpha-blend refined face back into original image using soft mask

**When to run:** If IDR < threshold (e.g., 0.8) OR if user requested `refine_face: true`.

**Blending:** Use Poisson blending or feathering to minimize seams.

---

## 10 — Security, privacy & abuse prevention

### Consent & user control

* Require `consent` checkbox on character creation: store `consent_given_at`.
* Provide `DELETE /api/v1/characters/{id}` to permanently remove semantic & episodic vectors (upsert deletion in VDB) and all images.
* Keep audit logs of deletions.

### Misuse safeguards

* Content moderation: run user prompt + generated image through safety checks (NSFW detector, face recognition for celebrities).
* Rate limit per-user and per-IP to prevent mass misuse.
* Watermark option for generated images (by default for public demo / trial).
* Takedown & reporting flow: `POST /api/v1/report` to flag a character or image and freeze access.

### Secrets & keys

* Store keys in secret manager (Vault / AWS SecretsManager). Do not commit to repo.
* Restrict Nano-Banana & LLM API keys usage; rotate regularly.

---

## 11 — Monitoring, logging & metrics

### Health & metrics

* **Prometheus metrics**: job queue length, job duration, error rates, average IDR per job, Nano-Banana latency, LLM latency, VDB latency.
* **SLOs**: 95% of generation jobs < 7s (if using Nano-Banana) — adjust based on real latencies.
* **Alerts**: worker crash rate > 1/hr, vector DB unavailable, storage errors.

### Logging

* Structured logs (JSON) with job_id, character_id, user_id, step, response times.
* Persist logs for 30 days (or per policy).

---

## 12 — Testing & evaluation plan

### Unit tests

* Identity extractor returns consistent vector shape & stable across small augmentations.
* Vector DB upsert/query round-trip.
* LLM prompt templating deterministic output.

### Integration tests

* End-to-end create character → generate scene → verify job status transitions and S3 objects.
* Failure injection tests for Nano-Banana timeouts and retries.

### Evaluation metrics

* **IDR (identity retention)**: cosine similarity between reference face embedding and generated face embedding. Track mean, std, min.
* **HumanSame%**: human raters evaluate same/different (target > 80%).
* **Latency**: end-to-end generation time, LLM time, Nano-Banana time.
* **Failure rate**: % of jobs that require manual intervention.

---

## 13 — Deployment & infra suggestions (dev → prod)

### Dev / hackathon

* Small VM (1x RTX 3090 if you must run any local GPU tasks) or pure cloud with Nano-Banana (no local GPU).
* Managed Pinecone free tier or FAISS for local testing.
* Postgres (managed or local Docker), Redis.

### Prod

* Kubernetes cluster with autoscaled worker pools.
* Managed vector DB (Pinecone / Milvus on GCP/Azure).
* Managed LLM provisioning or API keys (OpenAI/GPT-4o/Anthropic/Claude).
* Nano-Banana API with fallback to another provider.
* S3 for images with CDN for delivery.

### Approx costs (very approximate)

* Nano-Banana API: variable (per image). Estimate $0.02–$0.5 per image depending on quality and provider pricing.
* Pinecone: small index ~$20–$200/month depending on usage.
* LLM calls: GPT-4o costs vary strongly — minimize by templated prompts and caching.

---

## 14 — Dev tasks & timeline (prioritized) — sprint-style

> **Assumptions:** 2 engineers (backend + frontend), 48–72 hour hackathon MVP.

### Sprint 0 — Setup (2–4 hours)

* [ ] Create repo skeleton and `tasks.md` (this doc). **Owner:** Backend
* [ ] Create env and CI (GitHub Actions basic lint/test). **Owner:** Backend
* [ ] Provision Nano-Banana API key, Pinecone free account, S3 bucket. **Owner:** Backend

### Sprint 1 — Identity extraction & character create (6–10 hours)

* [ ] Implement `POST /api/v1/characters` endpoint (FastAPI). **Owner:** Backend
* [ ] Implement identity-extraction worker: face detection + ArcFace + CLIP. **Owner:** Backend
* [ ] Upsert semantic vector to Pinecone/FAISS and store metadata in Postgres. **Owner:** Backend
* [ ] Frontend: Character create UI (upload + consent). **Owner:** Frontend

**Acceptance:** create character → semantic vector exists → metadata visible.

### Sprint 2 — Basic generation pipeline (8–12 hours)

* [ ] Implement `POST /api/v1/generate` (enqueue job). **Owner:** Backend
* [ ] Worker: retrieve semantic + episodic (if any) and call LLM template with low temp. **Owner:** Backend
* [ ] Worker: call Nano-Banana with prompt and store result in S3. **Owner:** Backend
* [ ] Implement `GET /api/v1/job/{id}` status endpoint. **Owner:** Backend

**Acceptance:** Upload char → generate 1 scene → image in S3 and job status success.

### Sprint 3 — State extraction & memory update (6–10 hours)

* [ ] Implement image state extractor: run CLIP + simple classifier or LLM vision text → tags. **Owner:** Backend
* [ ] Upsert episodic vector to VDB and record in Postgres. **Owner:** Backend

**Acceptance:** After generation, episodic state appended and visible.

### Sprint 4 — Metrics, IDR, and demo polish (6–8 hours)

* [ ] Compute IDR (face embedding similarity) and show on job result. **Owner:** Backend
* [ ] Add negative tokens enforcement, prompt sanitization. **Owner:** Backend
* [ ] Add consent & watermark option. **Owner:** Frontend + Backend
* [ ] Demo prep and 3-scene script. **Owner:** Team

**Acceptance:** Demo works end-to-end with IDR numbers printed.

### Post-hackathon (weeks)

* Add FaceRefiner loop, LLM caching, UI improvements, rate limiting, multi-character scenes, video keyframe support.

---

## 15 — Appendix: sample code snippets & env vars

### Sample FastAPI endpoints (skeleton)

```python
# app/main.py (snippet)
from fastapi import FastAPI, UploadFile, File, Depends
from pydantic import BaseModel
import uuid, os
app = FastAPI(title="IDLock API")

@app.post("/api/v1/characters")
async def create_character(name: str, files: list[UploadFile] = File(...), consent: bool = True, user=Depends(get_user)):
    char_id = "char_" + uuid.uuid4().hex[:8]
    # Save files to S3 or /tmp
    # enqueue extract_identity job
    return {"character_id": char_id}

class GenerateRequest(BaseModel):
    character_id: str
    prompt: str
    pose_image_url: str = None
    options: dict = {}

@app.post("/api/v1/generate")
async def generate(req: GenerateRequest, user=Depends(get_user)):
    job_id = enqueue_job(req)
    return {"job_id": job_id, "status": "queued"}
```

### Nano-Banana call (pseudo)

```python
import requests
NANO_API_KEY = os.getenv("NANO_KEY")
def call_nanobanana(prompt, size="1024x1024"):
    payload = {"prompt": prompt, "size": size}
    headers = {"Authorization": f"Bearer {NANO_API_KEY}"}
    r = requests.post("https://api.nanobanana.com/v2/generate", json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    return r.content  # image bytes
```

### Environment variables (suggested)

```
DATABASE_URL=postgres://...
REDIS_URL=redis://...
PINECONE_API_KEY=...
PINECONE_ENV=...
NANO_KEY=...
LLM_API_KEY=...
S3_BUCKET=...
S3_ACCESS_KEY=...
S3_SECRET_KEY=...
JWT_SECRET=...
```

---

## Acceptance criteria (MVP)

* [ ] Create character from 1–3 images → semantic vector created.
* [ ] Generate 3 scenes for the character with Nano-Banana via API.
* [ ] Each generated image saved to S3 and job status reported.
* [ ] IDR metric computed and shown for each output.
* [ ] Episodic memory updated after each generation.
* [ ] Demo: show character registry + 3 consistent outputs + IDR numbers.

---

## Final notes & immediate next actions (copy to your project board)

1. **Provision API keys**: Nano-Banana, Pinecone, OpenAI/Claude, S3. (Owner: Ops)
2. **Implement character create flow** (Owner: Backend) — highest priority.
3. **Implement generate job + worker** (Owner: Backend) — second priority.
4. **Prepare demo script** (Owner: Team) — must show 3 scenes & IDR numbers.
5. **Implement basic abuse protections** (Owner: Backend) — consent + watermark + rate-limit.

---

If you want, I can now:

* Generate a `docker-compose.yml` for the minimal dev stack (Postgres + Redis + FastAPI)
* Produce the actual worker scripts (Celery example) and a full `app/main.py` ready to paste
* Create shell scripts to seed the DB and run a sample end-to-end demo with mock Nano-Banana responses

Tell me which code artifact you want next and I’ll generate it now.
