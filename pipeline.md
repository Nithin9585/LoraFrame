# 🏗️ CINEAI ARCHITECTURE (WITH LoRA INTEGRATION)

## 📊 FULL SYSTEM DIAGRAM

```mermaid
graph TD

%% User Layer
A[User / Client App] --> B[API Gateway]

%% Orchestration
B --> C[Job Queue (Redis RQ)]

%% Workers
C --> D[Generator Worker]
C --> E[Refiner Worker]
C --> F[State Analyzer Worker]
C --> L[LoRA Trainer Worker]

%% Memory Layer
D --> G[Postgres Metadata DB]
D --> H[Vector DB (Character Memory)]

%% LLM Layer
D --> I[Groq LLM Prompt Engine]

%% LoRA System
D --> J[LoRA Registry]
J -->|Load Active LoRA| D

%% Generation Layer
D --> K[Gemini / Imagen / Veo Models]

%% Validation
K --> M[Face & IDR Validator]

%% Refinement
M -->|Fail| E
E --> K

%% State Update
M -->|Pass| F
F --> H
F --> G

%% LoRA Data Loop
F --> N[LoRA Dataset Builder]
N --> L
L --> J

%% Output
M --> O[Final Image / Video Output]
```

## 🧩 COMPONENT BREAKDOWN

| Block | Technical Implementation | Purpose |
|-------|--------------------------|---------|
| **API Gateway** | `app.main` (FastAPI) | Accepts user requests, validates inputs |
| **Redis Queue** | `Redis RQ` | Manages async job distribution |
| **Generator** | `app.workers.generator` | Orchestrates content creation |
| **Refiner** | `app.workers.refiner` | Fixes identity drift (Auto-repair) |
| **State Analyzer** | `app.workers.state` | Updates episodic memory from outputs |
| **VectorDB** | `app.services.vectordb` | Stores visual history (Semantic/Episodic) |
| **Groq LLM** | `app.services.groq_llm` | Context-aware prompt engineering |
| **Gemini/Veo** | `app.services.gemini` / `veo_video` | Generates high-fidelity media |
| **Validator** | `InsightFace` (IDR) | Checks identity consistency (0-1 score) |
| **LoRA Registry** | *New Component* | Stores and manages trained character models |
| **LoRA Trainer** | *New Component* | Fine-tunes custom models on character data |

---

## 🔁 COMPLETE PIPELINE FLOW

### 🟢 STEP 1 — User Request
**Input:** "Generate scene where Kairo fights in rain"
**Data:** `character_id`, `pose`, `media_type` (video/image)
- Request hits the **API Gateway**.

### 🟢 STEP 2 — API + Job Queue
- **API**: Validates request & creates `job_id`.
- **Queue**: Pushes job to **Redis Queue**.
- *Why?* Enables massive parallel concurrency.

### 🟢 STEP 3 — Generator Worker Starts
**Code:** `app.workers.generator`
The worker wakes up and fetches context:
- **Postgres**: Character profile (Core traits).
- **VectorDB**: Last 3 scenes, current outfit, active injuries.
- *Result:* Builds a complete "World State" context.

### 🟢 STEP 4 — LoRA Check
- **Query**: "Do we have a trained LoRA for this character?"
- **If YES**: Load `char_kairo_v2.safetensors` (Identity Lock).
- **If NO**: Continue with standard prompting.

### 🟢 STEP 5 — Prompt Engineering (LLM)
**Code:** `app.services.groq_llm`
Context is sent to **Groq**.
- **Input**: User prompt + Memory + Style rules.
- **Output**: Optimized cinematic prompt.
  - *Example:* "Maintain amber cyber eye, left cheek scar visible, dark cloak..."
- *Benefit:* Prevents "catastrophic forgetting" of character details.

### 🟢 STEP 6 — Content Generation
The Generator dispatches to the appropriate model:
- **Image**: Gemini / Imagen 3
- **Video**: Google Veo 3.1 (`app.services.veo_video`)
- **Inputs**: Prompt, Reference Images, LoRA weights.

### 🟢 STEP 7 — Identity Validation (IDR)
**Code:** `app.workers.extractor` logic
The system analyzes the raw output:
- **Checks**: Face embedding, scar detection, hair/eye color.
- **Calculates**: **IDR** (Identity Retention Score) range 0.0 → 1.0.

### 🟢 STEP 8 — Decision Gate
- **If IDR ≥ 0.75** → **PASS** (Go to Step 10).
- **If IDR < 0.75** → **FAIL** (Go to Step 9).

### 🟢 STEP 9 — Refiner Worker (Auto-Repair)
**Code:** `app.workers.refiner`
Triggered on failure:
1.  **Crop** face region.
2.  **Inpaint** using Gemini with strict identity prompt.
3.  **Blend** back into original.
4.  **Recheck** IDR. (Repeat max 2 times).

### 🟢 STEP 10 — State Analyzer (Memory Update)
**Code:** `app.workers.state`
On success, the system scans the final image:
- **Extracts**: Clothes, Props, Injuries, Pose, Environment.
- **Creates**: New **Episodic Memory** entry.
- **Stores**: Saves to **VectorDB** for next turn continuity.

### 🟢 STEP 11 — LoRA Dataset Collection
- **Condition**: If IDR is very high (>0.85).
- **Action**: Add image to **LoRA Training Set**.
- *Goal:* Only "Golden" images are used for training data. No noise.

### 🟢 STEP 12 — Dataset Builder
- Background process aligns, resizes, cleans, and deduplicates the accumulated training images.

### 🟢 STEP 13 — LoRA Training
- **Trigger**: When data > 30 images.
- **Action**: Run PEFT + Diffusers training.
- **Output**: `char_kairo_v1.safetensors`.
- *Learns*: Face structure, scars, style, body proportions.

### 🟢 STEP 14 — LoRA Validation
- System generates test scenes with new model.
- Compares against baseline IDR.
- **Decision**: Approve if better than baseline; Discard if regressed.

### 🟢 STEP 15 — LoRA Registry Update
- Registry saves version, accuracy score, and path.
- Marks model as **Active**.

### 🟢 STEP 16 — Auto Deployment
- Next generation job (Step 4) automatically loads the new LoRA.
- **Result**: Character is now permanently consistent.

---

## 🔁 CONTINUOUS LEARNING LOOP
> **Generate → Validate → Learn → Improve** (Forever)

## 🧠 WHY THIS ARCHITECTURE IS SPECIAL

| Feature | Competitors | CineAI |
| :--- | :--- | :--- |
| **Generation** | Generate & Forget | **Remembers Everything** |
| **Consistency** | Random Drift | **Permanent Identity (LoRA)** |
| **Quality** | Manual Fixes | **Self-Healing (Refiner)** |
| **Evolution** | Static Models | **Self-Improving** |

## 🎤 ONE-LINE PITCH
> "We combine memory, LLM reasoning, visual validation, and LoRA training to turn AI characters into permanent digital actors."

## ✅ FINAL SUMMARY

| Feature | Status |
| :--- | :--- |
| **Character Memory** | ✅ Implemented |
| **Drift Repair** | ✅ Implemented |
| **Permanent Identity** | 🚧 LoRA Integration (In Progress) |
| **Self Learning** | 🚧 Planned |
| **Scalability** | ✅ Ready |
