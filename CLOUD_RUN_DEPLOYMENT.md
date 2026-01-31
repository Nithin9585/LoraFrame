# Google Cloud Run Deployment Guide

## 🚀 Complete Cloud API Setup for CineAI

This guide covers everything needed to deploy your CineAI character memory system to Google Cloud Run with full cloud infrastructure.

---

## 📋 Prerequisites

1. **Google Cloud Project** with billing enabled
2. **gcloud CLI** installed and authenticated
3. **Docker** installed locally
4. **API Keys**:
   - Gemini API Key (Google AI Studio)
   - Groq API Key (groq.com)
   - Pinecone API Key (pinecone.io)

---

## 🔧 Required Changes for Cloud Deployment

### 1. Database: SQLite → Cloud SQL (PostgreSQL)

**Why:** SQLite is file-based and doesn't persist in Cloud Run (ephemeral storage)

**Setup Cloud SQL:**
```bash
# Create Cloud SQL instance
gcloud sql instances create cineai-db \
  --database-version=POSTGRES_15 \
  --tier=db-f1-micro \
  --region=us-central1

# Create database
gcloud sql databases create cineai --instance=cineai-db

# Create user
gcloud sql users create cineai-user \
  --instance=cineai-db \
  --password=YOUR_SECURE_PASSWORD

# Get connection name
gcloud sql instances describe cineai-db --format='value(connectionName)'
# Output: PROJECT_ID:us-central1:cineai-db
```

**Update `config.py`:**
```python
# For Cloud Run, use Cloud SQL Proxy connection
DATABASE_URL: str = "postgresql+psycopg2://cineai-user:PASSWORD@/cineai?host=/cloudsql/PROJECT_ID:us-central1:cineai-db"
```

---

### 2. Storage: Local Files → Google Cloud Storage

**Why:** Cloud Run instances are ephemeral - files don't persist between deploys

**Setup Cloud Storage:**
```bash
# Create bucket for uploads
gsutil mb -l us-central1 gs://cineai-uploads

# Create bucket for outputs
gsutil mb -l us-central1 gs://cineai-outputs

# Set public read access for outputs (if needed)
gsutil iam ch allUsers:objectViewer gs://cineai-outputs
```

**Update `config.py`:**
```python
# Google Cloud Storage settings
GCS_BUCKET_UPLOADS: str = "cineai-uploads"
GCS_BUCKET_OUTPUTS: str = "cineai-outputs"
USE_LOCAL_STORAGE: bool = False
USE_GCS: bool = True
```

**Update `storage.py`** to use GCS:
```python
from google.cloud import storage

class StorageService:
    def __init__(self):
        self.use_gcs = settings.USE_GCS
        
        if self.use_gcs:
            self.client = storage.Client()
            self.bucket_uploads = self.client.bucket(settings.GCS_BUCKET_UPLOADS)
            self.bucket_outputs = self.client.bucket(settings.GCS_BUCKET_OUTPUTS)
        else:
            # Local fallback
            self.base_path = Path(settings.LOCAL_STORAGE_PATH)
    
    async def upload_bytes(self, data: bytes, path: str, content_type: str = "image/jpeg") -> str:
        """Upload to GCS and return public URL."""
        if self.use_gcs:
            # Determine bucket based on path
            if path.startswith("uploads/"):
                bucket = self.bucket_uploads
            else:
                bucket = self.bucket_outputs
            
            blob = bucket.blob(path)
            blob.upload_from_string(data, content_type=content_type)
            
            # Return public URL
            return f"https://storage.googleapis.com/{bucket.name}/{path}"
        else:
            # Local fallback
            return await self._upload_local(data, path)
```

---

### 3. InsightFace Models: Local Files → GCS

**Why:** Model files (~1GB) in `models/` directory won't persist in Cloud Run

**Options:**

**Option A: Bundle in Docker Image (Recommended for Cloud Run)**
```dockerfile
# In Dockerfile, copy models
COPY models/ ./models/
```

**Option B: Download from GCS at startup**
```bash
# Upload models to GCS
gsutil -m cp -r models/ gs://cineai-uploads/models/

# In Dockerfile, download at startup
RUN gsutil -m cp -r gs://cineai-uploads/models/ ./models/
```

**Option C: Use Cloud Storage FUSE (more complex)**
```dockerfile
# Mount GCS bucket as filesystem
RUN gcsfuse cineai-uploads /app/models
```

---

### 4. Vector Database: Local FAISS → Keep FAISS or Use Pinecone

**Option A: Keep FAISS with GCS Persistence (Recommended)**

Store FAISS index files in GCS and load on startup:

```python
# In vectordb.py
from google.cloud import storage

class LocalVectorStore:
    def __init__(self):
        self.storage_path = Path("./vector_store")
        self.gcs_bucket = "cineai-uploads"
        self.gcs_path = "vector_store/"
        
        # Download from GCS on startup
        self._download_from_gcs()
        self._load()
    
    def _download_from_gcs(self):
        """Download FAISS index from GCS."""
        client = storage.Client()
        bucket = client.bucket(self.gcs_bucket)
        
        for blob in bucket.list_blobs(prefix=self.gcs_path):
            local_path = self.storage_path / blob.name.replace(self.gcs_path, "")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(local_path)
    
    def _upload_to_gcs(self):
        """Upload FAISS index to GCS."""
        client = storage.Client()
        bucket = client.bucket(self.gcs_bucket)
        
        for file_path in self.storage_path.rglob("*"):
            if file_path.is_file():
                blob_name = f"{self.gcs_path}{file_path.relative_to(self.storage_path)}"
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(file_path)
    
    def _save(self):
        """Save and sync to GCS."""
        # Save locally first
        super()._save()
        # Upload to GCS
        self._upload_to_gcs()
```

**Option B: Use Pinecone (Fully Managed)**

Already configured! Just set `PINECONE_API_KEY` and remove local FAISS:

```python
# In vectordb.py - your code already supports this
use_pinecone = bool(settings.PINECONE_API_KEY)
```

---

### 5. Secrets Management: .env → Google Secret Manager

**Why:** Don't hardcode API keys in code or environment variables

**Setup Secrets:**
```bash
# Create secrets
echo -n "your_gemini_key" | gcloud secrets create gemini-api-key --data-file=-
echo -n "your_groq_key" | gcloud secrets create groq-api-key --data-file=-
echo -n "your_pinecone_key" | gcloud secrets create pinecone-api-key --data-file=-
echo -n "your_db_password" | gcloud secrets create db-password --data-file=-

# Grant Cloud Run access
gcloud secrets add-iam-policy-binding gemini-api-key \
  --member=serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com \
  --role=roles/secretmanager.secretAccessor
```

**Access in Cloud Run:**
```yaml
# In cloud-run-deploy.yaml
env:
  - name: GEMINI_API_KEY
    valueFrom:
      secretKeyRef:
        name: gemini-api-key
        key: latest
```

---

### 6. Update Dockerfile for Cloud Run

**Optimized Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Google Cloud Storage library
RUN pip install google-cloud-storage google-cloud-secret-manager

# Copy application
COPY app/ ./app/
COPY models/ ./models/

# Cloud Run uses PORT environment variable
ENV PORT=8080

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

# Run uvicorn with dynamic port
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
```

---

### 7. Add Requirements for Cloud

**Update `requirements.txt`:**
```txt
# Existing dependencies...

# Google Cloud
google-cloud-storage==2.14.0
google-cloud-secret-manager==2.18.0
google-cloud-sql-connector==1.7.0
psycopg2-binary==2.9.9

# Production server
gunicorn==21.2.0
```

---

### 8. Update Environment Configuration

**Create `config.py` cloud settings:**
```python
class Settings(BaseSettings):
    # ... existing settings ...
    
    # Cloud-specific settings
    CLOUD_RUN: bool = False  # Auto-detect in __init__
    GCP_PROJECT_ID: str = ""
    
    # Google Cloud Storage
    GCS_BUCKET_UPLOADS: str = "cineai-uploads"
    GCS_BUCKET_OUTPUTS: str = "cineai-outputs"
    USE_GCS: bool = False
    
    # Cloud SQL
    CLOUD_SQL_CONNECTION_NAME: str = ""  # PROJECT_ID:REGION:INSTANCE
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Auto-detect Cloud Run
        if os.getenv("K_SERVICE"):  # Cloud Run env variable
            self.CLOUD_RUN = True
            self.USE_GCS = True
            self.DEBUG = False
```

---

## 🚢 Deployment Steps

### Step 1: Build and Push Docker Image

```bash
# Set variables
export PROJECT_ID=your-gcp-project-id
export REGION=us-central1
export SERVICE_NAME=cineai-api

# Configure Docker for GCP
gcloud auth configure-docker

# Build image
docker build -t gcr.io/$PROJECT_ID/$SERVICE_NAME:latest .

# Push to Container Registry
docker push gcr.io/$PROJECT_ID/$SERVICE_NAME:latest
```

### Step 2: Deploy to Cloud Run

```bash
# Deploy with all configurations
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME:latest \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10 \
  --min-instances 1 \
  --concurrency 80 \
  --add-cloudsql-instances PROJECT_ID:us-central1:cineai-db \
  --set-env-vars "DATABASE_URL=postgresql+psycopg2://cineai-user:PASSWORD@/cineai?host=/cloudsql/PROJECT_ID:us-central1:cineai-db" \
  --set-env-vars "USE_GCS=true" \
  --set-env-vars "GCS_BUCKET_UPLOADS=cineai-uploads" \
  --set-env-vars "GCS_BUCKET_OUTPUTS=cineai-outputs" \
  --set-secrets "GEMINI_API_KEY=gemini-api-key:latest" \
  --set-secrets "GROQ_API_KEY=groq-api-key:latest" \
  --set-secrets "PINECONE_API_KEY=pinecone-api-key:latest"

# Get the deployed URL
gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)'
```

### Step 3: Run Database Migrations

```bash
# Connect to Cloud SQL and run migrations
gcloud sql connect cineai-db --user=cineai-user

# In psql:
CREATE TABLE IF NOT EXISTS characters (...);
CREATE TABLE IF NOT EXISTS jobs (...);
CREATE TABLE IF NOT EXISTS episodic_states (...);
```

Or use Alembic:
```bash
# From local machine with Cloud SQL Proxy
gcloud run jobs execute migrate-db \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME:latest \
  --command "alembic upgrade head"
```

### Step 4: Test Deployment

```bash
# Get service URL
export SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)')

# Test health endpoint
curl $SERVICE_URL/health

# Test character creation
curl -X POST $SERVICE_URL/api/characters \
  -F "name=TestChar" \
  -F "description=Test" \
  -F "consent=true" \
  -F "files=@test_image.jpg"
```

---

## 📊 Cost Optimization

### Cloud Run Pricing (as of 2026)
- **CPU**: $0.00002400 per vCPU-second
- **Memory**: $0.00000250 per GiB-second
- **Requests**: $0.40 per million requests
- **Free tier**: 2 million requests/month

### Recommendations:
1. **Use `min-instances=1`** for faster cold starts (costs ~$10/month)
2. **Use `max-instances=10`** to limit costs
3. **Set `concurrency=80`** to handle multiple requests per instance
4. **Use Cloud SQL `db-f1-micro`** ($7.67/month)
5. **Use Pinecone** free tier (1M vectors) instead of managing FAISS

### Estimated Monthly Cost:
- Cloud Run (1 instance): **~$10-15**
- Cloud SQL (db-f1-micro): **~$7.67**
- Cloud Storage (100GB): **~$2**
- Pinecone (free tier): **$0**
- **Total: ~$20-25/month**

---

## 🔒 Security Best Practices

1. **API Authentication** (add to `main.py`):
```python
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != settings.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return credentials.credentials

# Protect routes
@app.post("/api/characters")
async def create_character(..., token: str = Depends(verify_token)):
```

2. **CORS Configuration**:
```python
# Update config.py
CORS_ORIGINS: List[str] = [
    "https://your-frontend.com",
    "https://your-frontend.vercel.app"
]
```

3. **Rate Limiting**:
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/api/generate")
@limiter.limit("10/minute")
async def generate_image(...):
```

---

## 🔄 CI/CD with GitHub Actions

**Create `.github/workflows/deploy.yml`:**
```yaml
name: Deploy to Cloud Run

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Cloud SDK
        uses: google-github-actions/setup-gcloud@v1
        with:
          service_account_key: ${{ secrets.GCP_SA_KEY }}
          project_id: ${{ secrets.GCP_PROJECT_ID }}
      
      - name: Configure Docker
        run: gcloud auth configure-docker
      
      - name: Build image
        run: docker build -t gcr.io/${{ secrets.GCP_PROJECT_ID }}/cineai-api:${{ github.sha }} .
      
      - name: Push image
        run: docker push gcr.io/${{ secrets.GCP_PROJECT_ID }}/cineai-api:${{ github.sha }}
      
      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy cineai-api \
            --image gcr.io/${{ secrets.GCP_PROJECT_ID }}/cineai-api:${{ github.sha }} \
            --region us-central1 \
            --platform managed
```

---

## 📝 Summary of Required Changes

### Critical Changes:
1. ✅ **Database**: Switch from SQLite to Cloud SQL PostgreSQL
2. ✅ **Storage**: Switch from local files to Google Cloud Storage
3. ✅ **Secrets**: Move API keys to Secret Manager
4. ✅ **Dockerfile**: Add GCS libraries, update port handling
5. ✅ **Config**: Add GCS and Cloud SQL settings

### Optional Optimizations:
6. ⚠️ **Vector DB**: Keep FAISS with GCS or use Pinecone cloud
7. ⚠️ **Models**: Bundle in Docker or download from GCS
8. ⚠️ **Auth**: Add API key authentication
9. ⚠️ **Rate Limiting**: Add request rate limits
10. ⚠️ **CI/CD**: Setup GitHub Actions for auto-deploy

---

## 🚀 Quick Start Commands

```bash
# 1. Set up GCP
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com sqladmin.googleapis.com storage-api.googleapis.com

# 2. Create resources
./scripts/setup_cloud.sh  # We'll create this

# 3. Deploy
./scripts/deploy.sh  # We'll create this

# 4. Test
curl https://cineai-api-XXXXX.run.app/health
```

---

**Next Steps:**
1. I can create the helper scripts (`setup_cloud.sh`, `deploy.sh`)
2. I can update the code files with GCS support
3. I can create the GitHub Actions workflow

Would you like me to implement these changes now?
