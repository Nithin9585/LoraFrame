# 🚀 Quick Deployment Guide - Cloud Run

## ⚡ TL;DR - Deploy in 3 Steps

```bash
# 1. Setup GCP resources (one-time)
cd scripts && chmod +x setup_cloud.sh && ./setup_cloud.sh

# 2. Migrate database (one-time)
# Export your SQLite data and import to Cloud SQL
# See CLOUD_RUN_DEPLOYMENT.md for details

# 3. Deploy to Cloud Run
chmod +x deploy.sh && ./deploy.sh
```

---

## 📦 What You Need Before Starting

1. **Google Cloud Account** with billing enabled
2. **gcloud CLI** installed and authenticated
3. **Docker** installed locally
4. **API Keys** ready:
   - Gemini API Key (Google AI Studio)
   - Groq API Key (Groq Console)

---

## 🔑 Key Environment Variables

Create `.env.cloud`:

```bash
# Database (after Cloud SQL setup)
DATABASE_URL=postgresql://cineai:PASSWORD@/cineai?host=/cloudsql/PROJECT:REGION:INSTANCE

# Cloud Storage
USE_GCS=true
GCS_BUCKET_UPLOADS=cineai-uploads
GCS_BUCKET_OUTPUTS=cineai-outputs
GCP_PROJECT_ID=your-project-id

# API Keys (stored in Secret Manager)
GEMINI_API_KEY=your-gemini-key-here
GROQ_API_KEY=your-groq-key-here

# App Config
API_HOST=0.0.0.0
API_PORT=8080
```

---

## 📊 Changes Made to Your Code

| Component | Change | Status |
|-----------|--------|--------|
| Storage | Added GCS support | ✅ Complete |
| Vector DB | GCS persistence for FAISS | ✅ Complete |
| Config | Cloud settings added | ✅ Complete |
| Dockerfile | Cloud Run optimized | ✅ Complete |
| Dependencies | GCP libraries added | ✅ Complete |
| Scripts | Automation created | ✅ Complete |

---

## 🏗️ Infrastructure Created by Scripts

**`setup_cloud.sh` creates:**
- Cloud SQL PostgreSQL instance (db-f1-micro)
- GCS bucket: `cineai-uploads` (character images)
- GCS bucket: `cineai-outputs` (generated images)
- Secret Manager secrets (API keys)
- Service account with permissions

**`deploy.sh` creates:**
- Docker image in Container Registry
- Cloud Run service (`cineai-api`)
- Connects Cloud SQL + Secret Manager
- Sets up environment variables

---

## 💡 Quick Commands Reference

### Check Resources
```bash
# List Cloud Run services
gcloud run services list

# List Cloud SQL instances
gcloud sql instances list

# List GCS buckets
gsutil ls

# List secrets
gcloud secrets list
```

### View Logs
```bash
# Real-time logs
gcloud run services logs tail cineai-api --region us-central1

# Last 50 lines
gcloud run services logs read cineai-api --region us-central1 --limit=50
```

### Get Service URL
```bash
gcloud run services describe cineai-api \
  --region us-central1 \
  --format='value(status.url)'
```

### Update Environment Variable
```bash
gcloud run services update cineai-api \
  --region us-central1 \
  --set-env-vars "NEW_VAR=value"
```

### Rollback Deployment
```bash
# List revisions
gcloud run revisions list --service cineai-api

# Rollback to previous
gcloud run services update-traffic cineai-api \
  --to-revisions REVISION_NAME=100
```

---

## 🐛 Common Issues & Fixes

### ❌ Error: "Cloud SQL connection failed"
```bash
# Fix: Check connection name matches
gcloud sql instances describe cineai-db \
  --format='value(connectionName)'

# Update in deployment
gcloud run services update cineai-api \
  --set-cloudsql-instances PROJECT:REGION:INSTANCE
```

### ❌ Error: "Permission denied to bucket"
```bash
# Fix: Grant storage permissions
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:SERVICE_ACCOUNT_EMAIL" \
  --role="roles/storage.objectAdmin"
```

### ❌ Error: "Secret not accessible"
```bash
# Fix: Grant secret access
gcloud secrets add-iam-policy-binding GEMINI_API_KEY \
  --member="serviceAccount:SERVICE_ACCOUNT_EMAIL" \
  --role="roles/secretmanager.secretAccessor"
```

### ❌ Error: "Container failed to start"
```bash
# Fix: Check logs for error
gcloud run services logs read cineai-api --limit=100

# Common causes:
# 1. PORT not set correctly (should be 8080)
# 2. Database connection failed
# 3. Missing environment variables
```

---

## 💰 Cost Breakdown

| Service | Configuration | Monthly Cost |
|---------|---------------|--------------|
| Cloud Run | 1GB RAM, 100K requests | $5-10 |
| Cloud SQL | db-f1-micro, 10GB | $10-15 |
| Cloud Storage | 70GB total | $1-2 |
| Container Registry | 5GB images | <$1 |
| **Total** | | **~$20-25** |

**Save money:**
- Set Cloud Run min instances to 0
- Enable Cloud SQL auto-pause
- Add GCS lifecycle policy (delete old files after 90 days)

---

## 🔒 Security Checklist

Before going to production:

- [ ] Move API keys to Secret Manager ✅ (done by scripts)
- [ ] Enable Cloud SQL private IP
- [ ] Make GCS buckets private (use signed URLs)
- [ ] Add API authentication to endpoints
- [ ] Enable rate limiting
- [ ] Set up Cloud Armor WAF
- [ ] Configure CORS properly
- [ ] Enable HTTPS only
- [ ] Set up monitoring alerts
- [ ] Review IAM permissions (least privilege)

---

## 📈 Monitoring Setup

### Essential Metrics to Watch:
1. **Request Rate** - Track API usage
2. **Error Rate** - Catch failures
3. **Latency** - Monitor performance
4. **CPU/Memory** - Optimize resources
5. **Cloud SQL Connections** - Prevent exhaustion

### Create Alert (High Error Rate):
```bash
gcloud alpha monitoring policies create \
  --notification-channels=CHANNEL_ID \
  --display-name="CineAI Errors" \
  --condition-threshold-value=0.05 \
  --condition-threshold-duration=300s \
  --condition-display-name="Error rate > 5%"
```

---

## 🎯 Testing Your Deployment

```bash
# Get service URL
export SERVICE_URL=$(gcloud run services describe cineai-api \
  --region us-central1 --format='value(status.url)')

# 1. Health check
curl $SERVICE_URL/health

# 2. Create character
curl -X POST $SERVICE_URL/api/characters \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Cloud Test",
    "description": "Testing Cloud Run deployment"
  }'

# 3. Upload character image
# (Use your frontend or Postman)

# 4. Generate image
curl -X POST $SERVICE_URL/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "character_id": "char_xxx",
    "prompt": "portrait in a garden"
  }'

# 5. Check job status
curl $SERVICE_URL/api/jobs/job_xxx
```

---

## 🔄 CI/CD with GitHub Actions

After manual deployment works, automate it:

1. **Add GitHub secrets:**
   - `GCP_PROJECT_ID`
   - `GCP_SA_KEY` (service account JSON)
   - `GEMINI_API_KEY`
   - `GROQ_API_KEY`

2. **Push to main branch** → Auto-deploys to Cloud Run

See `CLOUD_RUN_DEPLOYMENT.md` for full GitHub Actions workflow.

---

## 📚 Full Documentation

- **Detailed Guide:** `CLOUD_RUN_DEPLOYMENT.md`
- **Changes Summary:** `CLOUD_DEPLOYMENT_SUMMARY.md`
- **Setup Script:** `scripts/setup_cloud.sh`
- **Deploy Script:** `scripts/deploy.sh`

---

## 🎉 Ready to Deploy?

```bash
# Navigate to your project
cd cineAI

# Run setup (creates all GCP resources)
./scripts/setup_cloud.sh

# Follow the prompts, then deploy
./scripts/deploy.sh

# Your API will be live at:
# https://cineai-api-xxx-uc.a.run.app
```

**Need help?** Check the troubleshooting sections in the full documentation.

Good luck! 🚀
