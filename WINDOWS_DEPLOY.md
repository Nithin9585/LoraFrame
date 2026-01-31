# 🪟 Windows Deployment Guide

## ⚠️ You're on Windows!

The bash scripts (`.sh` files) won't work in PowerShell. Use the PowerShell versions instead:

---

## ✅ Quick Start (Windows)

### Step 1: Navigate to scripts directory
```powershell
cd scripts
```

### Step 2: Run setup script
```powershell
.\setup_cloud.ps1
```

### Step 3: Deploy to Cloud Run
```powershell
.\deploy.ps1
```

---

## 📋 What You Need

1. **gcloud CLI** - [Download](https://cloud.google.com/sdk/docs/install)
2. **Docker Desktop** - [Download](https://www.docker.com/products/docker-desktop)
3. **PowerShell 5.1+** (already installed on Windows)

---

## 🔧 Installation Steps

### 1. Install gcloud CLI

Download and run the installer:
https://cloud.google.com/sdk/docs/install#windows

After installation:
```powershell
# Authenticate
gcloud auth login

# Set your project
gcloud config set project YOUR_PROJECT_ID
```

### 2. Install Docker Desktop

Download from: https://www.docker.com/products/docker-desktop

After installation, make sure Docker is running (check system tray).

---

## 🚀 Deployment Commands

### First Time Setup (creates all GCP resources)

```powershell
# Navigate to your project
cd C:\Users\Nithi\OneDrive\Pictures\Desktop\INVITRO\cineAI

# Run setup script
.\scripts\setup_cloud.ps1
```

**This will:**
- ✅ Create Cloud SQL PostgreSQL instance
- ✅ Create GCS buckets
- ✅ Set up Secret Manager
- ✅ Configure IAM permissions
- ✅ Generate `.env.cloud` configuration

**You'll be prompted for:**
- GCP Project ID
- Region (default: us-central1)
- Database password
- Gemini API Key
- Groq API Key

---

### Deploy to Cloud Run

```powershell
# Make sure you're in the project root
cd C:\Users\Nithi\OneDrive\Pictures\Desktop\INVITRO\cineAI

# Run deployment script
.\scripts\deploy.ps1
```

**This will:**
- ✅ Build Docker image
- ✅ Push to Google Container Registry
- ✅ Deploy to Cloud Run
- ✅ Connect Cloud SQL
- ✅ Configure secrets

---

## 🐛 Troubleshooting

### Error: "gcloud not found"
**Solution:**
```powershell
# Install gcloud CLI
# Download from: https://cloud.google.com/sdk/docs/install#windows

# After installation, restart PowerShell
```

### Error: "Docker not found"
**Solution:**
```powershell
# Make sure Docker Desktop is running
# Check system tray for Docker icon
```

### Error: "Execution Policy" (script won't run)
**Solution:**
```powershell
# Allow script execution (run PowerShell as Administrator)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then try again
.\scripts\setup_cloud.ps1
```

### Error: "Cannot find path"
**Solution:**
```powershell
# Make sure you're in the cineAI directory
cd C:\Users\Nithi\OneDrive\Pictures\Desktop\INVITRO\cineAI

# Check current directory
pwd

# Should show: C:\Users\Nithi\OneDrive\Pictures\Desktop\INVITRO\cineAI
```

---

## 📊 After Deployment

### Get your service URL
```powershell
gcloud run services describe cineai-api --region=us-central1 --format="value(status.url)"
```

### Test health endpoint
```powershell
$url = gcloud run services describe cineai-api --region=us-central1 --format="value(status.url)"
curl "$url/health"
```

### View logs
```powershell
gcloud run services logs read cineai-api --region=us-central1 --limit=50
```

### Real-time logs
```powershell
gcloud run services logs tail cineai-api --region=us-central1
```

---

## 🔄 Alternative: Use Git Bash

If you prefer bash commands, you can use **Git Bash** (comes with Git for Windows):

1. Right-click in your project folder
2. Select "Git Bash Here"
3. Run the bash scripts:
   ```bash
   cd scripts
   chmod +x setup_cloud.sh
   ./setup_cloud.sh
   ```

---

## 📝 Files Created

After running `setup_cloud.ps1`, you'll have:

- ✅ `.env.cloud` - Cloud configuration
- ✅ Cloud SQL instance in GCP
- ✅ GCS buckets created
- ✅ Secrets in Secret Manager

After running `deploy.ps1`, you'll have:

- ✅ `.service_url` - Your Cloud Run URL
- ✅ Docker image in Container Registry
- ✅ Running Cloud Run service

---

## 💡 Quick Commands Reference

```powershell
# Check gcloud version
gcloud --version

# Check Docker version
docker --version

# List Cloud Run services
gcloud run services list

# List Cloud SQL instances
gcloud sql instances list

# List GCS buckets
gsutil ls

# Check current directory
pwd

# List files
ls

# View file content
Get-Content .env.cloud
```

---

## ✅ Summary

**For Windows PowerShell, use:**
- ✅ `.\scripts\setup_cloud.ps1` (NOT setup_cloud.sh)
- ✅ `.\scripts\deploy.ps1` (NOT deploy.sh)
- ✅ Use `.\` instead of `./` for scripts
- ✅ No need for `chmod` on Windows

**Your next command:**
```powershell
.\scripts\setup_cloud.ps1
```

Good luck! 🚀
