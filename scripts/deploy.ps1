# CineAI Cloud Run Deployment Script for Windows PowerShell
# This script builds and deploys your application to Google Cloud Run

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  CineAI - Cloud Run Deployment" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check if gcloud is installed
try {
    gcloud --version | Out-Null
} catch {
    Write-Host "ERROR: gcloud CLI not found!" -ForegroundColor Red
    Write-Host "Install from: https://cloud.google.com/sdk/docs/install" -ForegroundColor Yellow
    exit 1
}

# Check if .env.cloud exists
if (-not (Test-Path "..\..\.env.cloud")) {
    Write-Host "ERROR: .env.cloud file not found!" -ForegroundColor Red
    Write-Host "Run .\scripts\setup_cloud.ps1 first to create GCP resources" -ForegroundColor Yellow
    exit 1
}

# Load configuration
Write-Host "Loading configuration from .env.cloud..." -ForegroundColor Yellow
Get-Content "..\..\.env.cloud" | ForEach-Object {
    if ($_ -match "^([^=]+)=(.+)$") {
        $name = $matches[1].Trim()
        $value = $matches[2].Trim()
        Set-Variable -Name $name -Value $value -Scope Script
    }
}

# Get project ID
$PROJECT_ID = $GCP_PROJECT_ID
if ([string]::IsNullOrWhiteSpace($PROJECT_ID)) {
    Write-Host "Enter your GCP Project ID: " -ForegroundColor Yellow -NoNewline
    $PROJECT_ID = Read-Host
}

# Set project
Write-Host "Setting GCP project to: $PROJECT_ID" -ForegroundColor Green
gcloud config set project $PROJECT_ID

# Get region
Write-Host "`nEnter deployment region (default: us-central1): " -ForegroundColor Yellow -NoNewline
$REGION = Read-Host
if ([string]::IsNullOrWhiteSpace($REGION)) {
    $REGION = "us-central1"
}

$SERVICE_NAME = "cineai-api"
$IMAGE_NAME = "gcr.io/$PROJECT_ID/$SERVICE_NAME"

Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host "  Step 1: Building Docker Image" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

Write-Host "Building Docker image..." -ForegroundColor Yellow
Set-Location ..\..
docker build -t $IMAGE_NAME .

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR: Docker build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host "  Step 2: Pushing to Container Registry" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

Write-Host "Configuring Docker authentication..." -ForegroundColor Yellow
gcloud auth configure-docker

Write-Host "`nPushing image to GCR..." -ForegroundColor Yellow
docker push $IMAGE_NAME

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR: Docker push failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host "  Step 3: Deploying to Cloud Run" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

Write-Host "Deploying $SERVICE_NAME to Cloud Run..." -ForegroundColor Yellow

gcloud run deploy $SERVICE_NAME `
    --image=$IMAGE_NAME `
    --region=$REGION `
    --platform=managed `
    --allow-unauthenticated `
    --set-cloudsql-instances=$CLOUD_SQL_CONNECTION_NAME `
    --set-env-vars="USE_GCS=true,GCS_BUCKET_UPLOADS=$GCS_BUCKET_UPLOADS,GCS_BUCKET_OUTPUTS=$GCS_BUCKET_OUTPUTS,GCP_PROJECT_ID=$PROJECT_ID,CLOUD_SQL_CONNECTION_NAME=$CLOUD_SQL_CONNECTION_NAME,API_HOST=0.0.0.0,API_PORT=8080,CORS_ORIGINS=[`"*`"]" `
    --set-secrets="GEMINI_API_KEY=GEMINI_API_KEY:latest,GROQ_API_KEY=GROQ_API_KEY:latest,DATABASE_URL=DATABASE_URL:latest" `
    --cpu=2 `
    --memory=2Gi `
    --min-instances=0 `
    --max-instances=10 `
    --concurrency=80 `
    --timeout=300

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR: Cloud Run deployment failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host "  Step 4: Getting Service URL" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

$SERVICE_URL = (gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")

Write-Host "`n================================================" -ForegroundColor Green
Write-Host "  ✅ Deployment Complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Service URL: $SERVICE_URL" -ForegroundColor Cyan
Write-Host ""
Write-Host "Test your deployment:" -ForegroundColor Yellow
Write-Host "  Health Check:" -ForegroundColor White
Write-Host "    curl $SERVICE_URL/health" -ForegroundColor Gray
Write-Host ""
Write-Host "  API Documentation:" -ForegroundColor White
Write-Host "    $SERVICE_URL/docs" -ForegroundColor Gray
Write-Host ""
Write-Host "  View Logs:" -ForegroundColor White
Write-Host "    gcloud run services logs read $SERVICE_NAME --region=$REGION" -ForegroundColor Gray
Write-Host ""
Write-Host "  View Metrics:" -ForegroundColor White
Write-Host "    Cloud Console -> Cloud Run -> $SERVICE_NAME" -ForegroundColor Gray
Write-Host ""

# Save service URL to file
$SERVICE_URL | Out-File -FilePath "..\..\.service_url" -Encoding utf8
Write-Host "Service URL saved to .service_url" -ForegroundColor Green
Write-Host ""
