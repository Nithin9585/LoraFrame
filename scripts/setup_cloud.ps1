# CineAI Cloud Setup Script for Windows PowerShell
# This script sets up all required Google Cloud Platform resources

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  CineAI - Google Cloud Platform Setup" -ForegroundColor Cyan
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

# Get project ID
Write-Host "Enter your GCP Project ID: " -ForegroundColor Yellow -NoNewline
$PROJECT_ID = Read-Host
if ([string]::IsNullOrWhiteSpace($PROJECT_ID)) {
    Write-Host "ERROR: Project ID is required!" -ForegroundColor Red
    exit 1
}

# Set project
Write-Host "`nSetting GCP project to: $PROJECT_ID" -ForegroundColor Green
gcloud config set project $PROJECT_ID

# Get region
Write-Host "`nEnter your preferred region (default: us-central1): " -ForegroundColor Yellow -NoNewline
$REGION = Read-Host
if ([string]::IsNullOrWhiteSpace($REGION)) {
    $REGION = "us-central1"
}

Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host "  Step 1: Enabling Required APIs" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

$apis = @(
    "run.googleapis.com",
    "sqladmin.googleapis.com",
    "storage.googleapis.com",
    "secretmanager.googleapis.com",
    "containerregistry.googleapis.com"
)

foreach ($api in $apis) {
    Write-Host "Enabling $api..." -ForegroundColor Yellow
    gcloud services enable $api
}

Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host "  Step 2: Creating Cloud SQL Instance" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

$DB_INSTANCE = "cineai-db"
$DB_NAME = "cineai"
$DB_USER = "cineai"

Write-Host "Enter database password: " -ForegroundColor Yellow -NoNewline
$DB_PASSWORD = Read-Host -AsSecureString
$DB_PASSWORD_TEXT = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($DB_PASSWORD))

Write-Host "`nCreating Cloud SQL PostgreSQL instance (this takes ~5 minutes)..." -ForegroundColor Yellow
gcloud sql instances create $DB_INSTANCE `
    --database-version=POSTGRES_14 `
    --tier=db-f1-micro `
    --region=$REGION `
    --storage-size=10GB `
    --storage-type=SSD `
    --storage-auto-increase

Write-Host "`nSetting database password..." -ForegroundColor Yellow
gcloud sql users set-password postgres `
    --instance=$DB_INSTANCE `
    --password=$DB_PASSWORD_TEXT

Write-Host "`nCreating database user..." -ForegroundColor Yellow
gcloud sql users create $DB_USER `
    --instance=$DB_INSTANCE `
    --password=$DB_PASSWORD_TEXT

Write-Host "`nCreating database..." -ForegroundColor Yellow
gcloud sql databases create $DB_NAME `
    --instance=$DB_INSTANCE

Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host "  Step 3: Creating Cloud Storage Buckets" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

$BUCKET_UPLOADS = "$PROJECT_ID-cineai-uploads"
$BUCKET_OUTPUTS = "$PROJECT_ID-cineai-outputs"

Write-Host "Creating uploads bucket: $BUCKET_UPLOADS" -ForegroundColor Yellow
gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$BUCKET_UPLOADS/

Write-Host "Creating outputs bucket: $BUCKET_OUTPUTS" -ForegroundColor Yellow
gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$BUCKET_OUTPUTS/

Write-Host "`nSetting bucket lifecycle policies (delete old files after 90 days)..." -ForegroundColor Yellow
$lifecycle = @"
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {"age": 90}
      }
    ]
  }
}
"@
$lifecycle | Out-File -FilePath "lifecycle.json" -Encoding utf8
gsutil lifecycle set lifecycle.json gs://$BUCKET_OUTPUTS/
Remove-Item lifecycle.json

Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host "  Step 4: Setting Up Secret Manager" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

Write-Host "Enter your Gemini API Key: " -ForegroundColor Yellow -NoNewline
$GEMINI_KEY = Read-Host -AsSecureString
$GEMINI_KEY_TEXT = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($GEMINI_KEY))

Write-Host "Enter your Groq API Key: " -ForegroundColor Yellow -NoNewline
$GROQ_KEY = Read-Host -AsSecureString
$GROQ_KEY_TEXT = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($GROQ_KEY))

Write-Host "`nCreating secrets..." -ForegroundColor Yellow
echo $GEMINI_KEY_TEXT | gcloud secrets create GEMINI_API_KEY --data-file=-
echo $GROQ_KEY_TEXT | gcloud secrets create GROQ_API_KEY --data-file=-

Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host "  Step 5: Configuring IAM Permissions" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Get project number
$PROJECT_NUMBER = (gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
$SERVICE_ACCOUNT = "$PROJECT_NUMBER-compute@developer.gserviceaccount.com"

Write-Host "Granting permissions to: $SERVICE_ACCOUNT" -ForegroundColor Yellow

# Grant Cloud SQL Client role
gcloud projects add-iam-policy-binding $PROJECT_ID `
    --member="serviceAccount:$SERVICE_ACCOUNT" `
    --role="roles/cloudsql.client"

# Grant Storage Admin role
gcloud projects add-iam-policy-binding $PROJECT_ID `
    --member="serviceAccount:$SERVICE_ACCOUNT" `
    --role="roles/storage.objectAdmin"

# Grant Secret Manager Accessor role
gcloud secrets add-iam-policy-binding GEMINI_API_KEY `
    --member="serviceAccount:$SERVICE_ACCOUNT" `
    --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding GROQ_API_KEY `
    --member="serviceAccount:$SERVICE_ACCOUNT" `
    --role="roles/secretmanager.secretAccessor"

Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host "  Step 6: Generating Configuration File" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

$CONNECTION_NAME = "$PROJECT_ID:$REGION:$DB_INSTANCE"

$envContent = @"
# CineAI Cloud Environment Configuration
# Generated on $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

# Database Configuration
DATABASE_URL=postgresql://${DB_USER}:${DB_PASSWORD_TEXT}@/${DB_NAME}?host=/cloudsql/${CONNECTION_NAME}
CLOUD_SQL_CONNECTION_NAME=${CONNECTION_NAME}

# Google Cloud Storage
USE_GCS=true
GCS_BUCKET_UPLOADS=${BUCKET_UPLOADS}
GCS_BUCKET_OUTPUTS=${BUCKET_OUTPUTS}
GCP_PROJECT_ID=${PROJECT_ID}

# API Keys (from Secret Manager)
GEMINI_API_KEY=${GEMINI_KEY_TEXT}
GROQ_API_KEY=${GROQ_KEY_TEXT}

# Application Configuration
API_HOST=0.0.0.0
API_PORT=8080
CORS_ORIGINS=["*"]

# Local Development Override
USE_LOCAL_STORAGE=false
"@

$envContent | Out-File -FilePath "..\..\.env.cloud" -Encoding utf8

Write-Host ".env.cloud file created at root directory" -ForegroundColor Green

Write-Host "`n================================================" -ForegroundColor Green
Write-Host "  ✅ Setup Complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Resources Created:" -ForegroundColor Cyan
Write-Host "  • Cloud SQL Instance: $DB_INSTANCE" -ForegroundColor White
Write-Host "  • Database: $DB_NAME" -ForegroundColor White
Write-Host "  • Uploads Bucket: $BUCKET_UPLOADS" -ForegroundColor White
Write-Host "  • Outputs Bucket: $BUCKET_OUTPUTS" -ForegroundColor White
Write-Host "  • Secrets: GEMINI_API_KEY, GROQ_API_KEY" -ForegroundColor White
Write-Host ""
Write-Host "Connection Details:" -ForegroundColor Cyan
Write-Host "  Cloud SQL Connection: $CONNECTION_NAME" -ForegroundColor White
Write-Host "  Database URL: postgresql://${DB_USER}:****@/${DB_NAME}" -ForegroundColor White
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Migrate your database:" -ForegroundColor White
Write-Host "     gcloud sql connect $DB_INSTANCE --user=$DB_USER" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. Deploy to Cloud Run:" -ForegroundColor White
Write-Host "     .\scripts\deploy.ps1" -ForegroundColor Gray
Write-Host ""
Write-Host "Configuration saved to: .env.cloud" -ForegroundColor Green
Write-Host ""
