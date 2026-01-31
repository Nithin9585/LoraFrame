#!/bin/bash
# Google Cloud Setup Script for CineAI
# This script creates all required GCP resources

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 CineAI Google Cloud Setup${NC}"
echo "======================================"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud CLI is not installed${NC}"
    echo "Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Get project ID
read -p "Enter your GCP Project ID: " PROJECT_ID
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}Error: Project ID is required${NC}"
    exit 1
fi

# Set project
echo -e "\n${YELLOW}Setting GCP project...${NC}"
gcloud config set project $PROJECT_ID

# Enable required APIs
echo -e "\n${YELLOW}Enabling required APIs...${NC}"
gcloud services enable \
    run.googleapis.com \
    sqladmin.googleapis.com \
    storage-api.googleapis.com \
    secretmanager.googleapis.com \
    containerregistry.googleapis.com

# Create Cloud SQL instance
echo -e "\n${YELLOW}Creating Cloud SQL PostgreSQL instance...${NC}"
read -p "Enter region (default: us-central1): " REGION
REGION=${REGION:-us-central1}

gcloud sql instances create cineai-db \
    --database-version=POSTGRES_15 \
    --tier=db-f1-micro \
    --region=$REGION \
    --no-backup \
    --database-flags=max_connections=100 || echo "Instance may already exist"

# Create database
echo -e "\n${YELLOW}Creating database...${NC}"
gcloud sql databases create cineai \
    --instance=cineai-db || echo "Database may already exist"

# Create database user
echo -e "\n${YELLOW}Creating database user...${NC}"
read -sp "Enter database password: " DB_PASSWORD
echo
gcloud sql users create cineai-user \
    --instance=cineai-db \
    --password=$DB_PASSWORD || echo "User may already exist"

# Get connection name
CONNECTION_NAME=$(gcloud sql instances describe cineai-db --format='value(connectionName)')
echo -e "${GREEN}✅ Cloud SQL Connection Name: $CONNECTION_NAME${NC}"

# Create GCS buckets
echo -e "\n${YELLOW}Creating Google Cloud Storage buckets...${NC}"
gsutil mb -l $REGION gs://cineai-uploads-$PROJECT_ID || echo "Bucket may already exist"
gsutil mb -l $REGION gs://cineai-outputs-$PROJECT_ID || echo "Bucket may already exist"

# Set public access for outputs bucket
gsutil iam ch allUsers:objectViewer gs://cineai-outputs-$PROJECT_ID

echo -e "${GREEN}✅ Created GCS buckets:${NC}"
echo "  - gs://cineai-uploads-$PROJECT_ID (private)"
echo "  - gs://cineai-outputs-$PROJECT_ID (public read)"

# Create secrets
echo -e "\n${YELLOW}Creating Secret Manager secrets...${NC}"

read -p "Enter Gemini API Key: " GEMINI_KEY
echo -n "$GEMINI_KEY" | gcloud secrets create gemini-api-key --data-file=- || \
    echo -n "$GEMINI_KEY" | gcloud secrets versions add gemini-api-key --data-file=-

read -p "Enter Groq API Key: " GROQ_KEY
echo -n "$GROQ_KEY" | gcloud secrets create groq-api-key --data-file=- || \
    echo -n "$GROQ_KEY" | gcloud secrets versions add groq-api-key --data-file=-

read -p "Enter Pinecone API Key (or press Enter to skip): " PINECONE_KEY
if [ ! -z "$PINECONE_KEY" ]; then
    echo -n "$PINECONE_KEY" | gcloud secrets create pinecone-api-key --data-file=- || \
        echo -n "$PINECONE_KEY" | gcloud secrets versions add pinecone-api-key --data-file=-
fi

echo -e "${GREEN}✅ Secrets created${NC}"

# Grant Cloud Run access to secrets
echo -e "\n${YELLOW}Granting Cloud Run access to secrets...${NC}"
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
SERVICE_ACCOUNT="$PROJECT_NUMBER-compute@developer.gserviceaccount.com"

gcloud secrets add-iam-policy-binding gemini-api-key \
    --member=serviceAccount:$SERVICE_ACCOUNT \
    --role=roles/secretmanager.secretAccessor

gcloud secrets add-iam-policy-binding groq-api-key \
    --member=serviceAccount:$SERVICE_ACCOUNT \
    --role=roles/secretmanager.secretAccessor

if [ ! -z "$PINECONE_KEY" ]; then
    gcloud secrets add-iam-policy-binding pinecone-api-key \
        --member=serviceAccount:$SERVICE_ACCOUNT \
        --role=roles/secretmanager.secretAccessor
fi

echo -e "${GREEN}✅ IAM permissions configured${NC}"

# Create .env.cloud file
echo -e "\n${YELLOW}Creating .env.cloud configuration...${NC}"
cat > .env.cloud << EOF
# CineAI Cloud Run Configuration
# Generated: $(date)

# App
APP_NAME=CineAI API
DEBUG=false

# Database (Cloud SQL)
DATABASE_URL=postgresql+psycopg2://cineai-user:${DB_PASSWORD}@/cineai?host=/cloudsql/${CONNECTION_NAME}
CLOUD_SQL_CONNECTION_NAME=${CONNECTION_NAME}

# Storage (Google Cloud Storage)
USE_LOCAL_STORAGE=false
USE_GCS=true
GCS_BUCKET_UPLOADS=cineai-uploads-${PROJECT_ID}
GCS_BUCKET_OUTPUTS=cineai-outputs-${PROJECT_ID}
GCP_PROJECT_ID=${PROJECT_ID}

# Vector DB (Pinecone)
PINECONE_ENV=us-east-1
PINECONE_INDEX=idlock-characters

# API Keys (loaded from Secret Manager)
# GEMINI_API_KEY - from secret
# GROQ_API_KEY - from secret
# PINECONE_API_KEY - from secret

# Models
GEMINI_MODEL=gemini-2.5-flash-image
GEMINI_MODEL_CHARACTER=gemini-3-pro-image-preview
GROQ_MODEL=llama-3.3-70b-versatile

# Settings
IDR_THRESHOLD=0.7
SEMANTIC_WEIGHT=0.6
EPISODIC_WEIGHT=0.4
EOF

echo -e "${GREEN}✅ Created .env.cloud${NC}"

# Summary
echo -e "\n${GREEN}======================================"
echo "✅ Setup Complete!"
echo "======================================${NC}"
echo
echo "📝 Next Steps:"
echo "1. Update Dockerfile (see CLOUD_RUN_DEPLOYMENT.md)"
echo "2. Update requirements.txt with GCP libraries"
echo "3. Run: ./scripts/deploy.sh"
echo
echo "🔗 Resources Created:"
echo "  - Cloud SQL: $CONNECTION_NAME"
echo "  - GCS Buckets: cineai-uploads-$PROJECT_ID, cineai-outputs-$PROJECT_ID"
echo "  - Secrets: gemini-api-key, groq-api-key"
echo
echo "💡 Database Connection:"
echo "  gcloud sql connect cineai-db --user=cineai-user"
echo
