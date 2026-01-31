#!/bin/bash
# Deploy CineAI using Cloud Build (faster - builds in Google's infrastructure)

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}🚀 Deploying CineAI to Cloud Run (using Cloud Build)${NC}"
echo "======================================"

# Get project ID
PROJECT_ID=$(gcloud config get-value project)
if [ -z "$PROJECT_ID" ]; then
    echo "Error: No GCP project set"
    exit 1
fi

# Get settings
read -p "Enter region (default: us-central1): " REGION
REGION=${REGION:-us-central1}

read -p "Enter service name (default: cineai-api): " SERVICE_NAME
SERVICE_NAME=${SERVICE_NAME:-cineai-api}

# Check for Redis instance
echo -e "\n${YELLOW}Note: RQ workers require Redis. Options:${NC}"
echo "  1. Use Cloud Memorystore Redis (recommended for production)"
echo "  2. Use sync mode (no background jobs) for simple deployments"
read -p "Do you have a Redis instance? (y/n, default: n): " HAS_REDIS
HAS_REDIS=${HAS_REDIS:-n}

if [ "$HAS_REDIS" = "y" ]; then
    read -p "Enter Redis URL (e.g., redis://10.0.0.1:6379): " REDIS_URL
fi

# Navigate to project root
cd ..

# Enable Cloud Build API
echo -e "\n${YELLOW}Enabling Cloud Build API...${NC}"
gcloud services enable cloudbuild.googleapis.com

# Create GCS buckets if they don't exist
echo -e "\n${YELLOW}Setting up GCS buckets...${NC}"
gsutil ls gs://cineai-uploads-$PROJECT_ID 2>/dev/null || gsutil mb -p $PROJECT_ID -l $REGION gs://cineai-uploads-$PROJECT_ID
gsutil ls gs://cineai-outputs-$PROJECT_ID 2>/dev/null || gsutil mb -p $PROJECT_ID -l $REGION gs://cineai-outputs-$PROJECT_ID

# Build using Cloud Build (faster - builds in Google's servers!)
echo -e "\n${YELLOW}Building with Cloud Build (this is much faster!)...${NC}"
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME:latest .

# Get Cloud SQL connection name
echo -e "\n${YELLOW}Getting Cloud SQL connection...${NC}"
CONNECTION_NAME=$(gcloud sql instances describe cineai-db --format='value(connectionName)' 2>/dev/null || echo "")

# Deploy to Cloud Run
echo -e "\n${YELLOW}Deploying to Cloud Run...${NC}"

# Build environment variables (DATABASE_URL comes from secret, not hardcoded)
ENV_VARS="USE_GCS=true,GCS_BUCKET_UPLOADS=cineai-uploads-$PROJECT_ID,GCS_BUCKET_OUTPUTS=cineai-outputs-$PROJECT_ID,GCP_PROJECT_ID=$PROJECT_ID,DEBUG=false,CLOUD_SQL_CONNECTION_NAME=$CONNECTION_NAME,USE_LOCAL_STORAGE=false"

if [ "$HAS_REDIS" = "y" ] && [ -n "$REDIS_URL" ]; then
    ENV_VARS="$ENV_VARS,REDIS_URL=$REDIS_URL"
    echo "  Including Redis configuration"
else
    echo "  Deploying without Redis (async jobs will fail gracefully)"
fi

gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/$SERVICE_NAME:latest \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10 \
    --min-instances 0 \
    --concurrency 80 \
    --add-cloudsql-instances $CONNECTION_NAME \
    --set-env-vars "$ENV_VARS" \
    --set-secrets "GEMINI_API_KEY=gemini-api-key:latest,GROQ_API_KEY=groq-api-key:latest,PINECONE_API_KEY=pinecone-api-key:latest,DATABASE_URL=DATABASE_URL:latest"

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)')

echo -e "\n${GREEN}======================================${NC}"
echo -e "${GREEN}✅ Deployment Complete!${NC}"
echo -e "${GREEN}======================================${NC}"
echo -e "\n${YELLOW}Service URL:${NC} $SERVICE_URL"
echo -e "\n${YELLOW}Test your deployment:${NC}"
echo "  curl $SERVICE_URL/health"
echo "  curl $SERVICE_URL/docs"
echo -e "\n${YELLOW}View logs:${NC}"
echo "  gcloud run services logs read $SERVICE_NAME --region=$REGION"
echo ""
