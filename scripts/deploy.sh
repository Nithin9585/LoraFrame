#!/bin/bash
# Deploy CineAI to Google Cloud Run
# Run this after setup_cloud.sh

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}🚀 Deploying CineAI to Cloud Run${NC}"
echo "======================================"

# Get project ID
PROJECT_ID=$(gcloud config get-value project)
if [ -z "$PROJECT_ID" ]; then
    echo "Error: No GCP project set"
    echo "Run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

# Get settings
read -p "Enter region (default: us-central1): " REGION
REGION=${REGION:-us-central1}

read -p "Enter service name (default: cineai-api): " SERVICE_NAME
SERVICE_NAME=${SERVICE_NAME:-cineai-api}

# Configure Docker
echo -e "\n${YELLOW}Configuring Docker...${NC}"
gcloud auth configure-docker

# Navigate to project root (Dockerfile is there)
cd ..

# Build image
echo -e "\n${YELLOW}Building Docker image...${NC}"
docker build -t gcr.io/$PROJECT_ID/$SERVICE_NAME:latest .

# Push image
echo -e "\n${YELLOW}Pushing to Container Registry...${NC}"
docker push gcr.io/$PROJECT_ID/$SERVICE_NAME:latest

# Get Cloud SQL connection name
CONNECTION_NAME=$(gcloud sql instances describe cineai-db --format='value(connectionName)')

# Deploy to Cloud Run
echo -e "\n${YELLOW}Deploying to Cloud Run...${NC}"
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
    --set-env-vars "USE_GCS=true" \
    --set-env-vars "GCS_BUCKET_UPLOADS=cineai-uploads-$PROJECT_ID" \
    --set-env-vars "GCS_BUCKET_OUTPUTS=cineai-outputs-$PROJECT_ID" \
    --set-env-vars "GCP_PROJECT_ID=$PROJECT_ID" \
    --set-env-vars "DEBUG=false" \
    --set-secrets "GEMINI_API_KEY=gemini-api-key:latest" \
    --set-secrets "GROQ_API_KEY=groq-api-key:latest" \
    --set-secrets "PINECONE_API_KEY=pinecone-api-key:latest" \
    --set-secrets "DATABASE_URL=db-url:latest" 2>/dev/null || \
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
        --set-env-vars "USE_GCS=true" \
        --set-env-vars "GCS_BUCKET_UPLOADS=cineai-uploads-$PROJECT_ID" \
        --set-env-vars "GCS_BUCKET_OUTPUTS=cineai-outputs-$PROJECT_ID" \
        --set-env-vars "GCP_PROJECT_ID=$PROJECT_ID" \
        --set-env-vars "DEBUG=false" \
        --set-secrets "GEMINI_API_KEY=gemini-api-key:latest" \
        --set-secrets "GROQ_API_KEY=groq-api-key:latest" \
        --set-secrets "PINECONE_API_KEY=pinecone-api-key:latest"

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)')

echo -e "\n${GREEN}======================================"
echo "✅ Deployment Complete!"
echo "======================================${NC}"
echo
echo "🌐 Service URL: $SERVICE_URL"
echo
echo "🧪 Test Commands:"
echo "  curl $SERVICE_URL/health"
echo
echo "  curl -X POST $SERVICE_URL/api/characters \\"
echo "    -F 'name=TestChar' \\"
echo "    -F 'description=Test' \\"
echo "    -F 'consent=true' \\"
echo "    -F 'files=@image.jpg'"
echo
echo "📊 View Logs:"
echo "  gcloud run services logs read $SERVICE_NAME --region $REGION"
echo
