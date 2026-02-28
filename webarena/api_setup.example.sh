#!/bin/bash

# Define your GCP project details
export PROJECT_ID="YOUR_GCP_PROJECT_ID_HERE"
export LOCATION="us-central1" # Or your preferred location

# Fetch a fresh access token
export OPENAI_API_KEY=$(gcloud auth application-default print-access-token)

# Route the framework to the Vertex AI endpoint
export OPENAI_BASE_URL="https://${LOCATION}-aiplatform.googleapis.com/v1beta1/projects/${PROJECT_ID}/locations/${LOCATION}/endpoints/openapi"
export OPENAI_API_BASE="https://${LOCATION}-aiplatform.googleapis.com/v1beta1/projects/${PROJECT_ID}/locations/${LOCATION}/endpoints/openapi"

echo "✅ Vertex AI environment variables set! Token is valid for 1 hour."