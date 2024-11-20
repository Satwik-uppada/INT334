#!/bin/bash

# Enable Ingress addon in Minikube (if not already enabled)
echo "Enabling Ingress addon in Minikube..."
minikube addons enable ingress

# Build Docker image
echo "Building Docker image..."
docker build -t heart-disease-app:latest .

# Load image into Minikube
echo "Loading image into Minikube..."
minikube image load heart-disease-app:latest

# Apply Kubernetes configurations
echo "Applying Kubernetes configurations..."
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/ingress.yaml

# Wait for deployment to be ready
echo "Waiting for deployment to be ready..."
kubectl rollout status deployment/heart-disease-app

# Wait for Ingress to get an IP address
echo "Waiting for Ingress to get an IP address..."
kubectl wait --for=condition=LoadBalancer --timeout=180s ingress/heart-disease-app-ingress

# Add local DNS entry (for local testing)
echo "Adding entry to /etc/hosts..."
echo "$(minikube ip) heart-disease-app.local" | sudo tee -a /etc/hosts

echo "Deployment complete!"
echo "You can access the application at: http://heart-disease-app.local"