#!/bin/bash

# Delete Kubernetes resources
echo "Cleaning up Kubernetes resources..."
kubectl delete ingress heart-disease-app-ingress
kubectl delete service heart-disease-app-service
kubectl delete deployment heart-disease-app
kubectl delete configmap heart-disease-app-config

# Remove Docker image
echo "Removing Docker image..."
docker rmi heart-disease-app:latest

# Remove local DNS entry
echo "Removing entry from /etc/hosts..."
sudo sed -i '/heart-disease-app.local/d' /etc/hosts

echo "Cleanup complete!"