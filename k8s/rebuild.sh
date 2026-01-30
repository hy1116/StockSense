#!/bin/bash

# 0. Argo 컨트롤러 체크
if ! kubectl get deployment workflow-controller -n argo > /dev/null 2>&1; then
    echo "⚠️ Argo 컨트롤러 설치 중..."
    kubectl create namespace argo --dry-run=client -o yaml | kubectl apply -f -
    kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/download/v3.5.4/install.yaml
    kubectl wait --for=condition=available --timeout=300s deployment/workflow-controller -n argo
fi

# kubectl create rolebinding stocksense-argo-workflow-binding --clusterrole=argo-workflow --serviceaccount=argo:default -n stocksense
# kubectl create rolebinding stocksense-workflow-binding --clusterrole=argo-workflow --serviceaccount=argo:default -n stocksense

# 1. 빌드 및 배포
docker build -t stocksense-backend:latest .
docker build -t stocksense-frontend:latest ./frontend

kubectl apply -f k8s/

kubectl rollout restart deployment/backend -n stocksense
kubectl rollout restart deployment/frontend -n stocksense

kubectl get pods -n stocksense