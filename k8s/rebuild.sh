cd C:/Users/Q219057/pyspace/StockSense

# 백엔드 빌드
Write-Host "Building backend image..." -ForegroundColor Green
docker build -t stocksense-backend:latest .

# 프론트엔드 빌드
Write-Host "Building frontend image..." -ForegroundColor Green
docker build -t stocksense-frontend:latest ./frontend

# Pod 재시작
Write-Host "Restarting pods..." -ForegroundColor Green
kubectl rollout restart deployment/backend -n stocksense
kubectl rollout restart deployment/frontend -n stocksense

# 상태 확인
Write-Host "Checking status..." -ForegroundColor Green
kubectl get pods -n stocksense

# 필요시 리소스 재적용
Write-Host "Reapplying Kubernetes resources..." -ForegroundColor Green
kubectl apply -n stocksense -f k8s/