#!/bin/bash
set -e

echo "=============================================="
echo "🚀 StockSense K8s Backend Rebuild Script"
echo "=============================================="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 함수: 성공 메시지
success() { echo -e "${GREEN}✅ $1${NC}"; }
# 함수: 경고 메시지
warn() { echo -e "${YELLOW}⚠️  $1${NC}"; }
# 함수: 에러 메시지
error() { echo -e "${RED}❌ $1${NC}"; }

# 스크립트 위치 확인
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

echo ""
echo "📁 Project root: $PROJECT_ROOT"
echo ""

# ============================================
# 0. Git
# ============================================
echo "▶ [1/9] Pulling latest code from Git..."
git pull origin main || {
    error "Git pull failed"
    exit 1
}

# ============================================
# 1. Docker 이미지 빌드 (먼저 빌드해야 Job에서 사용 가능)
# ============================================
echo "   Building backend image..."
docker build -t stocksense-backend:latest -f app/Dockerfile . || {
    error "Backend build failed"
    exit 1
}
success "Backend image built"

# ============================================
# 2. 기본 리소스 생성 (Namespace, ConfigMap, Secret)
# ============================================
echo ""
echo "▶ [2/9] Creating namespace, configmap, secret..."
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
success "Basic resources created"

# ============================================
# 3. Backend & Frontend 배포
# ============================================
kubectl apply -f k8s/backend-deployment.yaml

kubectl rollout restart deployment/backend -n stocksense 2>/dev/null || true

success "Backend & Frontend deployed"

# ============================================
# 4. Ingress 설정
# ============================================
echo ""
kubectl apply -f k8s/ingress.yaml
success "Ingress configured"

# ============================================
# 배포 완료 - 상태 확인
# ============================================
echo ""
echo "=============================================="
echo "📊 Deployment Status"
echo "=============================================="
echo ""

echo "🔹 Pods:"
kubectl get pods -n stocksense -o wide

echo ""
echo "🔹 Services:"
kubectl get svc -n stocksense

# ============================================
# Docker 미사용 이미지 정리
# ============================================
echo ""
echo "▶ Cleaning up dangling Docker images..."
docker image prune -f
success "Docker cleanup done"

echo ""
echo "=============================================="
echo "✅ All done!"
echo "=============================================="
