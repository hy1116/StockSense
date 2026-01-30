#!/bin/bash
set -e

echo "=============================================="
echo "🚀 StockSense K8s Full Rebuild Script"
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
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo ""
echo "📁 Project root: $PROJECT_ROOT"
echo ""

# ============================================
# 1. Docker 이미지 빌드 (먼저 빌드해야 Job에서 사용 가능)
# ============================================
echo "▶ [1/9] Building Docker images..."

echo "   Building backend image..."
docker build -t stocksense-backend:latest -f app/Dockerfile . || {
    error "Backend build failed"
    exit 1
}
success "Backend image built"

echo "   Building frontend image..."
docker build -t stocksense-frontend:latest -f frontend/Dockerfile ./frontend || {
    error "Frontend build failed"
    exit 1
}
success "Frontend image built"

echo "   Building ML image..."
docker build -t stocksense-ml:latest -f ml/Dockerfile . || {
    warn "ML image build failed, skipping..."
}
success "ML image built"

# ============================================
# 2. 기본 리소스 생성 (Namespace, ConfigMap, Secret)
# ============================================
echo ""
echo "▶ [2/9] Creating namespace, configmap, secret..."
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
success "Basic resources created"

# ============================================
# 3. PostgreSQL 배포
# ============================================
echo ""
echo "▶ [3/9] Deploying PostgreSQL..."
kubectl apply -f k8s/postgres-deployment.yaml
echo "   Waiting for PostgreSQL to be ready..."
kubectl wait --for=condition=ready pod -l app=postgres -n stocksense --timeout=120s || {
    error "PostgreSQL failed to start"
    exit 1
}
success "PostgreSQL is ready"

# ============================================
# 4. Redis 배포
# ============================================
echo ""
echo "▶ [4/9] Deploying Redis..."
kubectl apply -f k8s/redis-deployment.yaml
kubectl wait --for=condition=ready pod -l app=redis -n stocksense --timeout=60s || {
    warn "Redis may not be ready yet, continuing..."
}
success "Redis deployed"

# ============================================
# 5. DB 마이그레이션 실행
# ============================================
echo ""
echo "▶ [5/9] Running database migration..."

# 기존 Job 삭제 (있으면)
kubectl delete job db-migration -n stocksense --ignore-not-found=true
kubectl delete job init-collection-stocks -n stocksense --ignore-not-found=true

kubectl apply -f k8s/db-migration-job.yaml
echo "   Waiting for migration to complete..."
kubectl wait --for=condition=complete job/db-migration -n stocksense --timeout=180s || {
    error "Migration failed! Check logs: kubectl logs job/db-migration -n stocksense"
    exit 1
}
success "Database migration completed"

# 수집 종목 초기화 대기
echo "   Waiting for collection stocks initialization..."
kubectl wait --for=condition=complete job/init-collection-stocks -n stocksense --timeout=120s || {
    warn "Collection stocks init may have failed, check logs:"
    warn "kubectl logs job/init-collection-stocks -n stocksense"
}

# ============================================
# 6. Backend & Frontend 배포
# ============================================
echo ""
echo "▶ [6/9] Deploying Backend & Frontend..."
kubectl apply -f k8s/backend-deployment.yaml
kubectl apply -f k8s/frontend-deployment.yaml

# 기존 배포가 있으면 재시작
kubectl rollout restart deployment/backend -n stocksense 2>/dev/null || true
kubectl rollout restart deployment/frontend -n stocksense 2>/dev/null || true

success "Backend & Frontend deployed"

# ============================================
# 7. Ingress 설정
# ============================================
echo ""
echo "▶ [7/9] Configuring Ingress..."
kubectl apply -f k8s/ingress.yaml
success "Ingress configured"

# ============================================
# 8. Argo Workflows 설치 및 ML 파이프라인 설정
# ============================================
echo ""
echo "▶ [8/9] Setting up Argo Workflows..."

# Argo 컨트롤러 설치 확인
if ! kubectl get deployment workflow-controller -n argo > /dev/null 2>&1; then
    echo "   Installing Argo Workflows controller..."
    kubectl create namespace argo --dry-run=client -o yaml | kubectl apply -f -
    kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/download/v3.5.4/install.yaml
    echo "   Waiting for Argo controller to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/workflow-controller -n argo || {
        warn "Argo controller may not be ready yet"
    }
    success "Argo Workflows installed"
else
    success "Argo Workflows already installed"
fi

# ML 파이프라인 CronWorkflow 배포
kubectl apply -f k8s/argo-workflow.yaml
success "ML pipeline configured"

# ============================================
# 9. EFK Stack 배포 (로깅)
# ============================================
echo ""
echo "▶ [9/9] Deploying EFK Stack (Logging)..."
kubectl apply -f k8s/elasticsearch.yaml
kubectl apply -f k8s/fluentd.yaml
kubectl apply -f k8s/kibana.yaml
success "EFK Stack deployed"

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

echo ""
echo "🔹 Jobs:"
kubectl get jobs -n stocksense

echo ""
echo "🔹 CronWorkflows:"
kubectl get cronworkflows -n stocksense 2>/dev/null || echo "   No CronWorkflows found"

echo ""
echo "=============================================="
echo -e "${GREEN}🎉 StockSense deployment completed!${NC}"
echo "=============================================="
echo ""
echo "📌 Useful commands:"
echo "   kubectl logs -f deployment/backend -n stocksense    # Backend logs"
echo "   kubectl logs -f deployment/frontend -n stocksense   # Frontend logs"
echo "   kubectl logs job/db-migration -n stocksense         # Migration logs"
echo "   kubectl logs job/init-collection-stocks -n stocksense  # Init stocks logs"
echo "   argo list -n stocksense                             # Argo workflows"
echo ""
