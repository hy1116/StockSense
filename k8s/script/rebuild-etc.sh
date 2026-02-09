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
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

echo ""
echo "📁 Project root: $PROJECT_ROOT"
echo ""

# ============================================
# 0. Git
# ============================================
echo "▶ [0/9] Pulling latest code from Git..."
git pull origin main || {
    error "Git pull failed"
    exit 1
}

# ============================================
# 1. Docker 이미지 빌드 (먼저 빌드해야 Job에서 사용 가능)
# ============================================
echo "▶ [1/9] Building Docker images..."

echo "   Building ML image..."
docker build -t stocksense-ml:latest -f ml/Dockerfile . || {
    warn "ML image build failed, skipping..."
}
success "ML image built"

# ============================================
# 2. 기본 리소스 생성 (Namespace, ConfigMap, Secret, RBAC)
# ============================================
echo ""
echo "▶ [2/9] Creating namespace, configmap, secret, rbac..."
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/argo-rbac.yaml
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
# 5. DB 마이그레이션 실행 (Argo Workflow 버전)
# ============================================
echo ""
echo "▶ [5/9] Running database migration..."

# 1. 새 워크플로우 생성 및 이름 캡처
WF_NAME=$(kubectl create -f k8s/db-migration-workflow.yaml -n stocksense -o name | cut -d'/' -f2)

if [ -z "$WF_NAME" ]; then
    error "Failed to create Workflow!"
    exit 1
fi

echo "   Workflow '$WF_NAME' created. Waiting for completion..."

# 2. Workflow 완료 대기 (타임아웃 120초로 증가)
kubectl wait --for=jsonpath='{.status.phase}'=Succeeded workflow/$WF_NAME -n stocksense --timeout=120s || {
    echo "------------------------------------------------"
    echo "❌ MIGRATION WORKFLOW FAILED"
    echo ""
    echo "Workflow status:"
    kubectl get workflow/$WF_NAME -n stocksense -o jsonpath='{.status.phase}'
    echo ""
    echo ""
    echo "Workflow pods:"
    kubectl get pods -n stocksense -l workflows.argoproj.io/workflow=$WF_NAME
    echo ""
    echo "Pod logs:"
    # Argo Workflow Pod 이름 패턴으로 로그 조회
    kubectl logs -n stocksense -l workflows.argoproj.io/workflow=$WF_NAME --all-containers=true --tail=50 2>/dev/null || echo "No logs available"
    echo "------------------------------------------------"
    error "Migration failed! Check logs above."
    exit 1
}

# success "Database migration completed ($WF_NAME)"

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
echo "🔹 CronWorkflows:"
kubectl get cronworkflows -n stocksense 2>/dev/null || echo "   No CronWorkflows found"

echo ""
echo "🔹 Workflows:"
kubectl get workflows -n stocksense 2>/dev/null || echo "   No Workflows found"