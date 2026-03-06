#!/bin/bash
# ML 파이프라인 로그 조회
# Usage: ./ml-logs.sh [daily_train_batch|preprocess_data|crawl_news] [YYYY-MM-DD]

SERVICE=${1:-daily_train_batch}
DATE=${2:-$(date +%Y-%m-%d)}
LOCAL_DIR="$HOME/Library/Logs/StockSense/ml/${SERVICE}"

mkdir -p "$LOCAL_DIR"
echo "📋 ML 로그 조회: ${SERVICE} / ${DATE}"

POD_NAME="log-fetcher-$$"

kubectl run -n stocksense "$POD_NAME" \
  --image=busybox \
  --restart=Never \
  --overrides="{\"spec\":{\"containers\":[{\"name\":\"main\",\"image\":\"busybox\",\"command\":[\"sleep\",\"60\"],\"volumeMounts\":[{\"name\":\"ml-data\",\"mountPath\":\"/data\"}]}],\"volumes\":[{\"name\":\"ml-data\",\"persistentVolumeClaim\":{\"claimName\":\"ml-data-pvc\"}}],\"restartPolicy\":\"Never\"}}" 2>/dev/null

# Pod Ready 대기
kubectl wait -n stocksense pod/"$POD_NAME" --for=condition=Ready --timeout=20s 2>/dev/null

kubectl exec -n stocksense "$POD_NAME" -- cat "/data/logs/${SERVICE}/${DATE}.log" 2>&1 \
  | tee "${LOCAL_DIR}/${DATE}.log"

kubectl delete pod -n stocksense "$POD_NAME" --ignore-not-found 2>/dev/null
echo ""
echo "✅ 저장 위치: ${LOCAL_DIR}/${DATE}.log"
