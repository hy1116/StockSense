"""ML 파이프라인 실행 스크립트 (CronJob용)

수집 → 뉴스 크롤링 → 뉴스 감성분석 → 전처리 → 학습 순서로 전체 파이프라인을 실행합니다.
"""
import sys
import subprocess
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ml-pipeline")

SCRIPTS_DIR = Path(__file__).parent


def run_step(name: str, script: str, args: list = None):
    """파이프라인 단계 실행"""
    logger.info(f"[START] {name}")
    cmd = [sys.executable, str(SCRIPTS_DIR / script)]
    if args:
        cmd.extend(args)

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        logger.info(result.stdout)
    if result.stderr:
        logger.warning(result.stderr)

    if result.returncode != 0:
        logger.error(f"[FAIL] {name} (exit code: {result.returncode})")
        sys.exit(1)

    logger.info(f"[DONE] {name}")


def main():
    start = datetime.now()
    logger.info(f"=== ML Pipeline started at {start.isoformat()} ===")

    run_step("1. 일일 데이터 수집", "collect_daily_data.py")
    run_step("2. 전일 예측 평가", "evaluate_predictions.py")
    run_step("3. 뉴스 크롤링 및 감성분석", "crawl_news.py", ["--hours", "0"])
    run_step("4. 데이터 전처리", "preprocess_data.py")
    run_step("5. 모델 학습 (XGBoost + LSTM)", "daily_train_batch.py")

    elapsed = datetime.now() - start
    logger.info(f"=== ML Pipeline completed in {elapsed} ===")


if __name__ == "__main__":
    main()
