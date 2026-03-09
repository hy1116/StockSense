"""ML 스크립트 공통 로거 유틸리티

Usage:
    from ml.logger import get_logger, TeeStdout

    # logging 기반 스크립트
    logger = get_logger("collect_daily_data")
    logger.info("시작")

    # print 기반 스크립트 (main() 안에서)
    with TeeStdout("preprocess_data"):
        main_logic()

모든 스크립트의 로그는 단일 파일(logs/ml/YYYY-MM-DD.log)에 통합됩니다.
"""
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# 환경변수 ML_LOG_DIR이 있으면 사용, 없으면 로컬 Mac 경로
_log_root_env = os.environ.get("ML_LOG_DIR")
LOG_ROOT = Path(_log_root_env) if _log_root_env else Path.home() / "Library" / "Logs" / "StockSense" / "ml"

# 날짜별 단일 통합 로그 파일
_LOG_FILE = LOG_ROOT / f"{datetime.now().strftime('%Y-%m-%d')}.log"


def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """파일 + 콘솔 핸들러가 달린 logger 반환.

    로그 파일: logs/ml/YYYY-MM-DD.log (모든 스크립트 통합)
    """
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # 이미 설정됨

    logger.setLevel(level)
    fmt = logging.Formatter(f"%(asctime)s - [{name}] %(levelname)s - %(message)s")

    fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


class TeeStdout:
    """print() 출력을 콘솔 + 파일에 동시에 기록하는 컨텍스트 매니저.

    로그 파일: logs/ml/YYYY-MM-DD.log (모든 스크립트 통합)

    Usage:
        with TeeStdout("collect_daily_data"):
            main_logic()
    """

    def __init__(self, name: str):
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        self._path = _LOG_FILE
        self._file = None
        self._orig_stdout = None
        self._orig_stderr = None

    def __enter__(self):
        self._file = open(self._path, "a", encoding="utf-8")
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = _Tee(self._orig_stdout, self._file)
        sys.stderr = _Tee(self._orig_stderr, self._file)
        return self

    def __exit__(self, *_):
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        if self._file:
            self._file.close()


class _Tee:
    """두 스트림에 동시에 write하는 래퍼"""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()

    def fileno(self):
        return self._streams[0].fileno()
