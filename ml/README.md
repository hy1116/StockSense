# 데이터 수집 및 전처리 스크립트

## 📋 개요

머신러닝 모델 학습을 위한 주가 데이터 수집 및 전처리 스크립트입니다.

## 🗂️ 디렉토리 구조

```
StockSense/
├── scripts/
│   ├── collect_historical_data.py   # 과거 데이터 수집
│   ├── collect_daily_data.py        # 일일 데이터 수집
│   ├── preprocess_data.py           # 데이터 전처리
│   └── README.md
│
└── data/
    ├── raw/
    │   ├── daily/                    # 일일 수집 데이터
    │   └── historical/               # 과거 데이터
    ├── processed/                    # 전처리된 데이터
    └── datasets/                     # 학습용 데이터셋
        ├── train.csv
        ├── validation.csv
        └── test.csv
```

---

## ⚙️ 사전 준비

### 1. 환경 변수 확인
스크립트는 `.env` 파일에서 KIS API 키를 읽습니다. 다음 변수들이 설정되어 있는지 확인하세요:

```env
KIS_APP_KEY=your_app_key_here
KIS_APP_SECRET=your_app_secret_here
KIS_ACCOUNT_NUMBER=your_account_number
KIS_ACCOUNT_PRODUCT_CODE=01
KIS_BASE_URL=https://openapi.koreainvestment.com:9443
KIS_USE_MOCK=True
KIS_CUST_TYPE=P
```

### 2. 의존성 설치
```bash
pip install python-dotenv pandas numpy
```

---

## 🚀 사용 방법

### 1. 과거 데이터 수집

#### 전체 주요 종목 수집 (15개)
```bash
cd StockSense
python scripts/collect_historical_data.py
```

#### 특정 종목만 수집
```bash
python scripts/collect_historical_data.py --stock 005930
```

#### 옵션
```bash
# 수집 기간 지정 (기본: 365일)
python scripts/collect_historical_data.py --days 730

# JSON 형식으로 저장
python scripts/collect_historical_data.py --format json

# CSV와 JSON 둘 다 저장
python scripts/collect_historical_data.py --format both
```

#### 출력 예시
```
🚀 Starting data collection for 15 stocks...
📅 Period: 365 days
💾 Format: csv

[1/15] Processing 005930...
📊 Collecting data for 005930...
✅ Collected 100 days of data for 삼성전자 (005930)
💾 Saved to data\raw\historical\005930_historical.csv
...
==================================================
✅ Success: 15
❌ Failed: 0
📁 Data saved to: data\raw\historical
==================================================
```

---

### 2. 일일 데이터 수집

#### 오늘의 데이터 수집 (주요 10개 종목)
```bash
python scripts/collect_daily_data.py
```

#### 과거 데이터 파일 업데이트 안 함
```bash
python scripts/collect_daily_data.py --no-update-historical
```

#### 출력 예시
```
🚀 Daily data collection started at 2026-01-07 15:30:00
📊 Collecting 10 stocks...

💾 Saved: 005930_20260107.json
📝 Updated historical file: 005930_historical.csv
...
==================================================
✅ Success: 10
❌ Failed: 0
📁 Data saved to: data\raw\daily
==================================================
```

#### 자동화 (Cron / Task Scheduler)

**Linux/Mac (Cron):**
```bash
# crontab -e
# 매일 오후 4시에 실행 (장 마감 후)
0 16 * * * cd /path/to/StockSense && python scripts/collect_daily_data.py
```

**Windows (Task Scheduler):**
1. 작업 스케줄러 열기
2. "기본 작업 만들기" 클릭
3. 이름: "StockSense Daily Collection"
4. 트리거: 매일 오후 4시
5. 작업: Python 실행
   - 프로그램: `python`
   - 인수: `scripts/collect_daily_data.py`
   - 시작 위치: `C:\Users\...\StockSense`

---

### 3. 데이터 전처리

#### 전체 파이프라인 실행
```bash
python scripts/preprocess_data.py
```

이 명령은 다음을 수행합니다:
1. 모든 종목의 원본 데이터에서 ML 피처 생성
2. 전처리된 데이터를 `data/processed/` 에 저장
3. 전체 데이터를 결합
4. 학습/검증/테스트 데이터로 분할 (70% / 15% / 15%)
5. `data/datasets/` 에 저장

#### 특정 종목만 전처리
```bash
python scripts/preprocess_data.py --stock 005930
```

#### 출력 예시
```
==================================================
🚀 Starting data preprocessing pipeline
==================================================

Found 15 stock files to process

📊 Preprocessing 005930...
   Loaded 100 records
   Created technical indicators
   Created target variables
   Cleaned data: 80 records (removed 20 NaN rows)
   💾 Saved to 005930_features.csv
...

🔗 Combining all stock data...
   Loaded 005930_features.csv: 80 records
   ...
✅ Combined 15 files: Total 1200 records

✂️  Splitting dataset...
   Train: 840 records (70%)
   Validation: 180 records (15%)
   Test: 180 records (15%)
   💾 Saved to data\datasets

==================================================
✅ Data preprocessing completed!
==================================================

📁 Processed files: data\processed
📁 Dataset files: data\datasets
```

---

## 📊 생성되는 피처 (Features)

### 기본 OHLCV
- `open`, `high`, `low`, `close`, `volume`

### 이동평균 (Moving Averages)
- `ma5`, `ma10`, `ma20`, `ma60`
- `volume_ma5`, `volume_ma20`

### 기술적 지표
- `rsi` - Relative Strength Index (14일)
- `macd`, `macd_signal`, `macd_diff` - MACD
- `bb_upper`, `bb_middle`, `bb_lower`, `bb_width` - 볼린저 밴드

### 변화율
- `price_change_1d`, `price_change_5d`, `price_change_20d` - 가격 변화율
- `volume_change` - 거래량 변화율

### 변동성
- `volatility_5d`, `volatility_20d` - 표준편차

### 비율
- `high_low_ratio` - 고가/저가 비율
- `close_open_ratio` - 종가/시가 비율

### 타겟 변수
- `target_price` - 다음날 종가
- `target_return` - 다음날 수익률
- `target_direction` - 상승(1) / 하락(0)

---

## 📌 주요 종목 목록

기본적으로 KOSPI 시가총액 상위 종목을 수집합니다:

1. 005930 - 삼성전자
2. 000660 - SK하이닉스
3. 035420 - NAVER
4. 051910 - LG화학
5. 005380 - 현대차
6. 006400 - 삼성SDI
7. 035720 - 카카오
8. 000270 - 기아
9. 207940 - 삼성바이오로직스
10. 068270 - 셀트리온

종목을 추가하려면 각 스크립트의 `self.major_stocks` 리스트를 수정하세요.

---

## ⚠️  주의사항

### API Rate Limiting
- KIS API는 초당 요청 제한이 있습니다
- 스크립트에 `time.sleep()` 지연이 포함되어 있습니다
- 대량 수집 시 시간이 오래 걸릴 수 있습니다

### 데이터 품질
- 장 마감 후(오후 4시 이후) 수집을 권장합니다
- 공휴일/주말에는 데이터가 없을 수 있습니다
- 상장폐지 종목은 오류가 발생할 수 있습니다

### 디스크 용량
- CSV 파일은 종목당 약 10KB ~ 100KB
- 15개 종목 × 365일 ≈ 1MB ~ 10MB
- 충분한 여유 공간을 확보하세요

---

## 🔧 트러블슈팅

### ModuleNotFoundError
```bash
# 가상환경 활성화 확인
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 의존성 설치
pip install pandas numpy
```

### KIS API 오류
```
❌ Error for 005930: API 요청 실패
```
- `.env` 파일의 KIS API 키 확인
- API 토큰이 만료되었는지 확인
- 네트워크 연결 확인

### 데이터 파일 없음
```
⚠️  No raw data files found.
   Please run: python scripts/collect_historical_data.py
```
- 먼저 과거 데이터를 수집해야 합니다

---

## 🔄 전체 워크플로우

```bash
# 1. 과거 데이터 수집 (최초 1회)
python scripts/collect_historical_data.py

# 2. 데이터 전처리
python scripts/preprocess_data.py

# 3. 일일 데이터 수집 (매일 자동화)
python scripts/collect_daily_data.py

# 4. 필요시 재전처리 (주 1회)
python scripts/preprocess_data.py
```

---

## 📖 다음 단계

데이터 수집이 완료되면:
1. `ML_TRAINING_GUIDE.md` 참고하여 모델 학습
2. `scripts/train_model.py` 작성 및 실행
3. PredictionService에 ML 모델 통합

---

## 📞 문의

문제가 발생하면 다음을 확인하세요:
1. 로그 메시지 (❌, ⚠️ 표시)
2. `.env` 파일 설정
3. KIS API 상태
4. 디렉토리 권한
