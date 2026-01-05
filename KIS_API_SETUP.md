# 한국투자증권 Open API 연동 가이드

## 1. API 신청 방법

### Step 1: 회원가입 및 로그인
1. **KIS Developers** 접속: https://apiportal.koreainvestment.com
2. 회원가입 또는 로그인

### Step 2: 앱 등록
1. 로그인 후 **"앱 등록"** 메뉴 선택
2. 앱 이름 입력 (예: StockSense)
3. **모의투자** 또는 **실전투자** 선택
   - 개발/테스트: **모의투자** 권장
   - 실제 거래: **실전투자** (계좌 필요)

### Step 3: API Key 발급
앱 등록 완료 후 다음 정보를 확인:
- **App Key** (API 키)
- **App Secret** (시크릿 키)

### Step 4: 계좌 정보 확인
- **계좌번호** (8자리)
- **상품코드** (일반적으로 "01")

---

## 2. 환경 설정

### .env 파일 생성

프로젝트 루트에 `.env` 파일을 생성하고 아래 내용 입력:

```env
# Korea Investment & Securities API
KIS_APP_KEY=발급받은_앱_키
KIS_APP_SECRET=발급받은_시크릿_키
KIS_ACCOUNT_NUMBER=계좌번호_8자리
KIS_ACCOUNT_PRODUCT_CODE=01
KIS_BASE_URL=https://openapi.koreainvestment.com:9443
KIS_USE_MOCK=True
```

**주의사항:**
- `KIS_USE_MOCK=True`: 모의투자 (테스트용)
- `KIS_USE_MOCK=False`: 실전투자 (실제 거래)

---

## 3. 실행 방법

### 백엔드 실행

```bash
# 가상환경 활성화
venv\Scripts\activate

# 패키지 설치 (처음 한 번만)
pip install -r requirements.txt

# 서버 실행
uvicorn app.main:app --reload
```

### 프론트엔드 실행

```bash
cd frontend

# 패키지 설치 (처음 한 번만)
npm install

# 개발 서버 실행
npm run dev
```

---

## 4. 포트폴리오 기능 사용법

### 브라우저에서 접속
- 프론트엔드: http://localhost:3000
- 백엔드 API 문서: http://localhost:8000/docs

### 주요 기능

1. **포트폴리오 조회**
   - 계좌 잔고, 보유 종목, 평가손익 확인

2. **주식 매수/매도**
   - 종목코드 입력 (6자리, 예: 005930)
   - 수량 및 가격 입력
   - 지정가/시장가 선택

3. **주문 내역 조회**
   - 최근 30일 주문 내역 확인

---

## 5. API 엔드포인트

### 포트폴리오
- `GET /api/portfolio/balance` - 계좌 잔고 조회
- `GET /api/portfolio/stock/{stock_code}` - 주식 현재가 조회
- `POST /api/portfolio/buy` - 주식 매수
- `POST /api/portfolio/sell` - 주식 매도
- `GET /api/portfolio/orders` - 주문 내역 조회

### 요청 예시 (매수)

```json
POST /api/portfolio/buy
{
  "stock_code": "005930",
  "quantity": 10,
  "price": 70000,
  "order_type": "00"
}
```

---

## 6. 주요 종목코드

### 국내 주식 (6자리)
- **삼성전자**: 005930
- **SK하이닉스**: 000660
- **NAVER**: 035420
- **카카오**: 035720
- **LG에너지솔루션**: 373220

---

## 7. 문제 해결

### 토큰 발급 실패
- App Key, App Secret 확인
- 네트워크 연결 확인

### 주문 실패
- 계좌번호, 상품코드 확인
- 모의투자/실전투자 모드 확인 (KIS_USE_MOCK)
- 종목코드 형식 확인 (6자리)

### API 호출 오류
- `.env` 파일이 프로젝트 루트에 있는지 확인
- 환경변수 이름이 정확한지 확인

---

## 8. 참고 자료

- **KIS Developers**: https://apiportal.koreainvestment.com
- **API 문서**: 로그인 후 "문서" 메뉴에서 확인
- **고객센터**: 1544-5000

---

## 9. 보안 주의사항

⚠️ **중요**: `.env` 파일은 절대 Git에 커밋하지 마세요!

- `.gitignore`에 `.env` 추가됨
- API Key와 Secret은 외부에 노출되지 않도록 주의
- 실전투자 사용 시 더욱 주의

---

## 10. 다음 단계

- [ ] 실시간 시세 구독 (WebSocket)
- [ ] 차트 분석 기능 추가
- [ ] 자동 매매 전략 구현
- [ ] 포트폴리오 성과 분석
