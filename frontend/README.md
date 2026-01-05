# StockSense Frontend

React + Vite 기반 주식 예측 시스템 프론트엔드

## 기술 스택

- **Framework**: React 18
- **Build Tool**: Vite
- **Routing**: React Router v6
- **State Management**: Zustand
- **Data Fetching**: TanStack Query (React Query)
- **HTTP Client**: Axios
- **Charts**: Recharts
- **Date Handling**: Day.js

## 프로젝트 구조

```
frontend/
├── src/
│   ├── components/      # 재사용 가능한 컴포넌트
│   │   └── Layout.jsx   # 레이아웃 컴포넌트
│   ├── pages/           # 페이지 컴포넌트
│   │   ├── Home.jsx     # 홈 페이지
│   │   ├── StockDetail.jsx  # 종목 상세
│   │   └── Prediction.jsx   # 예측 페이지
│   ├── services/        # API 서비스
│   │   └── api.js       # API 클라이언트
│   ├── utils/           # 유틸리티 함수
│   ├── App.jsx          # 앱 루트 컴포넌트
│   ├── App.css          # 앱 스타일
│   ├── main.jsx         # 진입점
│   └── index.css        # 전역 스타일
├── public/              # 정적 파일
├── index.html           # HTML 템플릿
├── vite.config.js       # Vite 설정
└── package.json         # 의존성 관리
```

## 시작하기

### 1. 의존성 설치

```bash
cd frontend
npm install
```

### 2. 환경 변수 설정 (선택)

`.env` 파일 생성:

```env
VITE_API_URL=http://localhost:8000
```

### 3. 개발 서버 실행

```bash
npm run dev
```

브라우저에서 http://localhost:3000 접속

### 4. 프로덕션 빌드

```bash
npm run build
```

빌드된 파일은 `dist/` 폴더에 생성됩니다.

### 5. 빌드 미리보기

```bash
npm run preview
```

## 주요 기능

- **홈 페이지**: 인기 종목 및 검색 기능
- **종목 상세**: 실시간 차트 및 기본 정보
- **AI 예측**: 머신러닝 기반 주가 예측
- **반응형 디자인**: 모바일/데스크톱 지원
- **다크 모드**: 시스템 설정에 따라 자동 전환

## API 통신

백엔드 API와의 통신은 `src/services/api.js`에서 관리됩니다.

Vite의 프록시 설정으로 CORS 문제를 해결:
- `/api/*` 요청은 자동으로 `http://localhost:8000`으로 프록시됩니다.

## 개발 가이드

### 새 페이지 추가

1. `src/pages/`에 컴포넌트 생성
2. `src/App.jsx`에 라우트 추가

### API 함수 추가

`src/services/api.js`에 새 함수 추가:

```javascript
export const getNewData = async () => {
  const response = await api.get('/api/new-endpoint')
  return response.data
}
```

### 컴포넌트에서 사용

```javascript
import { useQuery } from '@tanstack/react-query'
import { getNewData } from '../services/api'

const { data, isLoading } = useQuery({
  queryKey: ['newData'],
  queryFn: getNewData,
})
```

## 스타일링

- CSS Modules 방식 사용
- 다크/라이트 모드 지원
- 반응형 디자인 (Grid/Flexbox)

## 라이선스

MIT
