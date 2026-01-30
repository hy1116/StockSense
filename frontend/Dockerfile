# 프론트엔드 Dockerfile
# 빌드 스테이지
FROM node:18-alpine AS build

WORKDIR /app

# package.json과 package-lock.json 복사
COPY package*.json ./

# 의존성 설치
RUN npm ci

# 소스 코드 복사
COPY . .

# 프로덕션 빌드
RUN npm run build

# 프로덕션 스테이지 - Nginx로 서빙
FROM nginx:alpine

# Nginx 설정 파일 복사 (SPA 라우팅 지원)
COPY nginx.docker.conf /etc/nginx/conf.d/default.conf

# 빌드된 파일을 Nginx로 복사
COPY --from=build /app/dist /usr/share/nginx/html

# 포트 노출
EXPOSE 80

# Nginx 실행
CMD ["nginx", "-g", "daemon off;"]
