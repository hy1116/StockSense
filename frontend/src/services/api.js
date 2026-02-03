import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || ''

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: true, // HttpOnly Cookie 전송을 위해 필요
})

// 토큰을 저장할 변수 (메모리에만 저장)
let accessToken = null

// 토큰 설정 함수
export const setAccessToken = (token) => {
  accessToken = token
}

// 토큰 가져오기 함수
export const getAccessToken = () => accessToken

// 토큰 삭제 함수
export const clearAccessToken = () => {
  accessToken = null
}

// Request interceptor - 토큰 자동 추가
api.interceptors.request.use(
  (config) => {
    if (accessToken) {
      config.headers.Authorization = `Bearer ${accessToken}`
    }
    return config
  },
  (error) => Promise.reject(error)
)

// Response interceptor - 401 에러 처리
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // 토큰 만료 시 처리
      clearAccessToken()
      // 필요시 로그인 페이지로 리다이렉트
    }
    return Promise.reject(error)
  }
)

export const getHealthCheck = async () => {
  const response = await api.get('/health')
  return response.data
}

export const getStockData = async (symbol, period = '1mo') => {
  const response = await api.get(`/api/stocks/${symbol}`, {
    params: { period }
  })
  return response.data
}

export const predictStock = async (symbol, days = 30) => {
  const response = await api.post('/api/predict', {
    symbol,
    days
  })
  return response.data
}

export const getPortfolio = async () => {
  const response = await api.get('/api/portfolio/balance')
  return response.data
}

export const getStockPrice = async (stockCode) => {
  const response = await api.get(`/api/portfolio/stock/${stockCode}`)
  return response.data
}

export const buyStock = async (orderData) => {
  const response = await api.post('/api/portfolio/buy', orderData)
  return response.data
}

export const sellStock = async (orderData) => {
  const response = await api.post('/api/portfolio/sell', orderData)
  return response.data
}

export const getOrderHistory = async () => {
  const response = await api.get('/api/portfolio/orders')
  return response.data
}

export const getTopStocks = async (limit = 10) => {
  const response = await api.get('/api/portfolio/top-stocks', {
    params: { limit }
  })
  return response.data
}

export const getMarketCapStocks = async (limit = 10) => {
  const response = await api.get('/api/portfolio/market-cap-stocks', {
    params: { limit }
  })
  return response.data
}

export const searchStocks = async (query, limit = 10) => {
  const response = await api.get('/api/portfolio/search', {
    params: { q: query, limit }
  })
  return response.data
}

// ===== Auth API =====

export const authLogin = async (credentials) => {
  const response = await api.post('/api/auth/login', {
    api_key: credentials.apiKey,
    api_secret: credentials.apiSecret,
    account_no: credentials.accountNo,
    account_product_code: credentials.accountProductCode || '01'
  })
  return response.data
}

export const authLogout = async () => {
  const response = await api.post('/api/auth/logout')
  return response.data
}

export const authCheck = async () => {
  const response = await api.get('/api/auth/check')
  return response.data
}

export const authGetMe = async () => {
  const response = await api.get('/api/auth/me')
  return response.data
}

export default api
