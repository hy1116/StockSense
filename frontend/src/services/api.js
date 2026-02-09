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

export const getFluctuationStocks = async (limit = 10, sort = 0) => {
  const response = await api.get('/api/portfolio/fluctuation-stocks', {
    params: { limit, sort }
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

export const authRegister = async (data) => {
  const response = await api.post('/api/auth/register', {
    username: data.username,
    nickname: data.nickname,
    password: data.password,
    kis_api_key: data.kisApiKey,
    kis_api_secret: data.kisApiSecret,
    kis_account_no: data.kisAccountNo,
    kis_account_product_code: data.kisAccountProductCode || '01'
  })
  return response.data
}

export const authLogin = async (credentials) => {
  const response = await api.post('/api/auth/login', {
    username: credentials.username,
    password: credentials.password
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

export const checkUsername = async (username) => {
  const response = await api.get(`/api/auth/check-username/${username}`)
  return response.data
}

export const checkNickname = async (nickname) => {
  const response = await api.get(`/api/auth/check-nickname/${nickname}`)
  return response.data
}

// ===== Prediction API =====

export const getPrediction = async (stockCode) => {
  const response = await api.post('/api/prediction/', {
    stock_code: stockCode.toUpperCase()
  })
  return response.data
}

// ===== Prediction Accuracy API =====

export const getPredictionAccuracy = async (stockCode, days = 30) => {
  const response = await api.get(`/api/prediction/${stockCode.toUpperCase()}/accuracy`, {
    params: { days }
  })
  return response.data
}

// ===== Stock Detail API =====

export const getStockDetail = async (symbol) => {
  const response = await api.get(`/api/portfolio/stock/${symbol}/detail`)
  return response.data
}

export const getStockIntraday = async (symbol, interval = 1) => {
  const response = await api.get(`/api/portfolio/stock/${symbol}/intraday`, {
    params: { interval }
  })
  return response.data
}

// ===== Comments API =====

export const getComments = async (stockCode, page = 1, pageSize = 20) => {
  const response = await api.get(`/api/comments/${stockCode}`, {
    params: { page, page_size: pageSize }
  })
  return response.data
}

export const createComment = async (stockCode, content) => {
  const response = await api.post(`/api/comments/${stockCode}`, { content })
  return response.data
}

export const updateComment = async (commentId, content) => {
  const response = await api.put(`/api/comments/${commentId}`, { content })
  return response.data
}

export const deleteComment = async (commentId) => {
  const response = await api.delete(`/api/comments/${commentId}`)
  return response.data
}

// ===== News API =====

export const getStockNews = async (stockCode, page = 1, pageSize = 10, days = null) => {
  const params = { page, page_size: pageSize }
  if (days) params.days = days
  const response = await api.get(`/api/news/${stockCode}`, { params })
  return response.data
}

export const getNewsSentimentStats = async (stockCode, days = 7) => {
  const response = await api.get(`/api/news/${stockCode}/stats`, {
    params: { days }
  })
  return response.data
}

// ===== Watchlist API =====

export const getWatchlist = async () => {
  const response = await api.get('/api/watchlist/')
  return response.data
}

export const addToWatchlist = async (stockCode, stockName, market) => {
  const response = await api.post(`/api/watchlist/${stockCode}`, {
    stock_name: stockName,
    market: market
  })
  return response.data
}

export const removeFromWatchlist = async (stockCode) => {
  const response = await api.delete(`/api/watchlist/${stockCode}`)
  return response.data
}

export const checkWatchlist = async (stockCode) => {
  const response = await api.get(`/api/watchlist/check/${stockCode}`)
  return response.data
}

export default api
