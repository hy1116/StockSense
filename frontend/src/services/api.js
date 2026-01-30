import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || ''

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
})

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

export default api
