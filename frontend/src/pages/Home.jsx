import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { getHealthCheck } from '../services/api'
import './Home.css'

function Home() {
  const [searchTerm, setSearchTerm] = useState('')

  const { data: health, isLoading } = useQuery({
    queryKey: ['health'],
    queryFn: getHealthCheck,
  })

  const popularStocks = [
    { symbol: 'AAPL', name: 'Apple Inc.' },
    { symbol: 'GOOGL', name: 'Alphabet Inc.' },
    { symbol: 'MSFT', name: 'Microsoft Corporation' },
    { symbol: 'TSLA', name: 'Tesla, Inc.' },
    { symbol: '005930.KS', name: '삼성전자' },
    { symbol: '000660.KS', name: 'SK하이닉스' },
  ]

  const handleSearch = (e) => {
    e.preventDefault()
    if (searchTerm.trim()) {
      window.location.href = `/stock/${searchTerm.toUpperCase()}`
    }
  }

  return (
    <div className="home">
      <section className="hero">
        <h1>StockSense</h1>
        <p>AI 기반 주식 예측 및 분석 시스템</p>
        {!isLoading && health && (
          <p className="status">서버 상태: {health.status}</p>
        )}
      </section>

      <section className="search-section">
        <form onSubmit={handleSearch} className="search-form">
          <input
            type="text"
            placeholder="종목 코드 입력 (예: AAPL, 005930.KS)"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="search-input"
          />
          <button type="submit" className="search-button">
            검색
          </button>
        </form>
      </section>

      <section className="popular-stocks">
        <h2>인기 종목</h2>
        <div className="stock-grid">
          {popularStocks.map((stock) => (
            <Link
              key={stock.symbol}
              to={`/stock/${stock.symbol}`}
              className="stock-card"
            >
              <h3>{stock.symbol}</h3>
              <p>{stock.name}</p>
            </Link>
          ))}
        </div>
      </section>

      <section className="features">
        <h2>주요 기능</h2>
        <div className="feature-grid">
          <div className="feature-card">
            <h3>실시간 데이터</h3>
            <p>주요 주식 시장의 실시간 데이터 제공</p>
          </div>
          <div className="feature-card">
            <h3>AI 예측</h3>
            <p>머신러닝 기반 주가 예측</p>
          </div>
          <div className="feature-card">
            <h3>차트 분석</h3>
            <p>다양한 기술적 지표 시각화</p>
          </div>
        </div>
      </section>
    </div>
  )
}

export default Home
