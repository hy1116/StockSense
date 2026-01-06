import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { getHealthCheck, getTopStocks, getPortfolio } from '../services/api'
import './Home.css'

function Home() {
  const [searchTerm, setSearchTerm] = useState('')

  const { data: health, isLoading } = useQuery({
    queryKey: ['health'],
    queryFn: getHealthCheck,
  })

  const { data: topStocksData, isLoading: isLoadingStocks } = useQuery({
    queryKey: ['topStocks'],
    queryFn: () => getTopStocks(8),
    refetchInterval: 60000, // 1분마다 새로고침
  })

  const { data: portfolio, isLoading: isLoadingPortfolio } = useQuery({
    queryKey: ['portfolio'],
    queryFn: getPortfolio,
    refetchInterval: 300000, // 5분마다 새로고침
  })

  const formatNumber = (num) => {
    if (!num) return '0'
    return num.toLocaleString('ko-KR')
  }

  const getPriceChangeClass = (change) => {
    if (change > 0) return 'price-up'
    if (change < 0) return 'price-down'
    return ''
  }

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
            placeholder="종목 코드 입력 (예: 005930, 035420)"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="search-input"
          />
          <button type="submit" className="search-button">
            검색
          </button>
        </form>
      </section>

      {/* 내 자산 섹션 */}
      <section className="portfolio-summary">
        <h2>내 자산</h2>
        {isLoadingPortfolio ? (
          <div className="loading">자산 정보를 불러오는 중...</div>
        ) : portfolio ? (
          <div className="portfolio-cards">
            <div className="portfolio-card">
              <div className="portfolio-label">총 자산</div>
              <div className="portfolio-value">{formatNumber(portfolio.total_asset)}원</div>
            </div>
            <div className="portfolio-card">
              <div className="portfolio-label">보유 현금</div>
              <div className="portfolio-value">{formatNumber(portfolio.cash)}원</div>
            </div>
            <div className="portfolio-card">
              <div className="portfolio-label">주식 평가액</div>
              <div className="portfolio-value">{formatNumber(portfolio.stock_eval_amount)}원</div>
            </div>
            <div className="portfolio-card">
              <div className="portfolio-label">평가 손익</div>
              <div className={`portfolio-value ${getPriceChangeClass(portfolio.total_profit_rate)}`}>
                {formatNumber(portfolio.total_profit_loss)}원
                <span className="portfolio-rate">
                  ({portfolio.total_profit_rate > 0 ? '+' : ''}{portfolio.total_profit_rate.toFixed(2)}%)
                </span>
              </div>
            </div>
          </div>
        ) : (
          <div className="error">자산 정보를 불러올 수 없습니다</div>
        )}
      </section>

      <section className="popular-stocks">
        <h2>거래량 상위 종목</h2>
        {isLoadingStocks ? (
          <div className="loading">종목 정보를 불러오는 중...</div>
        ) : topStocksData?.stocks ? (
          <div className="stock-grid">
            {topStocksData.stocks.map((stock) => (
              <Link
                key={stock.stock_code}
                to={`/stock/${stock.stock_code}`}
                className="stock-card"
              >
                <div className="stock-rank">#{stock.rank}</div>
                <h3>{stock.stock_name}</h3>
                <div className="stock-price">
                  {formatNumber(stock.current_price)}원
                </div>
                <div className={`stock-change ${getPriceChangeClass(stock.change_rate)}`}>
                  {stock.change_rate > 0 ? '+' : ''}
                  {stock.change_rate.toFixed(2)}%
                </div>
                {/* <div className="stock-market-cap">
                  거래량 순위: {stock.market_cap}
                </div> */}
                <div className="stock-code">{stock.stock_code}</div>
              </Link>
            ))}
          </div>
        ) : (
          <div className="error">종목 정보를 불러올 수 없습니다</div>
        )}
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
