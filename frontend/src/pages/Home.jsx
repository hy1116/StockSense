import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { getHealthCheck, getTopStocks, getMarketCapStocks, getPortfolio } from '../services/api'
import './Home.css'

function Home() {
  const [searchTerm, setSearchTerm] = useState('')
  const [activeTab, setActiveTab] = useState('volume') // 'volume' | 'marketCap' | 'holdings'

  const { data: health, isLoading } = useQuery({
    queryKey: ['health'],
    queryFn: getHealthCheck,
  })

  const { data: topStocksData, isLoading: isLoadingStocks } = useQuery({
    queryKey: ['topStocks'],
    queryFn: () => getTopStocks(10),
    refetchInterval: 60000,
  })

  const { data: marketCapData, isLoading: isLoadingMarketCap } = useQuery({
    queryKey: ['marketCapStocks'],
    queryFn: () => getMarketCapStocks(10),
    refetchInterval: 60000,
  })

  const { data: portfolio, isLoading: isLoadingPortfolio } = useQuery({
    queryKey: ['portfolio'],
    queryFn: getPortfolio,
    refetchInterval: 300000,
  })

  const formatNumber = (num) => {
    if (!num) return '0'
    return num.toLocaleString('ko-KR')
  }

  const formatMarketCap = (num) => {
    if (!num) return '-'
    const 억 = Math.floor(num / 100000000)
    if (억 >= 10000) {
      return `${(억 / 10000).toFixed(1)}조`
    }
    return `${formatNumber(억)}억`
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

  const currentStocks = activeTab === 'volume' ? topStocksData?.stocks : marketCapData?.stocks
  const isLoadingCurrentStocks = activeTab === 'volume' ? isLoadingStocks : isLoadingMarketCap

  return (
    <div className="home">
      <section className="hero">
        <h1>StockSense</h1>
        <p>AI 기반 주식 예측 및 분석 시스템</p>
        {!isLoading && health && (
          <span className="status-badge">서버 정상</span>
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

      {/* 상위 종목 섹션 - 탭 형태 */}
      <section className="ranking-section">
        <div className="ranking-header">
          <div className="tab-buttons">
            <button
              className={`tab-button ${activeTab === 'volume' ? 'active' : ''}`}
              onClick={() => setActiveTab('volume')}
            >
              거래량 상위
            </button>
            <button
              className={`tab-button ${activeTab === 'marketCap' ? 'active' : ''}`}
              onClick={() => setActiveTab('marketCap')}
            >
              시가총액 상위
            </button>
            <button
              className={`tab-button ${activeTab === 'holdings' ? 'active' : ''}`}
              onClick={() => setActiveTab('holdings')}
            >
              보유 종목
            </button>
          </div>
        </div>

        {/* 거래량/시총 상위 탭 */}
        {(activeTab === 'volume' || activeTab === 'marketCap') && (
          isLoadingCurrentStocks ? (
            <div className="loading">종목 정보를 불러오는 중...</div>
          ) : currentStocks ? (
            <div className="stock-list">
              <div className="stock-list-header">
                <span className="col-rank">순위</span>
                <span className="col-name">종목명</span>
                <span className="col-price">현재가</span>
                <span className="col-change">등락률</span>
                {activeTab === 'marketCap' && <span className="col-marketcap">시가총액</span>}
              </div>
              {currentStocks.map((stock) => (
                <Link
                  key={stock.stock_code}
                  to={`/stock/${stock.stock_code}`}
                  className="stock-list-item"
                >
                  <span className="col-rank">
                    <span className="rank-badge">{stock.rank}</span>
                  </span>
                  <span className="col-name">
                    <span className="stock-name">{stock.stock_name}</span>
                    <span className="stock-code">{stock.stock_code}</span>
                  </span>
                  <span className="col-price">{formatNumber(stock.current_price)}원</span>
                  <span className={`col-change ${getPriceChangeClass(stock.change_rate)}`}>
                    {stock.change_rate > 0 ? '+' : ''}{stock.change_rate.toFixed(2)}%
                  </span>
                  {activeTab === 'marketCap' && (
                    <span className="col-marketcap">{formatMarketCap(stock.market_cap)}</span>
                  )}
                </Link>
              ))}
            </div>
          ) : (
            <div className="error">종목 정보를 불러올 수 없습니다</div>
          )
        )}

        {/* 보유 종목 탭 */}
        {activeTab === 'holdings' && (
          isLoadingPortfolio ? (
            <div className="loading">보유 종목을 불러오는 중...</div>
          ) : portfolio?.holdings && portfolio.holdings.length > 0 ? (
            <div className="stock-list">
              <div className="stock-list-header holdings-header">
                <span className="col-rank">No.</span>
                <span className="col-name">종목명</span>
                <span className="col-price">현재가</span>
                <span className="col-change">수익률</span>
                <span className="col-quantity">보유수량</span>
              </div>
              {portfolio.holdings.map((holding, index) => (
                <Link
                  key={holding.stock_code}
                  to={`/stock/${holding.stock_code}`}
                  className="stock-list-item holdings-item"
                >
                  <span className="col-rank">
                    <span className="rank-badge">{index + 1}</span>
                  </span>
                  <span className="col-name">
                    <span className="stock-name">{holding.stock_name}</span>
                    <span className="stock-code">{holding.stock_code}</span>
                  </span>
                  <span className="col-price">{formatNumber(holding.current_price)}원</span>
                  <span className={`col-change ${getPriceChangeClass(holding.profit_rate)}`}>
                    {holding.profit_rate > 0 ? '+' : ''}{holding.profit_rate.toFixed(2)}%
                  </span>
                  <span className="col-quantity">{formatNumber(holding.quantity)}주</span>
                </Link>
              ))}
            </div>
          ) : (
            <div className="empty-holdings">
              <div className="empty-icon">📭</div>
              <p>보유 종목이 없습니다</p>
              <Link to="/portfolio" className="go-trade-btn">주문하러 가기</Link>
            </div>
          )
        )}
      </section>

      <section className="features">
        <h2>주요 기능</h2>
        <div className="feature-grid">
          <div className="feature-card">
            <div className="feature-icon">📊</div>
            <h3>실시간 데이터</h3>
            <p>주요 주식 시장의 실시간 데이터 제공</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">🤖</div>
            <h3>AI 예측</h3>
            <p>머신러닝 기반 주가 예측</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">📈</div>
            <h3>차트 분석</h3>
            <p>다양한 기술적 지표 시각화</p>
          </div>
        </div>
      </section>
    </div>
  )
}

export default Home
