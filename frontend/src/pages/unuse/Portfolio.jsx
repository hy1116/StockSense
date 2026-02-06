import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { getPortfolio } from '../services/api'
import useBalanceWebSocket from '../hooks/useBalanceWebSocket'
import './Portfolio.css'

function Portfolio() {
  const [assetExpanded, setAssetExpanded] = useState(false)

  // WebSocket 실시간 잔고
  const { data: wsData, connected: wsConnected } = useBalanceWebSocket(true)

  // HTTP 폴링 (WebSocket 연결 안 됐을 때 fallback)
  const { data: httpData, isLoading, error } = useQuery({
    queryKey: ['portfolio'],
    queryFn: getPortfolio,
    refetchInterval: wsConnected ? false : 10000,
  })

  const portfolio = wsData || httpData

  const formatNumber = (num) => {
    if (!num && num !== 0) return '0'
    return new Intl.NumberFormat('ko-KR').format(num)
  }

  const formatRate = (rate) => {
    const sign = rate >= 0 ? '+' : ''
    return `${sign}${rate.toFixed(2)}%`
  }

  const getPriceChangeClass = (change) => {
    if (change > 0) return 'price-up'
    if (change < 0) return 'price-down'
    return ''
  }

  if (isLoading && !wsData) return <div className="portfolio"><div className="loading">로딩 중...</div></div>
  if (error && !wsData) return <div className="portfolio"><div className="error">오류: {error.message}</div></div>

  return (
    <div className="portfolio">
      {/* 자산 Hero — 토스 스타일 */}
      <section className="asset-hero">
        <div className="asset-hero-main">
          <span className="asset-hero-label">보유주식</span>
          <span className="asset-hero-price">
            {formatNumber(portfolio?.total_asset || 0)}<span className="asset-hero-unit">원</span>
          </span>
          <div className={`asset-hero-change ${getPriceChangeClass(portfolio?.total_profit_rate)}`}>
            {portfolio?.total_profit_rate > 0 ? '+' : ''}{(portfolio?.total_profit_rate || 0).toFixed(2)}%
            {' '}({portfolio?.total_profit_loss > 0 ? '+' : ''}{formatNumber(portfolio?.total_profit_loss || 0)}원)
          </div>
          <span className="asset-hero-sub">
            총 투자금 {formatNumber((portfolio?.total_asset || 0) - (portfolio?.total_profit_loss || 0))}원
          </span>
        </div>
        <button
          className="asset-hero-toggle"
          onClick={() => setAssetExpanded(!assetExpanded)}
        >
          {assetExpanded ? '접기' : '상세'}
          <span className={`asset-arrow ${assetExpanded ? 'expanded' : ''}`}>&#9662;</span>
        </button>
        {assetExpanded && (
          <div className="asset-detail">
            <div className="asset-detail-row">
              <span className="asset-detail-label">보유 현금</span>
              <span className="asset-detail-value">{formatNumber(portfolio?.cash || 0)}원</span>
            </div>
            <div className="asset-detail-row">
              <span className="asset-detail-label">주식 평가액</span>
              <span className="asset-detail-value">{formatNumber(portfolio?.stock_eval_amount || 0)}원</span>
            </div>
            <div className="asset-detail-row">
              <span className="asset-detail-label">평가 손익</span>
              <span className={`asset-detail-value ${getPriceChangeClass(portfolio?.total_profit_rate)}`}>
                {portfolio?.total_profit_loss > 0 ? '+' : ''}{formatNumber(portfolio?.total_profit_loss || 0)}원
              </span>
            </div>
          </div>
        )}
      </section>

      {/* 보유 종목 리스트 */}
      <section className="portfolio-holdings">
        <h2>보유 종목</h2>
        {portfolio?.holdings && portfolio.holdings.length > 0 ? (
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
            <span className="empty-sub">종목 상세 페이지에서 주문할 수 있습니다</span>
          </div>
        )}
      </section>
    </div>
  )
}

export default Portfolio
