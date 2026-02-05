import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { getPortfolio, buyStock, sellStock, getStockPrice } from '../services/api'
import useBalanceWebSocket from '../hooks/useBalanceWebSocket'
import './Portfolio.css'

function Portfolio() {
  const queryClient = useQueryClient()
  const [activeTab, setActiveTab] = useState('holdings')
  const [orderForm, setOrderForm] = useState({
    stockCode: '',
    quantity: '',
    price: '',
    orderType: '00'
  })

  // WebSocket 실시간 잔고
  const { data: wsData, connected: wsConnected } = useBalanceWebSocket(true)

  // HTTP 폴링 (WebSocket 연결 안 됐을 때 fallback)
  const { data: httpData, isLoading, error } = useQuery({
    queryKey: ['portfolio'],
    queryFn: getPortfolio,
    refetchInterval: wsConnected ? false : 10000,
  })

  const portfolio = wsData || httpData

  const buyMutation = useMutation({
    mutationFn: buyStock,
    onSuccess: () => {
      queryClient.invalidateQueries(['portfolio'])
      alert('매수 주문이 접수되었습니다')
      resetForm()
    },
    onError: (error) => {
      alert(`매수 실패: ${error.message}`)
    }
  })

  const sellMutation = useMutation({
    mutationFn: sellStock,
    onSuccess: () => {
      queryClient.invalidateQueries(['portfolio'])
      alert('매도 주문이 접수되었습니다')
      resetForm()
    },
    onError: (error) => {
      alert(`매도 실패: ${error.message}`)
    }
  })

  const resetForm = () => {
    setOrderForm({
      stockCode: '',
      quantity: '',
      price: '',
      orderType: '00'
    })
  }

  const handleBuy = (e) => {
    e.preventDefault()
    buyMutation.mutate({
      stock_code: orderForm.stockCode,
      quantity: parseInt(orderForm.quantity),
      price: parseInt(orderForm.price),
      order_type: orderForm.orderType
    })
  }

  const handleSell = (e) => {
    e.preventDefault()
    sellMutation.mutate({
      stock_code: orderForm.stockCode,
      quantity: parseInt(orderForm.quantity),
      price: parseInt(orderForm.price),
      order_type: orderForm.orderType
    })
  }

  if (isLoading && !wsData) return <div className="loading">로딩 중...</div>
  if (error && !wsData) return <div className="error">오류: {error.message}</div>

  const formatNumber = (num) => {
    return new Intl.NumberFormat('ko-KR').format(num)
  }

  const formatRate = (rate) => {
    const sign = rate >= 0 ? '+' : ''
    return `${sign}${rate.toFixed(2)}%`
  }

  return (
    <div className="portfolio">
      <h1>Portfolio</h1>

      <section className="portfolio-summary card">
        <h2>계좌 요약</h2>
        <div className="summary-grid">
          <div className="summary-item">
            <span className="label">총 자산</span>
            <span className="value">{formatNumber(portfolio?.total_asset || 0)}원</span>
          </div>
          <div className="summary-item">
            <span className="label">예수금</span>
            <span className="value">{formatNumber(portfolio?.cash || 0)}원</span>
          </div>
          <div className="summary-item">
            <span className="label">주식 평가액</span>
            <span className="value">{formatNumber(portfolio?.stock_eval_amount || 0)}원</span>
          </div>
          <div className="summary-item">
            <span className="label">평가손익</span>
            <span className={`value ${portfolio?.total_profit_loss >= 0 ? 'profit' : 'loss'}`}>
              {formatNumber(portfolio?.total_profit_loss || 0)}원
            </span>
          </div>
          <div className="summary-item">
            <span className="label">수익률</span>
            <span className={`value ${portfolio?.total_profit_rate >= 0 ? 'profit' : 'loss'}`}>
              {formatRate(portfolio?.total_profit_rate || 0)}
            </span>
          </div>
        </div>
      </section>

      <div className="tabs">
        <button
          className={activeTab === 'holdings' ? 'active' : ''}
          onClick={() => setActiveTab('holdings')}
        >
          보유 종목
        </button>
        <button
          className={activeTab === 'trade' ? 'active' : ''}
          onClick={() => setActiveTab('trade')}
        >
          주문하기
        </button>
      </div>

      {activeTab === 'holdings' && (
        <section className="holdings card">
          <h2>보유 종목</h2>
          {portfolio?.holdings && portfolio.holdings.length > 0 ? (
            <div className="table-container">
              <table className="holdings-table">
                <thead>
                  <tr>
                    <th>종목코드</th>
                    <th>종목명</th>
                    <th>보유수량</th>
                    <th>평균매입가</th>
                    <th>현재가</th>
                    <th>평가금액</th>
                    <th>평가손익</th>
                    <th>수익률</th>
                  </tr>
                </thead>
                <tbody>
                  {portfolio.holdings.map((holding) => (
                    <tr key={holding.stock_code}>
                      <td>{holding.stock_code}</td>
                      <td>{holding.stock_name}</td>
                      <td>{formatNumber(holding.quantity)}</td>
                      <td>{formatNumber(holding.avg_price)}원</td>
                      <td>{formatNumber(holding.current_price)}원</td>
                      <td>{formatNumber(holding.eval_amount)}원</td>
                      <td className={holding.profit_loss >= 0 ? 'profit' : 'loss'}>
                        {formatNumber(holding.profit_loss)}원
                      </td>
                      <td className={holding.profit_rate >= 0 ? 'profit' : 'loss'}>
                        {formatRate(holding.profit_rate)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="empty-message">보유 종목이 없습니다</p>
          )}
        </section>
      )}

      {activeTab === 'trade' && (
        <section className="trade card">
          <h2>주문하기</h2>
          <form className="order-form">
            <div className="form-group">
              <label htmlFor="stockCode">종목코드</label>
              <input
                type="text"
                id="stockCode"
                value={orderForm.stockCode}
                onChange={(e) => setOrderForm({ ...orderForm, stockCode: e.target.value })}
                placeholder="6자리 (예: 005930)"
                maxLength={6}
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="quantity">수량</label>
              <input
                type="number"
                id="quantity"
                value={orderForm.quantity}
                onChange={(e) => setOrderForm({ ...orderForm, quantity: e.target.value })}
                placeholder="주문 수량"
                min="1"
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="price">가격</label>
              <input
                type="number"
                id="price"
                value={orderForm.price}
                onChange={(e) => setOrderForm({ ...orderForm, price: e.target.value })}
                placeholder="주문 가격 (0: 시장가)"
                min="0"
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="orderType">주문 유형</label>
              <select
                id="orderType"
                value={orderForm.orderType}
                onChange={(e) => setOrderForm({ ...orderForm, orderType: e.target.value })}
              >
                <option value="00">지정가</option>
                <option value="01">시장가</option>
              </select>
            </div>

            <div className="button-group">
              <button
                type="button"
                onClick={handleBuy}
                className="btn-buy"
                disabled={buyMutation.isPending}
              >
                {buyMutation.isPending ? '주문 중...' : '매수'}
              </button>
              <button
                type="button"
                onClick={handleSell}
                className="btn-sell"
                disabled={sellMutation.isPending}
              >
                {sellMutation.isPending ? '주문 중...' : '매도'}
              </button>
            </div>
          </form>
        </section>
      )}
    </div>
  )
}

export default Portfolio
