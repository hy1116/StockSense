import { useParams } from 'react-router-dom'
import { useState, useEffect, useRef, useCallback } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useAuth } from '../contexts/AuthContext'
import { getStockDetail, getComments, createComment, updateComment, deleteComment, getStockNews, buyStock, sellStock } from '../services/api'
import { createChart } from 'lightweight-charts'
import './StockDetail.css'

function StockDetail() {
  const { symbol } = useParams()
  const { isLoggedIn, user } = useAuth()
  const queryClient = useQueryClient()
  const [stockData, setStockData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [period, setPeriod] = useState('D')

  // 댓글 관련 state
  const [comments, setComments] = useState([])
  const [commentLoading, setCommentLoading] = useState(false)
  const [commentPage, setCommentPage] = useState(1)
  const [hasMoreComments, setHasMoreComments] = useState(false)
  const [totalComments, setTotalComments] = useState(0)
  const [newComment, setNewComment] = useState('')
  const [editingCommentId, setEditingCommentId] = useState(null)
  const [editingContent, setEditingContent] = useState('')
  const [commentSubmitting, setCommentSubmitting] = useState(false)

  // 뉴스 관련 state
  const [news, setNews] = useState([])
  const [newsLoading, setNewsLoading] = useState(false)
  const [newsPage, setNewsPage] = useState(1)
  const [hasMoreNews, setHasMoreNews] = useState(false)
  const [totalNews, setTotalNews] = useState(0)

  // 주문 관련 state
  const [tradeTab, setTradeTab] = useState('buy')
  const [orderForm, setOrderForm] = useState({
    quantity: '',
    price: '',
    orderType: '00'
  })

  const buyMutation = useMutation({
    mutationFn: buyStock,
    onSuccess: () => {
      queryClient.invalidateQueries(['portfolio'])
      alert('매수 주문이 접수되었습니다')
      setOrderForm(prev => ({ ...prev, quantity: '' }))
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
      setOrderForm(prev => ({ ...prev, quantity: '' }))
    },
    onError: (error) => {
      alert(`매도 실패: ${error.message}`)
    }
  })

  const handleOrder = (e) => {
    e.preventDefault()
    const orderData = {
      stock_code: symbol,
      quantity: parseInt(orderForm.quantity),
      price: parseInt(orderForm.price),
      order_type: orderForm.orderType
    }
    if (tradeTab === 'buy') {
      buyMutation.mutate(orderData)
    } else {
      sellMutation.mutate(orderData)
    }
  }

  // 차트 관련 ref
  const chartContainerRef = useRef(null)
  const chartInstanceRef = useRef(null)
  const candlestickSeriesRef = useRef(null)
  const volumeSeriesRef = useRef(null)
  const autoRefreshIntervalRef = useRef(null)

  // 주식 상세 정보 조회
  const fetchStockDetail = async (showLoading = true) => {
    if (showLoading) {
      setLoading(true)
    }
    setError(null)

    try {
      const data = await getStockDetail(symbol, period)
      setStockData(data)
    } catch (err) {
      setError(err.response?.data?.detail || '데이터를 불러오는데 실패했습니다')
      console.error('Error fetching stock detail:', err)
    } finally {
      if (showLoading) {
        setLoading(false)
      }
    }
  }

  // 댓글 목록 조회
  const fetchComments = async (page = 1) => {
    setCommentLoading(true)
    try {
      const response = await getComments(symbol, page, 20)
      if (page === 1) {
        setComments(response.comments)
      } else {
        setComments(prev => [...prev, ...response.comments])
      }
      setCommentPage(page)
      setHasMoreComments(response.has_more)
      setTotalComments(response.total)
    } catch (err) {
      console.error('Error fetching comments:', err)
    } finally {
      setCommentLoading(false)
    }
  }

  // 댓글 작성
  const handleSubmitComment = async (e) => {
    e.preventDefault()
    if (!newComment.trim() || commentSubmitting) return

    setCommentSubmitting(true)
    try {
      const response = await createComment(symbol, newComment.trim())
      setComments(prev => [response, ...prev])
      setTotalComments(prev => prev + 1)
      setNewComment('')
    } catch (err) {
      alert(err.response?.data?.detail || '댓글 작성에 실패했습니다')
    } finally {
      setCommentSubmitting(false)
    }
  }

  // 댓글 수정
  const handleUpdateComment = async (commentId) => {
    if (!editingContent.trim() || commentSubmitting) return

    setCommentSubmitting(true)
    try {
      const response = await updateComment(commentId, editingContent.trim())
      setComments(prev => prev.map(c => c.id === commentId ? response : c))
      setEditingCommentId(null)
      setEditingContent('')
    } catch (err) {
      alert(err.response?.data?.detail || '댓글 수정에 실패했습니다')
    } finally {
      setCommentSubmitting(false)
    }
  }

  // 댓글 삭제
  const handleDeleteComment = async (commentId) => {
    if (!confirm('댓글을 삭제하시겠습니까?')) return

    try {
      await deleteComment(commentId)
      setComments(prev => prev.filter(c => c.id !== commentId))
      setTotalComments(prev => prev - 1)
    } catch (err) {
      alert(err.response?.data?.detail || '댓글 삭제에 실패했습니다')
    }
  }

  // 뉴스 목록 조회
  const fetchNews = async (page = 1) => {
    setNewsLoading(true)
    try {
      const response = await getStockNews(symbol, page, 3, 7)
      if (page === 1) {
        setNews(response.news)
      } else {
        setNews(prev => [...prev, ...response.news])
      }
      setNewsPage(page)
      setHasMoreNews(response.has_more)
      setTotalNews(response.total)
    } catch (err) {
      console.error('Error fetching news:', err)
    } finally {
      setNewsLoading(false)
    }
  }

  // 뉴스 시간 포맷
  const formatNewsTime = (dateString) => {
    if (!dateString) return ''
    const date = new Date(dateString)
    if (isNaN(date.getTime())) return ''
    const now = new Date()
    const diff = (now - date) / 1000

    if (diff < 0) {
      return date.toLocaleString('ko-KR', {
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      })
    }
    if (diff < 60) return '방금 전'
    if (diff < 3600) return `${Math.floor(diff / 60)}분 전`
    if (diff < 86400) return `${Math.floor(diff / 3600)}시간 전`
    if (diff < 604800) return `${Math.floor(diff / 86400)}일 전`

    return date.toLocaleDateString('ko-KR', {
      month: 'short',
      day: 'numeric'
    })
  }

  // 댓글 시간 포맷
  const formatCommentTime = (dateString) => {
    const date = new Date(dateString)
    const now = new Date()
    const diff = (now - date) / 1000

    if (diff < 60) return '방금 전'
    if (diff < 3600) return `${Math.floor(diff / 60)}분 전`
    if (diff < 86400) return `${Math.floor(diff / 3600)}시간 전`
    if (diff < 604800) return `${Math.floor(diff / 86400)}일 전`

    return date.toLocaleDateString('ko-KR', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    })
  }

  const formatNumber = (num) => {
    if (!num && num !== 0) return '-'
    return num.toLocaleString('ko-KR')
  }

  const formatPrice = (price) => {
    if (!price && price !== 0) return '-'
    return `${formatNumber(price)}원`
  }

  // 시가총액 포맷 (억 단위)
  const formatMarketCap = (cap) => {
    if (!cap) return '-'
    if (cap >= 100000000) {
      return `${(cap / 100000000).toFixed(1)}조`
    }
    if (cap >= 10000) {
      return `${(cap / 10000).toFixed(0)}억`
    }
    return formatNumber(cap)
  }

  // 거래량 포맷 (만 단위)
  const formatVolume = (vol) => {
    if (!vol) return '-'
    if (vol >= 100000000) {
      return `${(vol / 100000000).toFixed(1)}억`
    }
    if (vol >= 10000) {
      return `${(vol / 10000).toFixed(0)}만`
    }
    return formatNumber(vol)
  }

  const getPriceChangeClass = (change) => {
    if (change > 0) return 'price-up'
    if (change < 0) return 'price-down'
    return ''
  }

  const getChangeIcon = (change) => {
    if (change > 0) return '▲'
    if (change < 0) return '▼'
    return '-'
  }

  // lightweight-charts 초기화
  const initChart = useCallback(() => {
    if (!chartContainerRef.current || !stockData) return

    const chartData = stockData.chart_data
    if (!chartData || chartData.length === 0) return

    // 기존 차트 제거
    if (chartInstanceRef.current) {
      chartInstanceRef.current.remove()
      chartInstanceRef.current = null
      candlestickSeriesRef.current = null
      volumeSeriesRef.current = null
    }

    const sortedData = [...chartData].reverse()

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 400,
      layout: {
        background: { color: 'transparent' },
        textColor: '#6b7280',
        fontSize: 11,
      },
      grid: {
        vertLines: { color: 'rgba(255,255,255,0.04)' },
        horzLines: { color: 'rgba(255,255,255,0.04)' },
      },
      crosshair: {
        mode: 1,
        vertLine: { color: 'rgba(255,255,255,0.15)', width: 1, style: 2 },
        horzLine: { color: 'rgba(255,255,255,0.15)', width: 1, style: 2 },
      },
      rightPriceScale: {
        borderVisible: false,
        scaleMargins: { top: 0.08, bottom: 0.2 },
      },
      timeScale: {
        borderVisible: false,
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 2,
        barSpacing: 10,
        minBarSpacing: 3,
      },
    })

    chartInstanceRef.current = chart

    const candleSeries = chart.addCandlestickSeries({
      upColor: '#ef4444',
      downColor: '#3b82f6',
      borderVisible: true,
      wickUpColor: '#ef4444',
      wickDownColor: '#3b82f6',
      borderUpColor: '#ef4444',
      borderDownColor: '#3b82f6',
    })
    candlestickSeriesRef.current = candleSeries

    const volumeSeries = chart.addHistogramSeries({
      color: '#3b82f6',
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
    })

    chart.priceScale('volume').applyOptions({
      scaleMargins: { top: 0.82, bottom: 0 },
    })

    volumeSeriesRef.current = volumeSeries

    let candleData = []
    let volumeData = []
    const seenTimes = new Set()

    sortedData.forEach(d => {
      if (!d.date || d.date.length < 8) return

      const time = `${d.date.slice(0, 4)}-${d.date.slice(4, 6)}-${d.date.slice(6, 8)}`

      if (seenTimes.has(time)) return
      seenTimes.add(time)

      if (d.open > 0 && d.high > 0 && d.low > 0 && d.close > 0) {
        candleData.push({
          time: time,
          open: d.open,
          high: d.high,
          low: d.low,
          close: d.close,
        })

        volumeData.push({
          time: time,
          value: d.volume || 0,
          color: d.close >= d.open ? 'rgba(239,68,68,0.35)' : 'rgba(59,130,246,0.35)',
        })
      }
    })

    candleSeries.setData(candleData)
    volumeSeries.setData(volumeData)

    chart.timeScale().fitContent()

    const handleResize = () => {
      if (chartContainerRef.current && chartInstanceRef.current) {
        chartInstanceRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        })
      }
    }

    const resizeObserver = new ResizeObserver(handleResize)
    if (chartContainerRef.current) {
      resizeObserver.observe(chartContainerRef.current)
    }

    return () => {
      resizeObserver.disconnect()
      if (chartInstanceRef.current) {
        chartInstanceRef.current.remove()
        chartInstanceRef.current = null
        candlestickSeriesRef.current = null
        volumeSeriesRef.current = null
      }
    }
  }, [stockData, period])

  // 차트 초기화
  useEffect(() => {
    const cleanup = initChart()
    return () => {
      if (cleanup) cleanup()
    }
  }, [initChart])

  // 장 운영 시간 체크 함수 (09:00 ~ 15:30)
  const isMarketOpen = () => {
    const now = new Date()
    const hours = now.getHours()
    const minutes = now.getMinutes()
    const currentTime = hours * 100 + minutes
    return currentTime >= 900 && currentTime <= 1530
  }

  // 주기적 데이터 갱신
  useEffect(() => {
    fetchStockDetail(true)

    if (isMarketOpen()) {
      autoRefreshIntervalRef.current = setInterval(() => {
        fetchStockDetail(false)
      }, 3000)
    }

    return () => {
      if (autoRefreshIntervalRef.current) {
        clearInterval(autoRefreshIntervalRef.current)
      }
    }
  }, [symbol, period])

  // 뉴스 및 댓글 로드
  useEffect(() => {
    if (symbol) {
      fetchComments(1)
      fetchNews(1)
    }
  }, [symbol])

  // 현재가를 주문 가격에 동기화
  useEffect(() => {
    if (stockData?.basic_info?.current_price) {
      setOrderForm(prev => ({ ...prev, price: String(stockData.basic_info.current_price) }))
    }
  }, [stockData?.basic_info?.current_price])

  if (loading && !stockData) {
    return (
      <div className="stock-detail">
        <div className="sd-loading">
          <div className="sd-loading-spinner" />
          <span>데이터를 불러오는 중...</span>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="stock-detail">
        <div className="sd-error">{error}</div>
      </div>
    )
  }

  if (!stockData) {
    return (
      <div className="stock-detail">
        <div className="sd-error">데이터를 찾을 수 없습니다</div>
      </div>
    )
  }

  const { basic_info, chart_data, prediction } = stockData

  // 예측 기반 등락 계산
  const predictionChange = prediction
    ? ((prediction.predicted_price - basic_info.current_price) / basic_info.current_price * 100)
    : null

  return (
    <div className="stock-detail">
      {/* 상단 종목 헤더 */}
      <section className="sd-hero">
        <div className="sd-hero-top">
          <div className="sd-name-area">
            <h1 className="sd-stock-name">{basic_info.stock_name}</h1>
            <div className="sd-meta">
              <span className="sd-stock-code">{basic_info.stock_code}</span>
              <span className="sd-market-badge">{basic_info.market}</span>
            </div>
          </div>
        </div>
        <div className="sd-price-area">
          <span className={`sd-current-price ${getPriceChangeClass(basic_info.change_price)}`}>
            {formatNumber(basic_info.current_price)}
            <span className="sd-price-unit">원</span>
          </span>
          <div className={`sd-change-row ${getPriceChangeClass(basic_info.change_price)}`}>
            <span className="sd-change-icon">{getChangeIcon(basic_info.change_price)}</span>
            <span className="sd-change-price">{formatNumber(Math.abs(basic_info.change_price))}원</span>
            <span className="sd-change-rate">
              {basic_info.change_rate > 0 ? '+' : ''}{basic_info.change_rate.toFixed(2)}%
            </span>
          </div>
        </div>
      </section>

      {/* 핵심 투자 지표 */}
      <section className="sd-metrics">
        <div className="sd-metric-item">
          <span className="sd-metric-label">시가총액</span>
          <span className="sd-metric-value">{formatMarketCap(basic_info.hts_avls)}</span>
        </div>
        <div className="sd-metric-divider" />
        <div className="sd-metric-item">
          <span className="sd-metric-label">거래량</span>
          <span className="sd-metric-value">{formatVolume(basic_info.volume)}</span>
        </div>
        <div className="sd-metric-divider" />
        <div className="sd-metric-item">
          <span className="sd-metric-label">PER</span>
          <span className="sd-metric-value">{basic_info.per ? basic_info.per.toFixed(2) : '-'}</span>
        </div>
        <div className="sd-metric-divider" />
        <div className="sd-metric-item">
          <span className="sd-metric-label">PBR</span>
          <span className="sd-metric-value">{basic_info.pbr ? basic_info.pbr.toFixed(2) : '-'}</span>
        </div>
      </section>

      {/* 차트 영역 */}
      <section className="sd-card">
        <div className="sd-card-header">
          <h2>차트</h2>
          <div className="sd-period-selector">
            {[
              { key: 'D', label: '일' },
              { key: 'W', label: '주' },
              { key: 'M', label: '월' },
            ].map(p => (
              <button
                key={p.key}
                className={period === p.key ? 'active' : ''}
                onClick={() => setPeriod(p.key)}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>
        <div className="sd-chart-container">
          {chart_data && chart_data.length > 0 ? (
            <div ref={chartContainerRef} style={{ position: 'relative', width: '100%', height: '400px' }} />
          ) : (
            <p className="sd-empty">차트 데이터가 없습니다</p>
          )}
        </div>
      </section>

      {/* 투자 정보 카드 */}
      <section className="sd-card">
        <h2>투자 정보</h2>
        <div className="sd-info-table">
          <div className="sd-info-row">
            <span className="sd-info-key">현재가</span>
            <span className={`sd-info-val ${getPriceChangeClass(basic_info.change_price)}`}>
              {formatPrice(basic_info.current_price)}
            </span>
          </div>
          <div className="sd-info-row">
            <span className="sd-info-key">전일대비</span>
            <span className={`sd-info-val ${getPriceChangeClass(basic_info.change_price)}`}>
              {basic_info.change_price > 0 ? '+' : ''}{formatPrice(basic_info.change_price)}
              <span className="sd-info-sub">
                ({basic_info.change_rate > 0 ? '+' : ''}{basic_info.change_rate.toFixed(2)}%)
              </span>
            </span>
          </div>
          <div className="sd-info-row">
            <span className="sd-info-key">거래량</span>
            <span className="sd-info-val">{formatNumber(basic_info.volume)}주</span>
          </div>
          {basic_info.hts_avls && (
            <div className="sd-info-row">
              <span className="sd-info-key">HTS 시가총액</span>
              <span className="sd-info-val">{formatMarketCap(basic_info.hts_avls)}</span>
            </div>
          )}
          {basic_info.per && (
            <div className="sd-info-row">
              <span className="sd-info-key">PER</span>
              <span className="sd-info-val">{basic_info.per.toFixed(2)}배</span>
            </div>
          )}
          {basic_info.pbr && (
            <div className="sd-info-row">
              <span className="sd-info-key">PBR</span>
              <span className="sd-info-val">{basic_info.pbr.toFixed(2)}배</span>
            </div>
          )}
        </div>
      </section>

      {/* AI 예측 */}
      {prediction && (
        <section className="sd-card">
          <h2>AI 예측</h2>
          <div className="sd-prediction">
            <div className="sd-pred-hero">
              <div className="sd-pred-price-block">
                <span className="sd-pred-label">예측가</span>
                <span className="sd-pred-price">{formatPrice(Math.round(prediction.predicted_price))}</span>
                {predictionChange !== null && (
                  <span className={`sd-pred-diff ${predictionChange >= 0 ? 'price-up' : 'price-down'}`}>
                    {predictionChange >= 0 ? '+' : ''}{predictionChange.toFixed(2)}%
                  </span>
                )}
              </div>
              <div className="sd-pred-date">
                <span className="sd-pred-label">예측일</span>
                <span>{prediction.prediction_date}</span>
              </div>
            </div>
            <div className="sd-pred-badges">
              <div className="sd-pred-badge-item">
                <span className="sd-pred-badge-label">추세</span>
                <span className={`sd-badge trend-${prediction.trend.includes('상승') ? 'up' : prediction.trend.includes('하락') ? 'down' : 'neutral'}`}>
                  {prediction.trend}
                </span>
              </div>
              <div className="sd-pred-badge-item">
                <span className="sd-pred-badge-label">투자의견</span>
                <span className={`sd-badge recommend-${prediction.recommendation}`}>
                  {prediction.recommendation}
                </span>
              </div>
              <div className="sd-pred-badge-item">
                <span className="sd-pred-badge-label">신뢰도</span>
                <div className="sd-confidence">
                  <div className="sd-confidence-bar">
                    <div
                      className="sd-confidence-fill"
                      style={{ width: `${(prediction.confidence * 100).toFixed(0)}%` }}
                    />
                  </div>
                  <span className="sd-confidence-text">{(prediction.confidence * 100).toFixed(0)}%</span>
                </div>
              </div>
            </div>
            <p className="sd-pred-disclaimer">
              * 이 예측은 단순 이동평균 기반 분석이며, 투자 권유가 아닙니다.
            </p>
          </div>
        </section>
      )}

      {/* 주문 */}
      <section className="sd-card">
        <h2>주문</h2>
        {isLoggedIn ? (
          <div className="sd-trade">
            <div className="sd-trade-tabs">
              <button
                className={`sd-trade-tab buy ${tradeTab === 'buy' ? 'active' : ''}`}
                onClick={() => setTradeTab('buy')}
              >
                매수
              </button>
              <button
                className={`sd-trade-tab sell ${tradeTab === 'sell' ? 'active' : ''}`}
                onClick={() => setTradeTab('sell')}
              >
                매도
              </button>
            </div>
            <form className="sd-trade-form" onSubmit={handleOrder}>
              <div className="sd-trade-group">
                <label>주문유형</label>
                <select
                  value={orderForm.orderType}
                  onChange={(e) => setOrderForm({ ...orderForm, orderType: e.target.value })}
                >
                  <option value="00">지정가</option>
                  <option value="01">시장가</option>
                </select>
              </div>
              <div className="sd-trade-group">
                <label>가격</label>
                <input
                  type="number"
                  value={orderForm.orderType === '01' ? '' : orderForm.price}
                  onChange={(e) => setOrderForm({ ...orderForm, price: e.target.value })}
                  placeholder={orderForm.orderType === '01' ? '시장가' : '주문 가격'}
                  disabled={orderForm.orderType === '01'}
                  min="0"
                />
              </div>
              <div className="sd-trade-group">
                <label>수량</label>
                <input
                  type="number"
                  value={orderForm.quantity}
                  onChange={(e) => setOrderForm({ ...orderForm, quantity: e.target.value })}
                  placeholder="주문 수량"
                  min="1"
                  required
                />
              </div>
              {orderForm.quantity && orderForm.price && orderForm.orderType !== '01' && (
                <div className="sd-trade-total">
                  <span className="sd-trade-total-label">총 주문금액</span>
                  <span className="sd-trade-total-value">
                    {formatNumber(parseInt(orderForm.quantity) * parseInt(orderForm.price))}원
                  </span>
                </div>
              )}
              <button
                type="submit"
                className={`sd-trade-btn ${tradeTab}`}
                disabled={
                  !orderForm.quantity ||
                  (orderForm.orderType !== '01' && !orderForm.price) ||
                  buyMutation.isPending ||
                  sellMutation.isPending
                }
              >
                {(buyMutation.isPending || sellMutation.isPending)
                  ? '주문 중...'
                  : tradeTab === 'buy' ? '매수' : '매도'
                }
              </button>
            </form>
          </div>
        ) : (
          <div className="sd-login-prompt">
            로그인하면 주문할 수 있습니다.
          </div>
        )}
      </section>

      {/* 뉴스 */}
      <section className="sd-card">
        <h2>관련 뉴스 <span className="sd-count">{totalNews}</span></h2>

        <div className="sd-news-list">
          {news.length === 0 && !newsLoading ? (
            <p className="sd-empty">최근 뉴스가 없습니다.</p>
          ) : (
            news.map((item) => (
              <a
                key={item.id}
                href={item.url}
                target="_blank"
                rel="noopener noreferrer"
                className="sd-news-item"
              >
                <div className="sd-news-body">
                  <span className="sd-news-title">{item.title}</span>
                  <div className="sd-news-meta">
                    {item.source && <span className="sd-news-source">{item.source}</span>}
                    <span className="sd-news-time">{formatNewsTime(item.published_at)}</span>
                  </div>
                </div>
                {item.image_url && (
                  <div className="sd-news-thumb">
                    <img src={item.image_url} alt="" loading="lazy" />
                  </div>
                )}
              </a>
            ))
          )}

          {newsLoading && <p className="sd-loading-text">뉴스를 불러오는 중...</p>}

          {hasMoreNews && !newsLoading && (
            <button className="sd-more-btn" onClick={() => fetchNews(newsPage + 1)}>
              더 보기
            </button>
          )}
        </div>
      </section>

      {/* 커뮤니티 (댓글) */}
      <section className="sd-card">
        <h2>커뮤니티 <span className="sd-count">{totalComments}</span></h2>

        {/* 댓글 작성 폼 */}
        {isLoggedIn ? (
          <form className="sd-comment-form" onSubmit={handleSubmitComment}>
            <textarea
              className="sd-comment-input"
              placeholder="이 종목에 대한 의견을 남겨주세요..."
              value={newComment}
              onChange={(e) => setNewComment(e.target.value)}
              maxLength={1000}
              rows={3}
            />
            <div className="sd-comment-form-footer">
              <span className="sd-char-count">{newComment.length}/1000</span>
              <button
                type="submit"
                className="sd-submit-btn"
                disabled={!newComment.trim() || commentSubmitting}
              >
                {commentSubmitting ? '작성 중...' : '작성'}
              </button>
            </div>
          </form>
        ) : (
          <div className="sd-login-prompt">
            로그인하면 댓글을 작성할 수 있습니다.
          </div>
        )}

        {/* 댓글 목록 */}
        <div className="sd-comments">
          {comments.length === 0 && !commentLoading ? (
            <p className="sd-empty">아직 댓글이 없습니다. 첫 번째 댓글을 남겨보세요!</p>
          ) : (
            comments.map((comment) => (
              <div key={comment.id} className="sd-comment">
                <div className="sd-comment-head">
                  <span className="sd-comment-author">{comment.username}</span>
                  <span className="sd-comment-time">{formatCommentTime(comment.created_at)}</span>
                  {comment.updated_at && <span className="sd-comment-edited">(수정됨)</span>}
                </div>

                {editingCommentId === comment.id ? (
                  <div className="sd-comment-edit">
                    <textarea
                      className="sd-comment-input"
                      value={editingContent}
                      onChange={(e) => setEditingContent(e.target.value)}
                      maxLength={1000}
                      rows={3}
                    />
                    <div className="sd-comment-edit-actions">
                      <button
                        className="sd-btn-ghost"
                        onClick={() => {
                          setEditingCommentId(null)
                          setEditingContent('')
                        }}
                      >
                        취소
                      </button>
                      <button
                        className="sd-btn-primary"
                        onClick={() => handleUpdateComment(comment.id)}
                        disabled={!editingContent.trim() || commentSubmitting}
                      >
                        저장
                      </button>
                    </div>
                  </div>
                ) : (
                  <>
                    <p className="sd-comment-body">{comment.content}</p>
                    {comment.is_mine && (
                      <div className="sd-comment-actions">
                        <button
                          className="sd-btn-text edit"
                          onClick={() => {
                            setEditingCommentId(comment.id)
                            setEditingContent(comment.content)
                          }}
                        >
                          수정
                        </button>
                        <button
                          className="sd-btn-text delete"
                          onClick={() => handleDeleteComment(comment.id)}
                        >
                          삭제
                        </button>
                      </div>
                    )}
                  </>
                )}
              </div>
            ))
          )}

          {commentLoading && <p className="sd-loading-text">댓글을 불러오는 중...</p>}

          {hasMoreComments && !commentLoading && (
            <button className="sd-more-btn" onClick={() => fetchComments(commentPage + 1)}>
              더 보기
            </button>
          )}
        </div>
      </section>
    </div>
  )
}

export default StockDetail
