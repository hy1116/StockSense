import { useParams } from 'react-router-dom'
import { useState, useEffect, useRef, useCallback } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useAuth } from '../contexts/AuthContext'
import { getStockDetail, getStockChart, getStockPrediction, getComments, createComment, updateComment, deleteComment, getStockNews, buyStock, sellStock, getPredictionAccuracy, checkWatchlist, addToWatchlist, removeFromWatchlist } from '../services/api'
import { createChart } from 'lightweight-charts'
import './StockDetail.css'

function StockDetail() {
  const { symbol } = useParams()
  const { isLoggedIn, user } = useAuth()
  const queryClient = useQueryClient()
  const [stockData, setStockData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [period, setPeriod] = useState('1D')
  const [chartCandles, setChartCandles] = useState([])

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

  // 적중률 관련 state
  const [accuracy, setAccuracy] = useState(null)
  const [accuracyLoading, setAccuracyLoading] = useState(false)

  // AI 예측 상태 (별도 로딩)
  const [prediction, setPrediction] = useState(null)
  const [predictionLoading, setPredictionLoading] = useState(true)

  // AI 예측 상세 드롭다운
  const [showPredDetails, setShowPredDetails] = useState(true)

  // 관심종목 관련 state
  const [isWatchlisted, setIsWatchlisted] = useState(false)
  const [watchlistLoading, setWatchlistLoading] = useState(false)

  // 주문 관련 state
  const [tradeTab, setTradeTab] = useState('buy')
  const [orderForm, setOrderForm] = useState({
    quantity: '1',
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
      const serverMessage = error.response?.data?.detail || error.message;
      alert(`${serverMessage}`);
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
      const serverMessage = error.response?.data?.detail || error.message;
      alert(`${serverMessage}`);
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
      const data = await getStockDetail(symbol)
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

  // AI 예측 조회 (캐시 없음 - 항상 최신)
  const fetchPrediction = async () => {
    setPredictionLoading(true)
    try {
      const data = await getStockPrediction(symbol)
      setPrediction(data)
    } catch (err) {
      console.error('Error fetching prediction:', err)
    } finally {
      setPredictionLoading(false)
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

  // 적중률 조회
  const fetchAccuracy = async () => {
    setAccuracyLoading(true)
    try {
      const data = await getPredictionAccuracy(symbol, 30)
      setAccuracy(data)
    } catch (err) {
      console.error('Error fetching accuracy:', err)
    } finally {
      setAccuracyLoading(false)
    }
  }

  // 관심종목 상태 확인
  const fetchWatchlistStatus = async () => {
    if (!isLoggedIn) return
    try {
      const data = await checkWatchlist(symbol)
      setIsWatchlisted(data.is_watchlisted)
    } catch (err) {
      console.error('Error checking watchlist:', err)
    }
  }

  // 관심종목 토글
  const handleToggleWatchlist = async () => {
    if (!isLoggedIn || watchlistLoading) return
    setWatchlistLoading(true)
    try {
      if (isWatchlisted) {
        await removeFromWatchlist(symbol)
        setIsWatchlisted(false)
      } else {
        const { basic_info } = stockData
        await addToWatchlist(symbol, basic_info.stock_name, basic_info.market)
        setIsWatchlisted(true)
      }
      // 홈 관심종목 탭 캐시 갱신
      queryClient.invalidateQueries(['watchlist'])
    } catch (err) {
      alert(err.response?.data?.detail || '관심종목 처리에 실패했습니다')
    } finally {
      setWatchlistLoading(false)
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

  // 재무 금액 포맷 (억/조 단위)
  const formatFinancialAmount = (val) => {
    if (val === null || val === undefined) return '-'
    const abs = Math.abs(val)
    if (abs >= 10000) return `${(val / 10000).toFixed(1)}조`
    if (abs >= 1) return `${val.toFixed(0)}억`
    return `${val.toFixed(2)}억`
  }

  // 기술적 지표 툴팁 컴포넌트
  const Tip = ({ text }) => (
    <span className="sd-tip">?<span className="sd-tip-box">{text}</span></span>
  )

  // lightweight-charts 초기화
  const initChart = useCallback(() => {
    if (!chartContainerRef.current || !stockData) return

    if (!chartCandles || chartCandles.length === 0) return

    const isIntraday = period === '1D'

    // 기존 차트 제거
    if (chartInstanceRef.current) {
      chartInstanceRef.current.remove()
      chartInstanceRef.current = null
      candlestickSeriesRef.current = null
      volumeSeriesRef.current = null
    }

    // 기간별 바 간격 계산 (차트 너비에 맞게 캔들을 고르게 배치)
    const containerWidth = chartContainerRef.current.clientWidth
    const visibleBarCount = { '1D': 40, '1W': 10, '1M': 22, '3M': 65, '1Y': 52 }[period] || 65
    const calcBarSpacing = Math.max(4, Math.min(60, (containerWidth - 80) / (visibleBarCount + 2)))

    const chart = createChart(chartContainerRef.current, {
      width: containerWidth,
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
      handleScroll: false,
      handleScale: false,
      timeScale: {
        borderVisible: false,
        timeVisible: isIntraday,
        secondsVisible: false,
        rightOffset: 2,
        barSpacing: calcBarSpacing,
        minBarSpacing: 3,
        fixLeftEdge: true,
        fixRightEdge: true,
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

    let candleResult = []
    let volumeResult = []
    let prevClose = null  // 1D: 이전 캔들 종가 (시가 보정용)

    chartCandles.forEach(d => {
      // 비정상 OHLCV 필터링
      if (!d.open || !d.close || !d.high || !d.low) return

      // 1D 당일 분봉: KIS API stck_oprc는 일봉 시가이므로 이전 캔들 종가로 보정
      // 단, [low, high] 범위를 벗어나지 않도록 클리핑
      const open = (isIntraday && prevClose !== null)
        ? Math.min(Math.max(prevClose, d.low), d.high)
        : d.open

      if (d.high < Math.max(open, d.close) || d.low > Math.min(open, d.close)) return

      let time
      if (isIntraday) {
        // dt = YYYYMMDDHHMM (12자리), KST 시간을 UTC+0 timestamp로 처리 (차트 축 표시용)
        const year  = parseInt(d.dt.slice(0, 4))
        const month = parseInt(d.dt.slice(4, 6)) - 1
        const day   = parseInt(d.dt.slice(6, 8))
        const hh    = parseInt(d.dt.slice(8, 10))
        const mi    = parseInt(d.dt.slice(10, 12))
        time = Math.floor(Date.UTC(year, month, day, hh, mi, 0) / 1000)
      } else {
        // dt = YYYYMMDD (8자리)
        time = `${d.dt.slice(0, 4)}-${d.dt.slice(4, 6)}-${d.dt.slice(6, 8)}`
      }
      prevClose = d.close
      const up = d.close >= open
      candleResult.push({ time, open, high: d.high, low: d.low, close: d.close })
      volumeResult.push({ time, value: d.volume, color: up ? 'rgba(239,68,68,0.35)' : 'rgba(59,130,246,0.35)' })
    })

    candleSeries.setData(candleResult)
    volumeSeries.setData(volumeResult)

    // 스크롤 비활성화 상태이므로 항상 전체 데이터를 표시
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
  }, [stockData, period, chartCandles])

  // 차트 초기화
  useEffect(() => {
    const cleanup = initChart()
    return () => {
      if (cleanup) cleanup()
    }
  }, [initChart])

  // 장 운영 시간 체크 함수 (KST 09:00 ~ 15:30)
  const isMarketOpen = () => {
    const now = new Date()
    const kstMinutes = (now.getUTCHours() * 60 + now.getUTCMinutes() + 9 * 60) % (24 * 60)
    const kstTime = Math.floor(kstMinutes / 60) * 100 + (kstMinutes % 60)
    return kstTime >= 900 && kstTime <= 1530
  }

  // 차트 데이터 조회 (네이버 금융)
  const fetchChartData = useCallback(async (periodOverride) => {
    const p = periodOverride || period
    try {
      const data = await getStockChart(symbol, p)
      setChartCandles(data)
    } catch (err) {
      console.error('차트 데이터 조회 실패:', err)
      setChartCandles([])
    }
  }, [symbol, period])

  // 주기적 데이터 갱신 (30초 간격, 1D만)
  useEffect(() => {
    fetchStockDetail(true)
    fetchChartData()
    fetchPrediction()

    if (isMarketOpen() && period === '1D') {
      autoRefreshIntervalRef.current = setInterval(() => {
        fetchStockDetail(false)
        fetchChartData()
      }, 30000)
    }

    return () => clearInterval(autoRefreshIntervalRef.current)
  }, [symbol, period])

  // 뉴스, 댓글, 적중률, 관심종목 로드
  useEffect(() => {
    if (symbol) {
      fetchComments(1)
      fetchNews(1)
      fetchAccuracy()
      fetchWatchlistStatus()
    }
  }, [symbol, isLoggedIn])

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

  const { basic_info, chart_data } = stockData

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
          {isLoggedIn && (
            <button
              className={`sd-watchlist-btn ${isWatchlisted ? 'active' : ''}`}
              onClick={handleToggleWatchlist}
              disabled={watchlistLoading}
              title={isWatchlisted ? '관심종목 해제' : '관심종목 등록'}
            >
              {isWatchlisted ? '\u2605' : '\u2606'}
            </button>
          )}
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
              { key: '1D', label: '1일' },
              { key: '1M', label: '1개월' },
              { key: '3M', label: '3개월' },
              { key: '1Y', label: '1년' },
            ].map(p => (
              <button
                key={p.key}
                className={period === p.key ? 'active' : ''}
                onClick={() => {
                  setPeriod(p.key)
                  fetchChartData(p.key)
                }}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>
        <div className="sd-chart-container">
          {chartCandles && chartCandles.length > 0 ? (
            <div ref={chartContainerRef} style={{ position: 'relative', width: '100%', height: '400px' }} />
          ) : (
            <p className="sd-empty">차트 데이터가 없습니다</p>
          )}
        </div>
      </section>

      {/* AI 예측 */}
      <section className="sd-card">
        <h2>AI 예측</h2>
        {predictionLoading ? (
          <div className="sd-pred-skeleton">
            <div className="sd-pred-skeleton-hero">
              <div className="sd-skeleton-block" style={{ width: '38%', height: '2.5rem' }} />
              <div className="sd-skeleton-block" style={{ width: '22%', height: '2rem' }} />
            </div>
            <div className="sd-skeleton-block" style={{ width: '100%', height: '1.4rem', marginTop: '1rem' }} />
            <div className="sd-skeleton-block" style={{ width: '100%', height: '3.5rem', marginTop: '0.75rem' }} />
            <div className="sd-skeleton-block" style={{ width: '60%', height: '1rem', marginTop: '0.75rem' }} />
          </div>
        ) : prediction ? (
        <div className="sd-prediction">
            <div className="sd-pred-hero">
              <div className="sd-pred-price-block">
                <span className="sd-pred-label">단기 예측가 <small>(1거래일)</small></span>
                <span className={`sd-pred-price${predictionChange !== null ? (predictionChange >= 0 ? ' price-up' : ' price-down') : ''}`}>{formatPrice(Math.round(prediction.predicted_price))}</span>
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
            {prediction.predicted_price_long != null && (() => {
              const longChange = (prediction.predicted_price_long - basic_info.current_price) / basic_info.current_price * 100
              return (
                <div className="sd-pred-hero sd-pred-hero-long">
                  <div className="sd-pred-price-block">
                    <span className="sd-pred-label">장기 예측가 <small>(20거래일 · 약 1개월)</small></span>
                    <span className={`sd-pred-price${longChange >= 0 ? ' price-up' : ' price-down'}`}>
                      {formatPrice(Math.round(prediction.predicted_price_long))}
                    </span>
                    <span className={`sd-pred-diff ${longChange >= 0 ? 'price-up' : 'price-down'}`}>
                      {longChange >= 0 ? '+' : ''}{longChange.toFixed(2)}%
                    </span>
                  </div>
                  <div className="sd-pred-date">
                    <span className="sd-pred-label">목표일</span>
                    <span>{prediction.prediction_date_long}</span>
                  </div>
                </div>
              )
            })()}
            {/* ML 한줄 의견 */}
            {prediction.details && prediction.details.model_used !== 'rule_based' && (() => {
              const rf = prediction.details.recommendation_factors
              const ti = prediction.details.technical_indicators
              const modelLabel = prediction.details.model_used === 'ensemble' ? 'XGBoost+LSTM 앙상블' : prediction.details.model_used === 'xgboost' ? 'XGBoost' : 'LSTM'
              const trendText = prediction.trend
              const rec = prediction.recommendation
              const conf = Math.round(prediction.confidence * 100)
              let reason = ''
              if (rf) {
                const scores = [
                  { label: '추세', v: rf.trend_score },
                  { label: 'RSI', v: rf.rsi_score },
                  { label: 'MACD', v: rf.macd_score },
                  { label: '볼린저밴드', v: rf.bb_score },
                ]
                const topPos = scores.filter(s => s.v > 0).sort((a, b) => b.v - a.v)[0]
                const topNeg = scores.filter(s => s.v < 0).sort((a, b) => a.v - b.v)[0]
                if (rec === '적극 매수' || rec === '매수') {
                  reason = topPos ? `${topPos.label} 지표가 긍정적이며` : '기술적 지표 종합 결과'
                } else if (rec === '적극 매도' || rec === '매도') {
                  reason = topNeg ? `${topNeg.label} 지표가 부정적이며` : '기술적 지표 종합 결과'
                } else {
                  reason = '기술적 지표가 혼재하여'
                }
              }
              const opinion = `${modelLabel} 모델 분석 결과, ${reason} ${trendText} 추세로 "${rec}" 의견입니다. (신뢰도 ${conf}%)`
              return (
                <div className="sd-ml-opinion">
                  <span className="sd-ml-opinion-icon">🤖</span>
                  <span className="sd-ml-opinion-text">{opinion}</span>
                </div>
              )
            })()}
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
            {/* 적중률 */}
            <div className="sd-accuracy-section">
              {accuracyLoading ? (
                <p className="sd-loading-text">적중률을 불러오는 중...</p>
              ) : accuracy && accuracy.evaluated_count > 0 ? (
                <>
                  <div className="sd-accuracy-stats">
                    <div className="sd-accuracy-stat">
                      <span className="sd-accuracy-label">방향 적중률</span>
                      <div className="sd-confidence">
                        <div className="sd-confidence-bar">
                          <div
                            className="sd-confidence-fill"
                            style={{ width: `${accuracy.direction_accuracy ?? 0}%` }}
                          />
                        </div>
                        <span className="sd-confidence-text">
                          {accuracy.direction_accuracy !== null ? `${accuracy.direction_accuracy}%` : '-'}
                        </span>
                      </div>
                    </div>
                    <div className="sd-accuracy-stat">
                      <span className="sd-accuracy-label">평균 오차율</span>
                      <span className="sd-accuracy-value">
                        {accuracy.avg_error_rate !== null ? `${accuracy.avg_error_rate}%` : '-'}
                      </span>
                    </div>
                    <div className="sd-accuracy-stat">
                      <span className="sd-accuracy-label">평가 건수</span>
                      <span className="sd-accuracy-value">
                        {accuracy.evaluated_count} / {accuracy.total_predictions}
                      </span>
                    </div>
                  </div>
                  {accuracy.recent_predictions && accuracy.recent_predictions.length > 0 && (
                    <div className="sd-accuracy-history">
                      <span className="sd-accuracy-history-title">최근 예측 기록</span>
                      <div className="sd-accuracy-list">
                        {accuracy.recent_predictions.map((item, idx) => (
                          <div key={idx} className="sd-accuracy-row">
                            <span className="sd-accuracy-date">{item.prediction_date}</span>
                            <span className="sd-accuracy-pred">
                              {formatNumber(Math.round(item.predicted_price))}
                            </span>
                            <span className="sd-accuracy-actual">
                              {item.actual_price ? formatNumber(Math.round(item.actual_price)) : '-'}
                            </span>
                            <span className={`sd-accuracy-icon ${item.direction_correct === true ? 'correct' : item.direction_correct === false ? 'wrong' : ''}`}>
                              {item.direction_correct === true ? 'O' : item.direction_correct === false ? 'X' : '-'}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <p className="sd-accuracy-empty">
                  예측 기록이 쌓이면 적중률이 표시됩니다
                </p>
              )}
            </div>
            {/* 자세히 보기 */}
            {prediction.details && (
              <>
                <button
                  className={`sd-pred-details-btn ${showPredDetails ? 'active' : ''}`}
                  onClick={() => setShowPredDetails(v => !v)}
                >
                  {showPredDetails ? '간략히 보기 ▲' : '자세히 보기 ▼'}
                </button>

                {showPredDetails && (
                  <div className="sd-pred-details">

                    {/* 사용 모델 */}
                    <div className="sd-detail-section">
                      <h4 className="sd-detail-title">사용 모델</h4>
                      <div className="sd-detail-content">
                        <div className="sd-detail-model-info">
                          <span className="sd-detail-label">분석 방식</span>
                          <span className="sd-detail-value">
                            {prediction.details.model_used === 'ensemble' ? 'XGBoost + LSTM 앙상블' :
                             prediction.details.model_used === 'xgboost' ? 'XGBoost 단독' :
                             prediction.details.model_used === 'lstm' ? 'LSTM 단독' : '규칙 기반 (Fallback)'}
                          </span>
                        </div>
                        {prediction.details.xgb_predicted != null && (() => {
                          const w = prediction.details.ensemble_weights
                          const xgbPct = w ? Math.round(w.xgb_weight * 100) : 60
                          return (
                            <div className="sd-detail-model-row">
                              <span className="sd-detail-model-name">XGBoost ({xgbPct}%)</span>
                              <span className="sd-detail-model-price">{formatPrice(Math.round(prediction.details.xgb_predicted))}</span>
                              <span className={`sd-detail-model-diff ${prediction.details.xgb_predicted >= prediction.current_price ? 'price-up' : 'price-down'}`}>
                                {((prediction.details.xgb_predicted - prediction.current_price) / prediction.current_price * 100).toFixed(2)}%
                              </span>
                            </div>
                          )
                        })()}
                        {prediction.details.lstm_predicted != null && (() => {
                          const w = prediction.details.ensemble_weights
                          const lstmPct = w ? Math.round(w.lstm_weight * 100) : 40
                          return (
                            <div className="sd-detail-model-row">
                              <span className="sd-detail-model-name">LSTM ({lstmPct}%)</span>
                              <span className="sd-detail-model-price">{formatPrice(Math.round(prediction.details.lstm_predicted))}</span>
                              <span className={`sd-detail-model-diff ${prediction.details.lstm_predicted >= prediction.current_price ? 'price-up' : 'price-down'}`}>
                                {((prediction.details.lstm_predicted - prediction.current_price) / prediction.current_price * 100).toFixed(2)}%
                              </span>
                            </div>
                          )
                        })()}
                      </div>
                    </div>

                    {/* 기술적 지표 */}
                    {prediction.details.technical_indicators && (() => {
                      const ti = prediction.details.technical_indicators
                      return (
                        <div className="sd-detail-section">
                          <h4 className="sd-detail-title">기술적 지표</h4>
                          <div className="sd-detail-content">

                            {/* 이동평균 */}
                            <div className="sd-detail-group">
                              <span className="sd-detail-group-label">이동평균 (이격도) <Tip text="5일·10일·20일 이동평균선과 현재가의 이격률입니다. 현재가가 이동평균 위에 있으면 상승 추세, 아래이면 하락 추세를 나타냅니다." /></span>
                              <div className="sd-detail-ind-list">
                                {[['MA5', ti.ma5], ['MA10', ti.ma10], ['MA20', ti.ma20]].map(([label, val]) => {
                                  const diff = ((prediction.current_price - val) / val * 100).toFixed(2)
                                  const isAbove = prediction.current_price >= val
                                  return (
                                    <div key={label} className="sd-detail-ind-row">
                                      <span className="sd-detail-ind-name">{label}</span>
                                      <span className="sd-detail-ind-val">{formatNumber(Math.round(val))}</span>
                                      <span className={`sd-detail-ind-signal ${isAbove ? 'signal-up' : 'signal-down'}`}>
                                        {isAbove ? '▲' : '▼'} {Math.abs(diff)}%
                                      </span>
                                    </div>
                                  )
                                })}
                              </div>
                            </div>

                            {/* RSI */}
                            <div className="sd-detail-group">
                              <span className="sd-detail-group-label">RSI (14일) <Tip text="상대강도지수(RSI)는 0~100 사이 값입니다. 70 이상이면 과매수(매도 신호), 30 이하면 과매도(매수 신호)로 해석하며, 추세 반전 가능성을 나타냅니다." /></span>
                              <div className="sd-detail-rsi-wrap">
                                <div className="sd-detail-rsi-track">
                                  <div className="sd-detail-rsi-zone zone-sell" />
                                  <div className="sd-detail-rsi-zone zone-neutral" />
                                  <div className="sd-detail-rsi-zone zone-buy" />
                                  <div className="sd-detail-rsi-needle" style={{ left: `${ti.rsi}%` }} />
                                </div>
                                <div className="sd-detail-rsi-labels">
                                  <span>0</span><span>과매도30</span><span>50</span><span>과매수70</span><span>100</span>
                                </div>
                                <div className="sd-detail-rsi-result">
                                  <span className="sd-detail-rsi-num">{ti.rsi.toFixed(1)}</span>
                                  <span className={`sd-detail-ind-signal ${ti.rsi > 70 ? 'signal-down' : ti.rsi < 30 ? 'signal-up' : 'signal-neutral'}`}>
                                    {ti.rsi > 70 ? '과매수 (매도 신호)' : ti.rsi < 30 ? '과매도 (매수 신호)' : '중립'}
                                  </span>
                                </div>
                              </div>
                            </div>

                            {/* 볼린저밴드 */}
                            <div className="sd-detail-group">
                              <span className="sd-detail-group-label">볼린저밴드 (20일, ±2σ) <Tip text="20일 이동평균선 기준 ±2 표준편차 밴드입니다. 현재가가 상단 근접 시 과열 구간(매도 신호), 하단 근접 시 저평가 구간(매수 신호)으로 해석합니다." /></span>
                              <div className="sd-detail-ind-list">
                                {[['상단', ti.bb_upper, 'signal-down'], ['중간(기준)', ti.bb_middle, 'signal-neutral'], ['하단', ti.bb_lower, 'signal-up']].map(([label, val, cls]) => (
                                  <div key={label} className="sd-detail-ind-row">
                                    <span className="sd-detail-ind-name">{label}</span>
                                    <span className="sd-detail-ind-val">{formatNumber(Math.round(val))}</span>
                                  </div>
                                ))}
                              </div>
                              {ti.bb_position != null && (
                                <div className="sd-detail-bb-pos-wrap">
                                  <div className="sd-detail-bb-track">
                                    <div className="sd-detail-bb-needle" style={{ left: `${(ti.bb_position * 100).toFixed(1)}%` }} />
                                  </div>
                                  <span className={`sd-detail-ind-signal ${ti.bb_position > 0.8 ? 'signal-down' : ti.bb_position < 0.2 ? 'signal-up' : 'signal-neutral'}`}>
                                    밴드 내 위치 {(ti.bb_position * 100).toFixed(0)}%
                                    {ti.bb_position > 0.8 ? ' · 상단 근접 (매도 신호)' : ti.bb_position < 0.2 ? ' · 하단 근접 (매수 신호)' : ' · 중간권'}
                                  </span>
                                </div>
                              )}
                            </div>

                            {/* MACD */}
                            <div className="sd-detail-group">
                              <span className="sd-detail-group-label">MACD (12-26-9) <Tip text="단기(12일)와 장기(26일) 이동평균의 차이입니다. 히스토그램이 양수(+)이면 상승 모멘텀, 음수(-)이면 하락 모멘텀을 나타냅니다. 시그널선(9일 평균) 돌파 시 추세 전환 신호입니다." /></span>
                              <div className="sd-detail-ind-list">
                                <div className="sd-detail-ind-row">
                                  <span className="sd-detail-ind-name">MACD</span>
                                  <span className="sd-detail-ind-val">{ti.macd.toFixed(1)}</span>
                                </div>
                                <div className="sd-detail-ind-row">
                                  <span className="sd-detail-ind-name">시그널</span>
                                  <span className="sd-detail-ind-val">{ti.macd_signal.toFixed(1)}</span>
                                </div>
                                <div className="sd-detail-ind-row">
                                  <span className="sd-detail-ind-name">히스토그램</span>
                                  <span className={`sd-detail-ind-val ${ti.macd_diff > 0 ? 'price-up' : 'price-down'}`}>
                                    {ti.macd_diff > 0 ? '+' : ''}{ti.macd_diff.toFixed(1)}
                                  </span>
                                  <span className={`sd-detail-ind-signal ${ti.macd_diff > 0 ? 'signal-up' : 'signal-down'}`}>
                                    {ti.macd_diff > 0 ? '상승 모멘텀' : '하락 모멘텀'}
                                  </span>
                                </div>
                              </div>
                            </div>

                            {/* 거래량 지표 */}
                            {(ti.volume_ratio != null || ti.mfi != null) && (
                              <div className="sd-detail-group">
                                <span className="sd-detail-group-label">거래량 지표 <Tip text="거래량 비율(Volume Ratio)은 당일 거래량을 20일 평균 거래량으로 나눈 값입니다. MFI(Money Flow Index)는 거래량을 반영한 RSI로 0~100 사이 값이며, 80 이상이면 과매수, 20 이하면 과매도 신호입니다." /></span>
                                <div className="sd-detail-ind-list">
                                  {ti.volume_ratio != null && (
                                    <div className="sd-detail-ind-row">
                                      <span className="sd-detail-ind-name">거래량 비율</span>
                                      <span className="sd-detail-ind-val">{ti.volume_ratio.toFixed(2)}x</span>
                                      <span className={`sd-detail-ind-signal ${ti.volume_ratio > 1.5 ? 'signal-down' : ti.volume_ratio > 1.0 ? 'signal-up' : 'signal-neutral'}`}>
                                        {ti.volume_ratio > 1.5 ? '급증 (불확실성↑)' : ti.volume_ratio > 1.0 ? '평균 초과' : '평균 이하'}
                                      </span>
                                    </div>
                                  )}
                                  {ti.mfi != null && (
                                    <div className="sd-detail-ind-row">
                                      <span className="sd-detail-ind-name">MFI (14일)</span>
                                      <span className="sd-detail-ind-val">{ti.mfi.toFixed(1)}</span>
                                      <span className={`sd-detail-ind-signal ${ti.mfi > 80 ? 'signal-down' : ti.mfi < 20 ? 'signal-up' : 'signal-neutral'}`}>
                                        {ti.mfi > 80 ? '과매수' : ti.mfi < 20 ? '과매도' : '중립'}
                                      </span>
                                    </div>
                                  )}
                                  {ti.obv_normalized != null && (
                                    <div className="sd-detail-ind-row">
                                      <span className="sd-detail-ind-name">OBV (정규화)</span>
                                      <span className={`sd-detail-ind-val ${ti.obv_normalized > 0 ? 'price-up' : 'price-down'}`}>
                                        {ti.obv_normalized > 0 ? '+' : ''}{ti.obv_normalized.toFixed(2)}
                                      </span>
                                      <span className={`sd-detail-ind-signal ${ti.obv_normalized > 0 ? 'signal-up' : 'signal-down'}`}>
                                        {ti.obv_normalized > 0 ? '매수 압력' : '매도 압력'}
                                      </span>
                                    </div>
                                  )}
                                </div>
                              </div>
                            )}

                          </div>
                        </div>
                      )
                    })()}

                    {/* 투자의견 근거 */}
                    {prediction.details.recommendation_factors && (() => {
                      const rf = prediction.details.recommendation_factors
                      const ti = prediction.details.technical_indicators
                      const factors = [
                        { key: 'pred_score', label: '예측 변동률', weight: 30, extra: `${rf.pred_change_pct >= 0 ? '+' : ''}${rf.pred_change_pct?.toFixed(2)}%`, tip: 'XGBoost+LSTM 앙상블이 예측한 내일 종가 변동률 기준 점수입니다. 가장 높은 비중(30%)을 차지합니다.' },
                        { key: 'trend_score', label: '추세 (MA)', weight: 25, extra: prediction.trend, tip: '5일·10일·20일 이동평균과 현재가의 위치로 계산한 추세 점수입니다. 이동평균 위에 있을수록 양수 점수입니다.' },
                        { key: 'rsi_score', label: 'RSI', weight: 20, extra: ti?.rsi?.toFixed(1), tip: 'RSI가 30 이하(과매도)면 양수, 70 이상(과매수)면 음수 점수를 부여합니다.' },
                        { key: 'macd_score', label: 'MACD', weight: 15, extra: ti?.macd_diff >= 0 ? '골든크로스↑' : '데드크로스↓', tip: 'MACD 히스토그램 부호로 모멘텀을 판단합니다. 양수(상승 모멘텀)면 양수 점수입니다.' },
                        { key: 'bb_score', label: '볼린저밴드', weight: 10, extra: ti?.bb_position != null ? `${(ti.bb_position * 100).toFixed(0)}%` : null, tip: '볼린저밴드 내 위치 기준 점수입니다. 하단(저평가) 근접 시 양수, 상단(과열) 근접 시 음수 점수를 부여합니다.' },
                      ]
                      return (
                        <div className="sd-detail-section">
                          <h4 className="sd-detail-title">투자의견 근거</h4>
                          <div className="sd-detail-content">
                            {factors.map(({ key, label, weight, extra, tip }) => {
                              const score = rf[key] ?? 0
                              return (
                                <div key={key} className="sd-detail-factor">
                                  <div className="sd-detail-factor-header">
                                    <span className="sd-detail-factor-label">{label}{tip && <Tip text={tip} />}</span>
                                    <span className="sd-detail-factor-weight">가중 {weight}%</span>
                                    {extra && <span className="sd-detail-factor-extra">{extra}</span>}
                                    <span className={`sd-detail-factor-score ${score > 0 ? 'price-up' : score < 0 ? 'price-down' : ''}`}>
                                      {score > 0 ? '+' : ''}{score.toFixed(2)}
                                    </span>
                                  </div>
                                  <div className="sd-detail-factor-bar">
                                    <div className="sd-detail-factor-center-line" />
                                    <div
                                      className={`sd-detail-factor-fill ${score >= 0 ? 'fill-up' : 'fill-down'}`}
                                      style={{
                                        left: score >= 0 ? '50%' : `${(score + 1) / 2 * 100}%`,
                                        width: `${Math.abs(score) * 50}%`
                                      }}
                                    />
                                  </div>
                                </div>
                              )
                            })}
                            <div className="sd-detail-total">
                              <span>종합 점수</span>
                              <span className={rf.total_score > 0 ? 'price-up' : rf.total_score < 0 ? 'price-down' : ''}>
                                {rf.total_score > 0 ? '+' : ''}{rf.total_score?.toFixed(3)}
                              </span>
                              <span className={`sd-badge recommend-${prediction.recommendation}`}>
                                {prediction.recommendation}
                              </span>
                            </div>
                          </div>
                        </div>
                      )
                    })()}

                    {/* 뉴스 감성 */}
                    {prediction.details.news_sentiment?.count > 0 && (
                      <div className="sd-detail-section">
                        <h4 className="sd-detail-title">뉴스 감성 분석</h4>
                        <div className="sd-detail-content">
                          <div className="sd-detail-news-grid">
                            {[
                              { label: '분석 뉴스', value: `${prediction.details.news_sentiment.count}건`, cls: '' },
                              { label: '감성 점수', value: `${prediction.details.news_sentiment.score > 0 ? '+' : ''}${prediction.details.news_sentiment.score.toFixed(1)}`, cls: prediction.details.news_sentiment.score > 0 ? 'price-up' : prediction.details.news_sentiment.score < 0 ? 'price-down' : '' },
                              { label: '긍정 비율', value: `${(prediction.details.news_sentiment.positive_ratio * 100).toFixed(0)}%`, cls: 'price-up' },
                              { label: '부정 비율', value: `${(prediction.details.news_sentiment.negative_ratio * 100).toFixed(0)}%`, cls: 'price-down' },
                            ].map(({ label, value, cls }) => (
                              <div key={label} className="sd-detail-news-item">
                                <span className="sd-detail-label">{label}</span>
                                <span className={`sd-detail-value ${cls}`}>{value}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    )}

                    {/* 재무 지표 */}
                    {prediction.details.financial_data && (() => {
                      const fd = prediction.details.financial_data
                      const hasData = fd.per || fd.pbr || fd.roe || fd.div_yield || fd.eps || fd.revenue || fd.operating_profit || fd.net_profit
                      if (!hasData) return null
                      return (
                        <div className="sd-detail-section">
                          <h4 className="sd-detail-title">재무 지표</h4>
                          <div className="sd-detail-content">
                            <div className="sd-detail-group">
                              <span className="sd-detail-group-label">밸류에이션</span>
                              <div className="sd-detail-ind-list">
                                {fd.per != null && fd.per !== 0 && (
                                  <div className="sd-detail-ind-row">
                                    <span className="sd-detail-ind-name">PER</span>
                                    <span className="sd-detail-ind-val">{fd.per.toFixed(2)}배</span>
                                    <span className={`sd-detail-hint ${fd.per < 10 ? 'hint-good' : fd.per > 30 ? 'hint-bad' : 'hint-neutral'}`}>
                                      {fd.per < 10 ? '저평가' : fd.per > 30 ? '고평가' : '적정'}
                                    </span>
                                  </div>
                                )}
                                {fd.pbr != null && fd.pbr !== 0 && (
                                  <div className="sd-detail-ind-row">
                                    <span className="sd-detail-ind-name">PBR</span>
                                    <span className="sd-detail-ind-val">{fd.pbr.toFixed(2)}배</span>
                                    <span className={`sd-detail-hint ${fd.pbr < 1 ? 'hint-good' : fd.pbr > 3 ? 'hint-bad' : 'hint-neutral'}`}>
                                      {fd.pbr < 1 ? '자산대비저평가' : fd.pbr > 3 ? '고평가' : '적정'}
                                    </span>
                                  </div>
                                )}
                                {fd.div_yield != null && fd.div_yield !== 0 && (
                                  <div className="sd-detail-ind-row">
                                    <span className="sd-detail-ind-name">배당수익률</span>
                                    <span className="sd-detail-ind-val">{fd.div_yield.toFixed(2)}%</span>
                                    <span className={`sd-detail-hint ${fd.div_yield >= 3 ? 'hint-good' : 'hint-neutral'}`}>
                                      {fd.div_yield >= 3 ? '고배당' : ''}
                                    </span>
                                  </div>
                                )}
                              </div>
                            </div>
                            <div className="sd-detail-group">
                              <span className="sd-detail-group-label">수익성</span>
                              <div className="sd-detail-ind-list">
                                {fd.roe != null && fd.roe !== 0 && (
                                  <div className="sd-detail-ind-row">
                                    <span className="sd-detail-ind-name">ROE</span>
                                    <span className="sd-detail-ind-val">{fd.roe.toFixed(2)}%</span>
                                    <span className={`sd-detail-hint ${fd.roe > 15 ? 'hint-good' : fd.roe < 5 ? 'hint-bad' : 'hint-neutral'}`}>
                                      {fd.roe > 15 ? '우수' : fd.roe < 5 ? '낮음' : '보통'}
                                    </span>
                                  </div>
                                )}
                                {fd.eps != null && fd.eps !== 0 && (
                                  <div className="sd-detail-ind-row">
                                    <span className="sd-detail-ind-name">EPS</span>
                                    <span className={`sd-detail-ind-val ${fd.eps < 0 ? 'price-down' : ''}`}>
                                      {formatNumber(Math.round(fd.eps))}원
                                    </span>
                                  </div>
                                )}
                              </div>
                            </div>
                            {(fd.revenue != null || fd.operating_profit != null || fd.net_profit != null) && (
                              <div className="sd-detail-group">
                                <span className="sd-detail-group-label">실적 (억원)</span>
                                <div className="sd-detail-ind-list">
                                  {fd.revenue != null && (
                                    <div className="sd-detail-ind-row">
                                      <span className="sd-detail-ind-name">매출액</span>
                                      <span className="sd-detail-ind-val">{formatFinancialAmount(fd.revenue)}</span>
                                    </div>
                                  )}
                                  {fd.operating_profit != null && (
                                    <div className="sd-detail-ind-row">
                                      <span className="sd-detail-ind-name">영업이익</span>
                                      <span className={`sd-detail-ind-val ${fd.operating_profit < 0 ? 'price-down' : ''}`}>
                                        {formatFinancialAmount(fd.operating_profit)}
                                      </span>
                                    </div>
                                  )}
                                  {fd.net_profit != null && (
                                    <div className="sd-detail-ind-row">
                                      <span className="sd-detail-ind-name">순이익</span>
                                      <span className={`sd-detail-ind-val ${fd.net_profit < 0 ? 'price-down' : ''}`}>
                                        {formatFinancialAmount(fd.net_profit)}
                                      </span>
                                    </div>
                                  )}
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      )
                    })()}

                  </div>
                )}
              </>
            )}

            <p className="sd-pred-disclaimer">
              * 이 예측은 XGBoost + LSTM 앙상블 모델과 뉴스 감성 분석을 활용한 AI 분석이며, 투자 권유가 아닙니다.
            </p>
          </div>
        ) : null}
      </section>

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
                <div className="sd-order-type-tabs">
                  <button
                    type="button"
                    className={`sd-order-type-tab ${orderForm.orderType === '00' ? 'active' : ''}`}
                    onClick={() => setOrderForm({ ...orderForm, orderType: '00' })}
                  >
                    지정가
                  </button>
                  <button
                    type="button"
                    className={`sd-order-type-tab ${orderForm.orderType === '01' ? 'active' : ''}`}
                    onClick={() => setOrderForm({ ...orderForm, orderType: '01', price: '' })}
                  >
                    시장가
                  </button>
                </div>
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
                <div className="sd-quantity-control">
                  <button
                    type="button"
                    className="sd-qty-btn"
                    onClick={() => setOrderForm({ ...orderForm, quantity: String(Math.max(1, (parseInt(orderForm.quantity) || 1) - 1)) })}
                  >
                    -
                  </button>
                  <input
                    type="number"
                    value={orderForm.quantity}
                    onChange={(e) => setOrderForm({ ...orderForm, quantity: e.target.value })}
                    placeholder="주문 수량"
                    min="1"
                    required
                  />
                  <button
                    type="button"
                    className="sd-qty-btn"
                    onClick={() => setOrderForm({ ...orderForm, quantity: String((parseInt(orderForm.quantity) || 0) + 1) })}
                  >
                    +
                  </button>
                </div>
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
                  {item.summary && item.summary !== item.title && (
                    <p className="sd-news-summary">{item.summary}</p>
                  )}
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
