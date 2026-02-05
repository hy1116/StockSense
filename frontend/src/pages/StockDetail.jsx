import { useParams } from 'react-router-dom'
import { useState, useEffect, useRef, useCallback } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { getStockDetail, getComments, createComment, updateComment, deleteComment, getStockNews } from '../services/api'
import { createChart } from 'lightweight-charts'
import './StockDetail.css'

function StockDetail() {
  const { symbol } = useParams()
  const { isLoggedIn, user } = useAuth()
  const [stockData, setStockData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [period, setPeriod] = useState('D') // 기본값을 일봉으로

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
      // 미래 시간 (예약 기사) - 실제 게시 시간 표시
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
    if (!num) return '0'
    return num.toLocaleString('ko-KR')
  }

  const formatPrice = (price) => {
    return `${formatNumber(price)}원`
  }

  const getPriceChangeClass = (change) => {
    if (change > 0) return 'price-up'
    if (change < 0) return 'price-down'
    return ''
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

    // 일봉/주봉/월봉만 지원
    const isMinute = false

    // 날짜순 정렬 (오래된 것부터)
    const sortedData = [...chartData].reverse()

    // 차트 생성
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 500,
      layout: {
        background: { color: '#1e222d' },
        textColor: '#d1d4dc',
      },
      grid: {
        vertLines: { color: '#2b2b43' },
        horzLines: { color: '#2b2b43' },
      },
      crosshair: {
        mode: 1, // Normal mode
      },
      rightPriceScale: {
        borderVisible: false,
        scaleMargins: {
          top: 0.1,
          bottom: 0.2, // 거래량을 위한 하단 여백 증가
        },
      },
      timeScale: {
        borderVisible: false,
        timeVisible: true, // 항상 시간 표시
        secondsVisible: false,
        rightOffset: 1,
        barSpacing: 10,
        minBarSpacing: 3,
      },
    })

    chartInstanceRef.current = chart

    // 캔들스틱 시리즈 추가 (v4 API) - 기본 priceScale 사용
    const candleSeries = chart.addCandlestickSeries({
      upColor: '#ef5350',
      downColor: '#26a69a',
      borderVisible: true,
      wickUpColor: '#ef5350',
      wickDownColor: '#26a69a',
      borderUpColor: '#ef5350',
      borderDownColor: '#26a69a',
    })
    candlestickSeriesRef.current = candleSeries

    // 거래량 시리즈 추가 (v4 API) - 별도 priceScale 사용
    const volumeSeries = chart.addHistogramSeries({
      color: '#26a69a',
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: 'volume', // 별도 스케일 ID 지정
    })

    // 거래량 priceScale 설정
    chart.priceScale('volume').applyOptions({
      scaleMargins: {
        top: 0.8, // 상단 85% 지점부터 시작
        bottom: 0, // 하단 0%
      },
    })

    volumeSeriesRef.current = volumeSeries

    // 데이터 변환 및 설정
    let candleData = []
    let volumeData = []
    const seenTimes = new Set() // 중복 시간값 체크

    // 일봉/주봉/월봉: YYYYMMDD → YYYY-MM-DD
    sortedData.forEach(d => {
      if (!d.date || d.date.length < 8) return // 유효하지 않은 데이터 건너뛰기

      const time = `${d.date.slice(0, 4)}-${d.date.slice(4, 6)}-${d.date.slice(6, 8)}`

      // 중복 시간값 제거
      if (seenTimes.has(time)) return
      seenTimes.add(time)

      // 유효한 OHLC 데이터인지 확인
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
          color: d.close >= d.open ? '#ef535080' : '#26a69a80',
        })
      }
    })

    candleSeries.setData(candleData)
    volumeSeries.setData(volumeData)

    // 차트 자동 맞춤
    chart.timeScale().fitContent()

    // 리사이즈 핸들러
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

    // 09:00 ~ 15:30 (0900 ~ 1530)
    return currentTime >= 900 && currentTime <= 1530
  }

  // 주기적 데이터 갱신 - 장 운영 시간에만 3초마다
  useEffect(() => {
    // 초기 로드 (loading 표시)
    fetchStockDetail(true)

    // 장 운영 시간에만 3초마다 자동 갱신 (loading 표시 안함)
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

  if (loading && !stockData) {
    return (
      <div className="stock-detail">
        <div className="loading">데이터를 불러오는 중...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="stock-detail">
        <div className="error">{error}</div>
      </div>
    )
  }

  if (!stockData) {
    return (
      <div className="stock-detail">
        <div className="error">데이터를 찾을 수 없습니다</div>
      </div>
    )
  }

  const { basic_info, chart_data, prediction } = stockData

  return (
    <div className="stock-detail">
      {/* 헤더 - 종목 기본 정보 */}
      <div className="stock-header">
        <div className="stock-title">
          <h1>{basic_info.stock_name}</h1>
          <span className="market-badge">{basic_info.market}</span>
          <span className="stock-code">{basic_info.stock_code}</span>
        </div>
        <div className="stock-price-info">
          <div className="current-price">
            {formatPrice(basic_info.current_price)}
          </div>
          <div className={`price-change ${getPriceChangeClass(basic_info.change_price)}`}>
            {basic_info.change_price > 0 ? '+' : ''}
            {formatPrice(basic_info.change_price)} ({basic_info.change_rate > 0 ? '+' : ''}
            {basic_info.change_rate.toFixed(2)}%)
          </div>
        </div>
      </div>

      {/* 차트 */}
      <div className="card">
        <div className="card-header">
          <h2>차트</h2>
          <div className="period-selector">
            <button
              className={period === 'D' ? 'active' : ''}
              onClick={() => setPeriod('D')}
            >
              일
            </button>
            <button
              className={period === 'W' ? 'active' : ''}
              onClick={() => setPeriod('W')}
            >
              주
            </button>
            <button
              className={period === 'M' ? 'active' : ''}
              onClick={() => setPeriod('M')}
            >
              월
            </button>
          </div>
        </div>
        <div className="chart-container">
          {chart_data && chart_data.length > 0 ? (
            <div ref={chartContainerRef} style={{ position: 'relative', width: '100%', height: '500px' }} />
          ) : (
            <p>차트 데이터가 없습니다</p>
          )}
        </div>
      </div>

      {/* 기본 정보 */}
      <div className="card">
        <h2>기본 정보</h2>
        <div className="info-grid">
          <div className="info-item">
            <span className="info-label">현재가</span>
            <span className="info-value">{formatPrice(basic_info.current_price)}</span>
          </div>
          <div className="info-item">
            <span className="info-label">전일대비</span>
            <span className={`info-value ${getPriceChangeClass(basic_info.change_price)}`}>
              {basic_info.change_price > 0 ? '+' : ''}
              {formatPrice(basic_info.change_price)}
            </span>
          </div>
          <div className="info-item">
            <span className="info-label">등락률</span>
            <span className={`info-value ${getPriceChangeClass(basic_info.change_rate)}`}>
              {basic_info.change_rate > 0 ? '+' : ''}
              {basic_info.change_rate.toFixed(2)}%
            </span>
          </div>
          <div className="info-item">
            <span className="info-label">거래량</span>
            <span className="info-value">{formatNumber(basic_info.volume)}</span>
          </div>
          {basic_info.market_cap && (
            <div className="info-item">
              <span className="info-label">시가총액</span>
              <span className="info-value">{formatPrice(basic_info.market_cap)}</span>
            </div>
          )}
          {basic_info.per && (
            <div className="info-item">
              <span className="info-label">PER</span>
              <span className="info-value">{basic_info.per.toFixed(2)}</span>
            </div>
          )}
          {basic_info.pbr && (
            <div className="info-item">
              <span className="info-label">PBR</span>
              <span className="info-value">{basic_info.pbr.toFixed(2)}</span>
            </div>
          )}
        </div>
      </div>

      {/* 예측 정보 */}
      {prediction && (
        <div className="card">
          <h2>AI 예측</h2>
          <div className="prediction-container">
            <div className="prediction-main">
              <div className="prediction-item">
                <span className="prediction-label">예측가</span>
                <span className="prediction-value price-highlight">
                  {formatPrice(Math.round(prediction.predicted_price))}
                </span>
              </div>
              <div className="prediction-item">
                <span className="prediction-label">예측일</span>
                <span className="prediction-value">{prediction.prediction_date}</span>
              </div>
            </div>
            <div className="prediction-details">
              <div className="prediction-item">
                <span className="prediction-label">추세</span>
                <span className={`prediction-badge trend-${prediction.trend.includes('상승') ? 'up' : prediction.trend.includes('하락') ? 'down' : 'neutral'}`}>
                  {prediction.trend}
                </span>
              </div>
              <div className="prediction-item">
                <span className="prediction-label">투자의견</span>
                <span className={`prediction-badge recommend-${prediction.recommendation}`}>
                  {prediction.recommendation}
                </span>
              </div>
              <div className="prediction-item">
                <span className="prediction-label">신뢰도</span>
                <span className="prediction-value">
                  {(prediction.confidence * 100).toFixed(0)}%
                </span>
              </div>
            </div>
            <div className="prediction-note">
              * 이 예측은 단순 이동평균 기반 분석이며, 투자 권유가 아닙니다.
            </div>
          </div>
        </div>
      )}

      {/* 뉴스 섹션 */}
      <div className="card">
        <h2>관련 뉴스 ({totalNews})</h2>

        <div className="news-list">
          {news.length === 0 && !newsLoading ? (
            <div className="no-news">
              최근 뉴스가 없습니다.
            </div>
          ) : (
            news.map((item) => (
              <a
                key={item.id}
                href={item.url}
                target="_blank"
                rel="noopener noreferrer"
                className="news-item"
              >
                <div className="news-content">
                  <div className="news-title">{item.title}</div>
                  <div className="news-meta">
                    {item.source && <span className="news-source">{item.source}</span>}
                    <span className="news-time">{formatNewsTime(item.published_at)}</span>
                  </div>
                </div>
                {item.image_url && (
                  <div className="news-thumbnail">
                    <img src={item.image_url} alt="" loading="lazy" />
                  </div>
                )}
              </a>
            ))
          )}

          {newsLoading && (
            <div className="news-loading">뉴스를 불러오는 중...</div>
          )}

          {hasMoreNews && !newsLoading && (
            <button
              className="load-more-btn"
              onClick={() => fetchNews(newsPage + 1)}
            >
              더 보기
            </button>
          )}
        </div>
      </div>

      {/* 댓글 섹션 */}
      <div className="card">
        <h2>커뮤니티 ({totalComments})</h2>

        {/* 댓글 작성 폼 */}
        {isLoggedIn ? (
          <form className="comment-form" onSubmit={handleSubmitComment}>
            <div className="comment-input-wrapper">
              <textarea
                className="comment-input"
                placeholder="이 종목에 대한 의견을 남겨주세요..."
                value={newComment}
                onChange={(e) => setNewComment(e.target.value)}
                maxLength={1000}
                rows={3}
              />
              <div className="comment-form-footer">
                <span className="char-count">{newComment.length}/1000</span>
                <button
                  type="submit"
                  className="comment-submit-btn"
                  disabled={!newComment.trim() || commentSubmitting}
                >
                  {commentSubmitting ? '작성 중...' : '작성'}
                </button>
              </div>
            </div>
          </form>
        ) : (
          <div className="login-prompt">
            로그인하면 댓글을 작성할 수 있습니다.
          </div>
        )}

        {/* 댓글 목록 */}
        <div className="comments-list">
          {comments.length === 0 && !commentLoading ? (
            <div className="no-comments">
              아직 댓글이 없습니다. 첫 번째 댓글을 남겨보세요!
            </div>
          ) : (
            comments.map((comment) => (
              <div key={comment.id} className="comment-item">
                <div className="comment-header">
                  <span className="comment-author">{comment.username}</span>
                  <span className="comment-time">{formatCommentTime(comment.created_at)}</span>
                  {comment.updated_at && (
                    <span className="comment-edited">(수정됨)</span>
                  )}
                </div>

                {editingCommentId === comment.id ? (
                  <div className="comment-edit-form">
                    <textarea
                      className="comment-input"
                      value={editingContent}
                      onChange={(e) => setEditingContent(e.target.value)}
                      maxLength={1000}
                      rows={3}
                    />
                    <div className="comment-edit-actions">
                      <button
                        className="btn-cancel"
                        onClick={() => {
                          setEditingCommentId(null)
                          setEditingContent('')
                        }}
                      >
                        취소
                      </button>
                      <button
                        className="btn-save"
                        onClick={() => handleUpdateComment(comment.id)}
                        disabled={!editingContent.trim() || commentSubmitting}
                      >
                        저장
                      </button>
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="comment-content">{comment.content}</div>
                    {comment.is_mine && (
                      <div className="comment-actions">
                        <button
                          className="btn-edit"
                          onClick={() => {
                            setEditingCommentId(comment.id)
                            setEditingContent(comment.content)
                          }}
                        >
                          수정
                        </button>
                        <button
                          className="btn-delete"
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

          {commentLoading && (
            <div className="comment-loading">댓글을 불러오는 중...</div>
          )}

          {hasMoreComments && !commentLoading && (
            <button
              className="load-more-btn"
              onClick={() => fetchComments(commentPage + 1)}
            >
              더 보기
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

export default StockDetail
