import { useParams } from 'react-router-dom'
import { useState, useEffect } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { getStockDetail, getComments, createComment, updateComment, deleteComment } from '../services/api'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js'
import { Line } from 'react-chartjs-2'
import './StockDetail.css'

// Chart.js 등록
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
)

function StockDetail() {
  const { symbol } = useParams()
  const { isLoggedIn, user } = useAuth()
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

  useEffect(() => {
    fetchStockDetail()
  }, [symbol, period])

  useEffect(() => {
    if (symbol) {
      fetchComments(1)
    }
  }, [symbol])

  const fetchStockDetail = async () => {
    setLoading(true)
    setError(null)

    try {
      const data = await getStockDetail(symbol, period)
      setStockData(data)
    } catch (err) {
      setError(err.response?.data?.detail || '데이터를 불러오는데 실패했습니다')
      console.error('Error fetching stock detail:', err)
    } finally {
      setLoading(false)
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

  // 주식 차트 데이터 준비
  const prepareChartData = () => {
    if (!stockData || !chart_data || chart_data.length === 0) {
      return null
    }

    // 날짜순 정렬 (오래된 것부터)
    const sortedData = [...chart_data].reverse()

    // 종가 데이터 추출
    const prices = sortedData.map(d => d.close)
    const dates = sortedData.map(d => {
      const date = d.date
      return `${date.slice(4, 6)}/${date.slice(6, 8)}`
    })

    // 가격 변화에 따른 색상 결정
    const firstPrice = prices[0]
    const lastPrice = prices[prices.length - 1]
    const isUp = lastPrice >= firstPrice

    return {
      labels: dates,
      datasets: [
        {
          label: '종가',
          data: prices,
          borderColor: isUp ? '#ef5350' : '#26a69a',
          backgroundColor: isUp
            ? 'rgba(239, 83, 80, 0.1)'
            : 'rgba(38, 166, 154, 0.1)',
          borderWidth: 2,
          fill: true,
          tension: 0.4,
          pointRadius: 0,
          pointHoverRadius: 4,
          pointHoverBackgroundColor: isUp ? '#ef5350' : '#26a69a',
          pointHoverBorderColor: '#fff',
          pointHoverBorderWidth: 2
        }
      ]
    }
  }

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false
    },
    plugins: {
      legend: {
        display: false
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        padding: 12,
        titleColor: '#fff',
        bodyColor: '#fff',
        borderColor: 'rgba(255, 255, 255, 0.2)',
        borderWidth: 1,
        displayColors: false,
        callbacks: {
          title: function(context) {
            return context[0].label
          },
          label: function(context) {
            return `${formatNumber(context.parsed.y)}원`
          }
        }
      }
    },
    scales: {
      x: {
        grid: {
          display: false
        },
        ticks: {
          color: 'rgba(255, 255, 255, 0.6)',
          maxTicksLimit: 8
        }
      },
      y: {
        position: 'right',
        grid: {
          color: 'rgba(255, 255, 255, 0.05)',
          drawBorder: false
        },
        ticks: {
          color: 'rgba(255, 255, 255, 0.6)',
          callback: function(value) {
            return formatNumber(value)
          }
        }
      }
    }
  }

  if (loading) {
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
              일봉
            </button>
            <button
              className={period === 'W' ? 'active' : ''}
              onClick={() => setPeriod('W')}
            >
              주봉
            </button>
            <button
              className={period === 'M' ? 'active' : ''}
              onClick={() => setPeriod('M')}
            >
              월봉
            </button>
          </div>
        </div>
        <div className="chart-container">
          {chart_data && chart_data.length > 0 ? (
            <div style={{ height: '400px', padding: '1rem' }}>
              <Line data={prepareChartData()} options={chartOptions} />
            </div>
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
