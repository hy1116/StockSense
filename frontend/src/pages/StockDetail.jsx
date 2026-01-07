import { useParams } from 'react-router-dom'
import { useState, useEffect } from 'react'
import axios from 'axios'
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

const API_BASE_URL = 'http://localhost:8000'

function StockDetail() {
  const { symbol } = useParams()
  const [stockData, setStockData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [period, setPeriod] = useState('D')

  useEffect(() => {
    fetchStockDetail()
  }, [symbol, period])

  const fetchStockDetail = async () => {
    setLoading(true)
    setError(null)

    try {
      const response = await axios.get(
        `${API_BASE_URL}/api/portfolio/stock/${symbol}/detail`,
        { params: { period } }
      )
      setStockData(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || '데이터를 불러오는데 실패했습니다')
      console.error('Error fetching stock detail:', err)
    } finally {
      setLoading(false)
    }
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
    </div>
  )
}

export default StockDetail
