import { useState } from 'react'
import './Prediction.css'

const API_BASE_URL = import.meta.env.VITE_API_URL || ''

function Prediction() {
  const [stockCode, setStockCode] = useState('')
  const [loading, setLoading] = useState(false)
  const [prediction, setPrediction] = useState(null)
  const [error, setError] = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()

    if (!stockCode.trim()) {
      setError('종목 코드를 입력해주세요')
      return
    }

    setLoading(true)
    setError('')
    setPrediction(null)

    try {
      const url = "/api/prediction/"
      console.log('🚀 Calling API:', url, 'with stock_code:', stockCode.toUpperCase())

      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          stock_code: stockCode.toUpperCase()
        })
      })

      console.log('📡 Response status:', response.status)

      if (!response.ok) {
        const errorData = await response.json()
        console.error('❌ Error response:', errorData)
        throw new Error(errorData.detail || '예측을 가져오는데 실패했습니다')
      }

      const data = await response.json()
      console.log('✅ Prediction data received:', data)

      // 데이터 유효성 검사
      if (!data || data.current_price === 0 || data.predicted_price === 0) {
        console.warn('⚠️ Invalid prediction data - prices are 0')
        throw new Error('예측 데이터가 유효하지 않습니다. 다시 시도해주세요.')
      }

      setPrediction(data)
    } catch (err) {
      console.error('예측 오류:', err)
      setError(err.message || '예측 중 오류가 발생했습니다')
    } finally {
      setLoading(false)
    }
  }

  const formatPrice = (price) => {
    return new Intl.NumberFormat('ko-KR').format(Math.round(price))
  }

  const getChangePercent = () => {
    if (!prediction) return 0
    return (((prediction.predicted_price - prediction.current_price) / prediction.current_price) * 100).toFixed(2)
  }

  const getRecommendationColor = (recommendation) => {
    if (recommendation.includes('매수')) return '#4caf50'
    if (recommendation.includes('매도')) return '#f44336'
    return '#ff9800'
  }

  return (
    <div className="prediction">
      <h1>AI 주가 예측</h1>

      <div className="card">
        <h2>예측 설정</h2>
        <form className="prediction-form" onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="symbol">종목 코드</label>
            <input
              type="text"
              id="symbol"
              placeholder="예: 005930 (삼성전자)"
              value={stockCode}
              onChange={(e) => setStockCode(e.target.value)}
              disabled={loading}
            />
          </div>

          <button type="submit" disabled={loading}>
            {loading ? '예측 중...' : '예측 실행'}
          </button>
        </form>

        {error && (
          <div className="error-message" style={{ marginTop: '1rem', color: '#f44336' }}>
            {error}
          </div>
        )}
      </div>

      <div className="card">
        <h2>예측 결과</h2>
        {!prediction && !loading && (
          <p>종목 코드를 입력하고 예측 실행 버튼을 클릭하세요</p>
        )}

        {loading && (
          <p>예측 중입니다...</p>
        )}

        {prediction && (
          <div className="prediction-result">
            <div className="stock-info">
              <h3>{prediction.stock_name} ({prediction.stock_code})</h3>
            </div>

            <div className="prediction-details">
              <div className="detail-row">
                <span className="label">현재가</span>
                <span className="value">{formatPrice(prediction.current_price)}원</span>
              </div>

              <div className="detail-row">
                <span className="label">예측가</span>
                <span className="value price-highlight">
                  {formatPrice(prediction.predicted_price)}원
                </span>
              </div>

              <div className="detail-row">
                <span className="label">변동 예상</span>
                <span className={`value ${getChangePercent() >= 0 ? 'positive' : 'negative'}`}>
                  {getChangePercent() >= 0 ? '+' : ''}{getChangePercent()}%
                </span>
              </div>

              <div className="detail-row">
                <span className="label">예측 날짜</span>
                <span className="value">{prediction.prediction_date}</span>
              </div>

              <div className="detail-row">
                <span className="label">신뢰도</span>
                <span className="value">{(prediction.confidence * 100).toFixed(0)}%</span>
              </div>

              <div className="detail-row">
                <span className="label">추세</span>
                <span className="value">{prediction.trend}</span>
              </div>

              <div className="detail-row">
                <span className="label">투자의견</span>
                <span
                  className="value recommendation"
                  style={{
                    color: getRecommendationColor(prediction.recommendation),
                    fontWeight: 'bold'
                  }}
                >
                  {prediction.recommendation}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default Prediction
