import { useParams } from 'react-router-dom'
import './StockDetail.css'

function StockDetail() {
  const { symbol } = useParams()

  return (
    <div className="stock-detail">
      <h1>{symbol}</h1>
      <p>종목 상세 정보 (구현 예정)</p>

      <div className="card">
        <h2>차트</h2>
        <p>주가 차트가 여기에 표시됩니다</p>
      </div>

      <div className="card">
        <h2>기본 정보</h2>
        <p>종목의 기본 정보가 여기에 표시됩니다</p>
      </div>

      <div className="card">
        <h2>예측</h2>
        <p>AI 기반 주가 예측 결과가 여기에 표시됩니다</p>
      </div>
    </div>
  )
}

export default StockDetail
