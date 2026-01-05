import './Prediction.css'

function Prediction() {
  return (
    <div className="prediction">
      <h1>AI 주가 예측</h1>

      <div className="card">
        <h2>예측 설정</h2>
        <form className="prediction-form">
          <div className="form-group">
            <label htmlFor="symbol">종목 코드</label>
            <input
              type="text"
              id="symbol"
              placeholder="예: AAPL"
            />
          </div>

          <div className="form-group">
            <label htmlFor="days">예측 기간 (일)</label>
            <input
              type="number"
              id="days"
              defaultValue="30"
              min="1"
              max="365"
            />
          </div>

          <button type="submit">예측 실행</button>
        </form>
      </div>

      <div className="card">
        <h2>예측 결과</h2>
        <p>예측 결과가 여기에 표시됩니다</p>
      </div>
    </div>
  )
}

export default Prediction
