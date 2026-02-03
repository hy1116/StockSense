import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import './Login.css'

function Login() {
  const navigate = useNavigate()
  const { login } = useAuth()
  const [formData, setFormData] = useState({
    apiKey: '',
    apiSecret: '',
    accountNo: ''
  })
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const [showSecrets, setShowSecrets] = useState({
    apiKey: false,
    apiSecret: false
  })

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))
    setError('')
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setIsLoading(true)
    setError('')

    // 유효성 검사
    if (!formData.apiKey.trim()) {
      setError('API Key를 입력해주세요')
      setIsLoading(false)
      return
    }
    if (!formData.apiSecret.trim()) {
      setError('API Secret을 입력해주세요')
      setIsLoading(false)
      return
    }
    if (!formData.accountNo.trim()) {
      setError('계좌번호를 입력해주세요')
      setIsLoading(false)
      return
    }

    try {
      // AuthContext의 login 함수 사용 (API 호출)
      const result = await login({
        apiKey: formData.apiKey,
        apiSecret: formData.apiSecret,
        accountNo: formData.accountNo,
      })

      if (result.success) {
        // 메인 페이지로 이동
        navigate('/')
      } else {
        setError(result.message || '로그인에 실패했습니다')
      }
    } catch (err) {
      setError(err.message || '로그인에 실패했습니다')
    } finally {
      setIsLoading(false)
    }
  }

  const toggleShowSecret = (field) => {
    setShowSecrets(prev => ({
      ...prev,
      [field]: !prev[field]
    }))
  }

  return (
    <div className="login-page">
      <div className="login-container">
        <div className="login-header">
          <h1>StockSense</h1>
          <p>한국투자증권 API 연동</p>
        </div>

        <form onSubmit={handleSubmit} className="login-form">
          <div className="form-group">
            <label htmlFor="apiKey">API Key (App Key)</label>
            <div className="input-wrapper">
              <input
                type={showSecrets.apiKey ? 'text' : 'password'}
                id="apiKey"
                name="apiKey"
                value={formData.apiKey}
                onChange={handleChange}
                placeholder="발급받은 App Key 입력"
                autoComplete="off"
              />
              <button
                type="button"
                className="toggle-visibility"
                onClick={() => toggleShowSecret('apiKey')}
              >
                {showSecrets.apiKey ? '🙈' : '👁️'}
              </button>
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="apiSecret">API Secret (App Secret)</label>
            <div className="input-wrapper">
              <input
                type={showSecrets.apiSecret ? 'text' : 'password'}
                id="apiSecret"
                name="apiSecret"
                value={formData.apiSecret}
                onChange={handleChange}
                placeholder="발급받은 App Secret 입력"
                autoComplete="off"
              />
              <button
                type="button"
                className="toggle-visibility"
                onClick={() => toggleShowSecret('apiSecret')}
              >
                {showSecrets.apiSecret ? '🙈' : '👁️'}
              </button>
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="accountNo">계좌번호</label>
            <input
              type="text"
              id="accountNo"
              name="accountNo"
              value={formData.accountNo}
              onChange={handleChange}
              placeholder="계좌번호 입력 (예: 12345678-01)"
              autoComplete="off"
            />
          </div>

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}

          <button
            type="submit"
            className="login-button"
            disabled={isLoading}
          >
            {isLoading ? '연결 중...' : '연결하기'}
          </button>
        </form>

        <div className="login-info">
          <h3>API Key 발급 안내</h3>
          <ol>
            <li>
              <a href="https://apiportal.koreainvestment.com/" target="_blank" rel="noopener noreferrer">
                한국투자증권 API 포털
              </a>
              에 접속
            </li>
            <li>회원가입 후 로그인</li>
            <li>API 신청 메뉴에서 App Key/Secret 발급</li>
            <li>발급받은 Key와 계좌번호를 위에 입력</li>
          </ol>
        </div>

        <div className="login-notice">
          <p>* 입력한 정보는 암호화되어 서버에 안전하게 저장됩니다.</p>
          <p>* 실제 거래는 본인 책임 하에 이루어집니다.</p>
        </div>
      </div>
    </div>
  )
}

export default Login
