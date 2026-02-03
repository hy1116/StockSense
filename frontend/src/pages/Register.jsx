import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { checkUsername } from '../services/api'
import './Login.css'

function Register() {
  const navigate = useNavigate()
  const { register } = useAuth()
  const [step, setStep] = useState(1) // 1: 계정정보, 2: KIS API 정보
  const [formData, setFormData] = useState({
    username: '',
    password: '',
    passwordConfirm: '',
    kisApiKey: '',
    kisApiSecret: '',
    kisAccountNo: ''
  })
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const [usernameStatus, setUsernameStatus] = useState({ checked: false, available: false })
  const [showSecrets, setShowSecrets] = useState({
    password: false,
    kisApiKey: false,
    kisApiSecret: false
  })

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))
    setError('')

    // 아이디 변경 시 중복확인 초기화
    if (name === 'username') {
      setUsernameStatus({ checked: false, available: false })
    }
  }

  const handleCheckUsername = async () => {
    if (!formData.username.trim() || formData.username.length < 4) {
      setError('아이디는 4자 이상 입력해주세요')
      return
    }

    try {
      const result = await checkUsername(formData.username)
      setUsernameStatus({ checked: true, available: result.available })
      if (!result.available) {
        setError('이미 사용 중인 아이디입니다')
      }
    } catch (err) {
      setError('아이디 확인 중 오류가 발생했습니다')
    }
  }

  const validateStep1 = () => {
    if (!formData.username.trim() || formData.username.length < 4) {
      setError('아이디는 4자 이상 입력해주세요')
      return false
    }
    if (!usernameStatus.checked || !usernameStatus.available) {
      setError('아이디 중복확인을 해주세요')
      return false
    }
    if (!formData.password || formData.password.length < 6) {
      setError('비밀번호는 6자 이상 입력해주세요')
      return false
    }
    if (formData.password !== formData.passwordConfirm) {
      setError('비밀번호가 일치하지 않습니다')
      return false
    }
    return true
  }

  const handleNextStep = () => {
    if (validateStep1()) {
      setStep(2)
      setError('')
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setIsLoading(true)
    setError('')

    if (!formData.kisApiKey.trim()) {
      setError('API Key를 입력해주세요')
      setIsLoading(false)
      return
    }
    if (!formData.kisApiSecret.trim()) {
      setError('API Secret을 입력해주세요')
      setIsLoading(false)
      return
    }
    if (!formData.kisAccountNo.trim()) {
      setError('계좌번호를 입력해주세요')
      setIsLoading(false)
      return
    }

    try {
      const result = await register({
        username: formData.username,
        password: formData.password,
        kisApiKey: formData.kisApiKey,
        kisApiSecret: formData.kisApiSecret,
        kisAccountNo: formData.kisAccountNo
      })

      if (result.success) {
        alert('회원가입이 완료되었습니다. 로그인해주세요.')
        navigate('/login')
      } else {
        setError(result.message || '회원가입에 실패했습니다')
      }
    } catch (err) {
      setError(err.message || '회원가입에 실패했습니다')
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
          <p>회원가입</p>
        </div>

        {/* 스텝 인디케이터 */}
        <div className="step-indicator">
          <div className={`step ${step >= 1 ? 'active' : ''}`}>
            <span className="step-number">1</span>
            <span className="step-label">계정 정보</span>
          </div>
          <div className="step-line"></div>
          <div className={`step ${step >= 2 ? 'active' : ''}`}>
            <span className="step-number">2</span>
            <span className="step-label">API 연동</span>
          </div>
        </div>

        <form onSubmit={step === 1 ? (e) => { e.preventDefault(); handleNextStep(); } : handleSubmit} className="login-form">
          {step === 1 && (
            <>
              <div className="form-group">
                <label htmlFor="username">아이디</label>
                <div className="input-with-button">
                  <input
                    type="text"
                    id="username"
                    name="username"
                    value={formData.username}
                    onChange={handleChange}
                    placeholder="4자 이상 입력"
                    autoComplete="off"
                  />
                  <button
                    type="button"
                    className="check-button"
                    onClick={handleCheckUsername}
                    disabled={formData.username.length < 4}
                  >
                    중복확인
                  </button>
                </div>
                {usernameStatus.checked && usernameStatus.available && (
                  <span className="success-text">사용 가능한 아이디입니다</span>
                )}
              </div>

              <div className="form-group">
                <label htmlFor="password">비밀번호</label>
                <div className="input-wrapper">
                  <input
                    type={showSecrets.password ? 'text' : 'password'}
                    id="password"
                    name="password"
                    value={formData.password}
                    onChange={handleChange}
                    placeholder="6자 이상 입력"
                    autoComplete="new-password"
                  />
                  <button
                    type="button"
                    className="toggle-visibility"
                    onClick={() => toggleShowSecret('password')}
                  >
                    {showSecrets.password ? '🙈' : '👁️'}
                  </button>
                </div>
              </div>

              <div className="form-group">
                <label htmlFor="passwordConfirm">비밀번호 확인</label>
                <input
                  type="password"
                  id="passwordConfirm"
                  name="passwordConfirm"
                  value={formData.passwordConfirm}
                  onChange={handleChange}
                  placeholder="비밀번호 재입력"
                  autoComplete="new-password"
                />
              </div>
            </>
          )}

          {step === 2 && (
            <>
              <div className="form-group">
                <label htmlFor="kisApiKey">API Key (App Key)</label>
                <div className="input-wrapper">
                  <input
                    type={showSecrets.kisApiKey ? 'text' : 'password'}
                    id="kisApiKey"
                    name="kisApiKey"
                    value={formData.kisApiKey}
                    onChange={handleChange}
                    placeholder="한국투자증권 App Key"
                    autoComplete="off"
                  />
                  <button
                    type="button"
                    className="toggle-visibility"
                    onClick={() => toggleShowSecret('kisApiKey')}
                  >
                    {showSecrets.kisApiKey ? '🙈' : '👁️'}
                  </button>
                </div>
              </div>

              <div className="form-group">
                <label htmlFor="kisApiSecret">API Secret (App Secret)</label>
                <div className="input-wrapper">
                  <input
                    type={showSecrets.kisApiSecret ? 'text' : 'password'}
                    id="kisApiSecret"
                    name="kisApiSecret"
                    value={formData.kisApiSecret}
                    onChange={handleChange}
                    placeholder="한국투자증권 App Secret"
                    autoComplete="off"
                  />
                  <button
                    type="button"
                    className="toggle-visibility"
                    onClick={() => toggleShowSecret('kisApiSecret')}
                  >
                    {showSecrets.kisApiSecret ? '🙈' : '👁️'}
                  </button>
                </div>
              </div>

              <div className="form-group">
                <label htmlFor="kisAccountNo">계좌번호</label>
                <input
                  type="text"
                  id="kisAccountNo"
                  name="kisAccountNo"
                  value={formData.kisAccountNo}
                  onChange={handleChange}
                  placeholder="계좌번호 (숫자만 입력)"
                  autoComplete="off"
                />
              </div>
            </>
          )}

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}

          <div className="button-group">
            {step === 2 && (
              <button
                type="button"
                className="back-button"
                onClick={() => setStep(1)}
                disabled={isLoading}
              >
                이전
              </button>
            )}
            <button
              type="submit"
              className="login-button"
              disabled={isLoading}
            >
              {step === 1 ? '다음' : (isLoading ? '처리 중...' : '회원가입')}
            </button>
          </div>
        </form>

        {step === 2 && (
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
        )}

        <div className="login-footer">
          <p>이미 계정이 있으신가요? <Link to="/login">로그인</Link></p>
        </div>
      </div>
    </div>
  )
}

export default Register
