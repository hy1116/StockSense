import { Outlet, Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import './Layout.css'

function Layout() {
  const navigate = useNavigate()
  const { isLoggedIn, user, logout } = useAuth()

  const handleLogout = () => {
    logout()
    navigate('/')
  }

  return (
    <div className="layout">
      <header className="header">
        <div className="header-content">
          <h1 className="logo"><Link to="/">StockSense</Link></h1>
          <nav className="nav">
            <Link to="/">홈</Link>
            {isLoggedIn && <Link to="/portfolio">포트폴리오</Link>}
            <Link to="/prediction">예측</Link>
          </nav>
          <div className="auth-section">
            {isLoggedIn ? (
              <>
                <span className="account-info">{user?.accountNo}</span>
                <button onClick={handleLogout} className="auth-button logout">
                  로그아웃
                </button>
              </>
            ) : (
              <Link to="/login" className="auth-button login">
                로그인
              </Link>
            )}
          </div>
        </div>
      </header>
      <main className="main-content">
        <Outlet />
      </main>
      <footer className="footer">
        <p className="deployed-by">Deployed by <strong>Hyunyoung</strong> : stable build v1.0</p>
        <p>&copy; 2026 StockSense. All rights reserved.</p>
      </footer>
    </div>
  )
}

export default Layout
