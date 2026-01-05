import { Outlet, Link } from 'react-router-dom'
import './Layout.css'

function Layout() {
  return (
    <div className="layout">
      <header className="header">
        <div className="header-content">
          <h1 className="logo">StockSense</h1>
          <nav className="nav">
            <Link to="/">홈</Link>
            <Link to="/portfolio">포트폴리오</Link>
            <Link to="/prediction">예측</Link>
          </nav>
        </div>
      </header>
      <main className="main-content">
        <Outlet />
      </main>
      <footer className="footer">
        <p>&copy; 2026 StockSense. All rights reserved.</p>
      </footer>
    </div>
  )
}

export default Layout
