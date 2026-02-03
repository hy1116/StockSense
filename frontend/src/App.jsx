import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Home from './pages/Home'
import StockDetail from './pages/StockDetail'
import Prediction from './pages/Prediction'
import Portfolio from './pages/Portfolio'
import Login from './pages/Login'
import Register from './pages/Register'
import './App.css'

function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />
      <Route path="/" element={<Layout />}>
        <Route index element={<Home />} />
        <Route path="stock/:symbol" element={<StockDetail />} />
        <Route path="prediction" element={<Prediction />} />
        <Route path="portfolio" element={<Portfolio />} />
      </Route>
    </Routes>
  )
}

export default App
