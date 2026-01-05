import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Home from './pages/Home'
import StockDetail from './pages/StockDetail'
import Prediction from './pages/Prediction'
import Portfolio from './pages/Portfolio'
import './App.css'

function App() {
  return (
    <Routes>
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
