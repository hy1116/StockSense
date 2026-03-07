import { useQuery } from '@tanstack/react-query'
import { getMarketMacro } from '../services/api'
import './MarketTicker.css'

function MarketTicker() {
  const { data } = useQuery({
    queryKey: ['marketMacro'],
    queryFn: getMarketMacro,
    refetchInterval: 60000,
    staleTime: 60000,
  })

  if (!data || data.length === 0) return null

  const items = data.filter(d => d.price !== null)

  const formatPrice = (item) => {
    const p = item.price
    if (item.label === 'USD/KRW') return `${Math.round(p).toLocaleString('ko-KR')}${item.unit}`
    if (item.unit === '$') return `$${p.toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 })}`
    return p.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
  }

  return (
    <div className="market-ticker">
      <div className="ticker-track">
        {[...items, ...items].map((item, i) => (
          <span key={i} className="ticker-item">
            <span className="ticker-label">{item.label}</span>
            <span className="ticker-price">{formatPrice(item)}</span>
            <span className={`ticker-change ${item.change_pct > 0 ? 'up' : item.change_pct < 0 ? 'down' : 'flat'}`}>
              {item.change_pct > 0 ? '▲' : item.change_pct < 0 ? '▼' : ''}
              {Math.abs(item.change_pct).toFixed(2)}%
            </span>
          </span>
        ))}
      </div>
    </div>
  )
}

export default MarketTicker
