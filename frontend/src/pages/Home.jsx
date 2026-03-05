import { useState, useEffect, useRef } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { getHealthCheck, getTopStocks, getMarketCapStocks, getFluctuationStocks, getPortfolio, searchStocks, getWatchlist } from '../services/api'
import './Home.css'

function Home() {
  const { isLoggedIn } = useAuth();
  const queryClient = useQueryClient();
  const [searchTerm, setSearchTerm] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [showDropdown, setShowDropdown] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const [assetExpanded, setAssetExpanded] = useState(false);
  const searchRef = useRef(null);
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();

  const activeTab = searchParams.get('tab') || (isLoggedIn ? 'holdings' : 'marketCap');

  const handleTabChange = (tabName) => {
    // 뒤로가기를 지원하려면 replace: false (기본값)로 두세요.
    setSearchParams({ tab: tabName });
  };

  const { data: health, isLoading } = useQuery({
    queryKey: ['health'],
    queryFn: getHealthCheck,
  })

  const { data: topStocksData, isLoading: isLoadingStocks } = useQuery({
    queryKey: ['topStocks'],
    queryFn: () => getTopStocks(20),
    refetchInterval: 60000,
    enabled: activeTab === 'volume',
  })

  const { data: marketCapData, isLoading: isLoadingMarketCap } = useQuery({
    queryKey: ['marketCapStocks'],
    queryFn: () => getMarketCapStocks(20),
    refetchInterval: 60000,
    enabled: activeTab === 'marketCap',
  })

  const { data: fluctuationAscStocksData, isLoading: isLoadingFluctuationAsc } = useQuery({
    queryKey: ['fluctuationStocks', 0],
    queryFn: () => getFluctuationStocks(20, 0),
    refetchInterval: 60000,
    enabled: activeTab === 'fluctuation_asc',
  })

  const { data: fluctuationDescStocksData, isLoading: isLoadingFluctuationDesc } = useQuery({
    queryKey: ['fluctuationStocks', 1],
    queryFn: () => getFluctuationStocks(20, 1),
    refetchInterval: 60000,
    enabled: activeTab === 'fluctuation_desc',
  })

  // 관심종목
  const { data: watchlistData, isLoading: isLoadingWatchlist } = useQuery({
    queryKey: ['watchlist'],
    queryFn: getWatchlist,
    refetchInterval: 60000,
    enabled: isLoggedIn && activeTab === 'watchlist',
  })

  // HTTP polling 잔고 조회 (30초 간격)
  const { data: portfolio, isLoading: isLoadingPortfolio } = useQuery({
    queryKey: ['portfolio'],
    queryFn: getPortfolio,
    refetchInterval: 30000,
    enabled: isLoggedIn,
  })

  const formatNumber = (num) => {
    if (!num) return '0'
    return num.toLocaleString('ko-KR')
  }

  const formatMarketCap = (num) => {
    if (!num) return '-'
    
    if (num >= 10000) {
      return `${(num / 10000).toFixed(1)}조`
    }
    return `${formatNumber(num)}억`
  }

  const getPriceChangeClass = (change) => {
    if (change > 0) return 'price-up'
    if (change < 0) return 'price-down'
    return ''
  }

  // 검색어 변경 시 자동완성 검색
  useEffect(() => {
    const delaySearch = setTimeout(async () => {
      if (searchTerm.trim().length >= 1) {
        setIsSearching(true)
        try {
          const data = await searchStocks(searchTerm, 10)
          setSearchResults(data.results || [])
          setShowDropdown(true)
          setSelectedIndex(-1)
        } catch (error) {
          console.error('Search error:', error)
          setSearchResults([])
        } finally {
          setIsSearching(false)
        }
      } else {
        setSearchResults([])
        setShowDropdown(false)
      }
    }, 300) // 300ms 디바운스

    return () => clearTimeout(delaySearch)
  }, [searchTerm])

  // 외부 클릭 시 드롭다운 닫기
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (searchRef.current && !searchRef.current.contains(event.target)) {
        setShowDropdown(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const handleSearch = (e) => {
    e.preventDefault()
    if (selectedIndex >= 0 && searchResults[selectedIndex]) {
      navigate(`/stock/${searchResults[selectedIndex].stock_code}`)
      setShowDropdown(false)
    } else if (searchResults.length === 1) {
      // 결과가 하나면 바로 이동
      navigate(`/stock/${searchResults[0].stock_code}`)
      setShowDropdown(false)
    } else if (searchTerm.trim().match(/^\d{6}$/)) {
      // 6자리 숫자면 종목코드로 직접 이동
      navigate(`/stock/${searchTerm.trim()}`)
      setShowDropdown(false)
    }
    // 결과 없거나 이름 검색인 경우 드롭다운 유지 (잘못된 URL 이동 방지)
  }

  const handleKeyDown = (e) => {
    if (!showDropdown || searchResults.length === 0) return

    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setSelectedIndex((prev) => (prev < searchResults.length - 1 ? prev + 1 : prev))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setSelectedIndex((prev) => (prev > 0 ? prev - 1 : -1))
    } else if (e.key === 'Escape') {
      setShowDropdown(false)
    }
  }

  const handleResultClick = (stockCode) => {
    navigate(`/stock/${stockCode}`)
    setShowDropdown(false)
    setSearchTerm('')
  }

  // 포트폴리오 새로고침 핸들러
  const handleRefreshPortfolio = () => {
    queryClient.invalidateQueries({ queryKey: ['portfolio'] })
  }

  // 1. 데이터와 로딩 상태를 객체에 담기
  const stockDataMap = {
    volume: topStocksData?.stocks,
    marketCap: marketCapData?.stocks,
    fluctuation_asc: fluctuationAscStocksData?.stocks,
    fluctuation_desc: fluctuationDescStocksData?.stocks,

  };

  const loadingStateMap = {
    volume: isLoadingStocks,
    marketCap: isLoadingMarketCap,
    fluctuation_asc: isLoadingFluctuationAsc,
    fluctuation_desc: isLoadingFluctuationDesc,
  };

  // 2. 현재 활성화된 탭에 맞는 값 추출
  const currentStocks = stockDataMap[activeTab] || [];
  const isLoadingCurrentStocks = loadingStateMap[activeTab];

  return (
    <div className="home">
      {/* 자산 Hero — 로그인 시 자산 메인, 비로그인 시 브랜드 표시 */}
      {isLoggedIn && portfolio ? (
        <section className="asset-hero">
          <div className="asset-hero-main">
            <span className="asset-hero-label">보유주식</span>
            <div className="asset-hero-price-row">
              <span className="asset-hero-price">
                {formatNumber(portfolio.total_asset)}<span className="asset-hero-unit">원</span>
              </span>
              <button
                className="refresh-button"
                onClick={handleRefreshPortfolio}
                title="새로고침"
              >
                &#x21bb;
              </button>
            </div>
            <div className={`asset-hero-change ${getPriceChangeClass(portfolio.total_profit_rate)}`}>
              {portfolio.total_profit_rate > 0 ? '+' : ''}{portfolio.total_profit_rate.toFixed(2)}%
              {' '}({portfolio.total_profit_loss > 0 ? '+' : ''}{formatNumber(portfolio.total_profit_loss)}원)
            </div>
            <span className="asset-hero-sub">총 투자금 {formatNumber(portfolio.total_asset - portfolio.total_profit_loss)}원</span>
          </div>
          <button
            className="asset-hero-toggle"
            onClick={() => setAssetExpanded(!assetExpanded)}
          >
            {assetExpanded ? '접기' : '상세'}
            <span className={`asset-arrow ${assetExpanded ? 'expanded' : ''}`}>&#9662;</span>
          </button>
          {assetExpanded && (
            <div className="asset-detail">
              <div className="asset-detail-row">
                <span className="asset-detail-label">보유 현금</span>
                <span className="asset-detail-value">{formatNumber(portfolio.cash)}원</span>
              </div>
              <div className="asset-detail-row">
                <span className="asset-detail-label">주식 평가액</span>
                <span className="asset-detail-value">{formatNumber(portfolio.stock_eval_amount)}원</span>
              </div>
              <div className="asset-detail-row">
                <span className="asset-detail-label">평가 손익</span>
                <span className={`asset-detail-value ${getPriceChangeClass(portfolio.total_profit_rate)}`}>
                  {portfolio.total_profit_loss > 0 ? '+' : ''}{formatNumber(portfolio.total_profit_loss)}원
                </span>
              </div>
            </div>
          )}
        </section>
      ) : isLoggedIn && isLoadingPortfolio ? (
        <section className="asset-hero">
          <div className="loading">자산 정보를 불러오는 중...</div>
        </section>
      ) : (
        <section className="hero">
          <h1>StockSense</h1>
          <p>AI 기반 주식 예측 · 분석</p>
          {!isLoading && health && (
            <span className="status-badge">서버 정상</span>
          )}
        </section>
      )}

      {/* 검색 바 */}
      <section className="search-section">
        <div className="search-container" ref={searchRef}>
          <form onSubmit={handleSearch} className="search-form">
            <span className="search-icon">&#128269;</span>
            <input
              type="text"
              placeholder="종목명 또는 코드 검색"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              onKeyDown={handleKeyDown}
              onFocus={() => searchTerm.trim().length >= 1 && setShowDropdown(true)}
              className="search-input"
            />
          </form>
          {showDropdown && (
            <div className="search-dropdown">
              {isSearching ? (
                <div className="search-dropdown-empty">검색 중...</div>
              ) : searchResults.length > 0 ? (
                searchResults.map((stock, index) => (
                  <div
                    key={stock.stock_code}
                    className={`search-dropdown-item ${index === selectedIndex ? 'selected' : ''}`}
                    onClick={() => handleResultClick(stock.stock_code)}
                    onMouseEnter={() => setSelectedIndex(index)}
                  >
                    <span className="dropdown-name">{stock.stock_name}</span>
                    <span className="dropdown-code">{stock.stock_code}</span>
                    {stock.market && <span className="dropdown-market">{stock.market}</span>}
                  </div>
                ))
              ) : (
                <div className="search-dropdown-empty">검색 결과가 없습니다</div>
              )}
            </div>
          )}
        </div>
      </section>

      {/* 상위 종목 섹션 - 탭 형태 */}
      <section className="ranking-section">
        <div className="ranking-header">
          <div className="tab-buttons">
            {isLoggedIn && (
              <button
                className={`tab-button ${activeTab === 'holdings' ? 'active' : ''}`}
                onClick={() => handleTabChange('holdings')}
              >
                보유 종목
              </button>
            )}
            {isLoggedIn && (
              <button
                className={`tab-button ${activeTab === 'watchlist' ? 'active' : ''}`}
                onClick={() => handleTabChange('watchlist')}
              >
                관심종목
              </button>
            )}
            <button
              className={`tab-button ${activeTab === 'marketCap' ? 'active' : ''}`}
              onClick={() => handleTabChange('marketCap')}
            >
              시가총액 상위
            </button>
            <button
              className={`tab-button ${activeTab === 'volume' ? 'active' : ''}`}
              onClick={() => handleTabChange('volume')}
            >
              거래량 상위
            </button>
            <button
              className={`tab-button ${activeTab === 'fluctuation_asc' ? 'active' : ''}`}
              onClick={() => handleTabChange('fluctuation_asc')}
            >
              상승률 상위
            </button>
            <button
              className={`tab-button ${activeTab === 'fluctuation_desc' ? 'active' : ''}`}
              onClick={() => handleTabChange('fluctuation_desc')}
            >
              하락률 상위
            </button>
          </div>
        </div>

        {/* 거래량/시총 상위 탭 */}
        {(activeTab === 'volume' || activeTab === 'marketCap') && (
          isLoadingCurrentStocks ? (
            <div className="loading">종목 정보를 불러오는 중...</div>
          ) : currentStocks ? (
            <div className="stock-list">
              <div className="stock-list-header">
                <span className="col-rank">순위</span>
                <span className="col-name">종목명</span>
                <span className="col-price">현재가</span>
                <span className="col-change">등락률</span>
                {activeTab === 'marketCap' && <span className="col-marketcap">시가총액</span>}
              </div>
              {currentStocks.map((stock) => (
                <Link
                  key={stock.stock_code}
                  to={`/stock/${stock.stock_code}`}
                  className="stock-list-item"
                >
                  <span className="col-rank">
                    <span className="rank-badge">{stock.rank}</span>
                  </span>
                  <span className="col-name">
                    <span className="stock-name">{stock.stock_name}</span>
                    <span className="stock-code">{stock.stock_code}</span>
                  </span>
                  <span className="col-price">{formatNumber(stock.current_price)}원</span>
                  <span className={`col-change ${getPriceChangeClass(stock.change_rate)}`}>
                    {stock.change_rate > 0 ? '+' : ''}{stock.change_rate.toFixed(2)}%
                  </span>
                  {activeTab === 'marketCap' && (
                    <span className="col-marketcap">{formatMarketCap(stock.market_cap)}</span>
                  )}
                </Link>
              ))}
            </div>
          ) : (
            <div className="error">종목 정보를 불러올 수 없습니다</div>
          )
        )}

        {/* 상승률 상위 탭 */}
        {activeTab === 'fluctuation_asc' && (
          isLoadingCurrentStocks ? (
            <div className="loading">종목 정보를 불러오는 중...</div>
          ) : currentStocks ? (
            <div className="stock-list">
              <div className="stock-list-header">
                <span className="col-rank">순위</span>
                <span className="col-name">종목명</span>
                <span className="col-price">현재가</span>
                <span className="col-change">등락률</span>
              </div>
              {currentStocks.map((stock) => (
                <Link
                  key={stock.stock_code}
                  to={`/stock/${stock.stock_code}`}
                  className="stock-list-item"
                >
                  <span className="col-rank">
                    <span className="rank-badge">{stock.rank}</span>
                  </span>
                  <span className="col-name">
                    <span className="stock-name">{stock.stock_name}</span>
                    <span className="stock-code">{stock.stock_code}</span>
                  </span>
                  <span className="col-price">{formatNumber(stock.current_price)}원</span>
                  <span className={`col-change ${getPriceChangeClass(stock.change_rate)}`}>
                    {stock.change_rate > 0 ? '+' : ''}{stock.change_rate.toFixed(2)}%
                  </span>
                </Link>
              ))}
            </div>
          ) : (
            <div className="error">종목 정보를 불러올 수 없습니다</div>
          )
        )}

        {/* 하락률 상위 탭 */}
        {activeTab === 'fluctuation_desc' && (
          isLoadingCurrentStocks ? (
            <div className="loading">종목 정보를 불러오는 중...</div>
          ) : currentStocks ? (
            <div className="stock-list">
              <div className="stock-list-header">
                <span className="col-rank">순위</span>
                <span className="col-name">종목명</span>
                <span className="col-price">현재가</span>
                <span className="col-change">등락률</span>
              </div>
              {currentStocks.map((stock) => (
                <Link
                  key={stock.stock_code}
                  to={`/stock/${stock.stock_code}`}
                  className="stock-list-item"
                >
                  <span className="col-rank">
                    <span className="rank-badge">{stock.rank}</span>
                  </span>
                  <span className="col-name">
                    <span className="stock-name">{stock.stock_name}</span>
                    <span className="stock-code">{stock.stock_code}</span>
                  </span>
                  <span className="col-price">{formatNumber(stock.current_price)}원</span>
                  <span className={`col-change ${getPriceChangeClass(stock.change_rate)}`}>
                    {stock.change_rate > 0 ? '+' : ''}{stock.change_rate.toFixed(2)}%
                  </span>
                </Link>
              ))}
            </div>
          ) : (
            <div className="error">종목 정보를 불러올 수 없습니다</div>
          )
        )}

        {/* 보유 종목 탭 - 로그인 상태에서만 표시 */}
        {isLoggedIn && activeTab === 'holdings' && (
          isLoadingPortfolio ? (
            <div className="loading">보유 종목을 불러오는 중...</div>
          ) : portfolio?.holdings && portfolio.holdings.length > 0 ? (
            <div className="stock-list">
              <div className="stock-list-header holdings-header">
                <span className="col-rank">No.</span>
                <span className="col-name">종목명</span>
                <span className="col-price">현재가</span>
                <span className="col-change">수익률</span>
                <span className="col-quantity">보유수량</span>
              </div>
              {portfolio.holdings.map((holding, index) => (
                <Link
                  key={holding.stock_code}
                  to={`/stock/${holding.stock_code}`}
                  className="stock-list-item holdings-item"
                >
                  <span className="col-rank">
                    <span className="rank-badge">{index + 1}</span>
                  </span>
                  <span className="col-name">
                    <span className="stock-name">{holding.stock_name}</span>
                    <span className="stock-code">{holding.stock_code}</span>
                  </span>
                  <span className="col-price">{formatNumber(holding.current_price)}원</span>
                  <span className={`col-change ${getPriceChangeClass(holding.profit_rate)}`}>
                    {holding.profit_rate > 0 ? '+' : ''}{holding.profit_rate.toFixed(2)}%
                  </span>
                  <span className="col-quantity">{formatNumber(holding.quantity)}주</span>
                </Link>
              ))}
            </div>
          ) : (
            <div className="empty-holdings">
              <div className="empty-icon">📭</div>
              <p>보유 종목이 없습니다</p>
              <Link to="/portfolio" className="go-trade-btn">주문하러 가기</Link>
            </div>
          )
        )}

        {/* 관심종목 탭 - 로그인 상태에서만 표시 */}
        {isLoggedIn && activeTab === 'watchlist' && (
          isLoadingWatchlist ? (
            <div className="loading">관심종목을 불러오는 중...</div>
          ) : watchlistData?.items && watchlistData.items.length > 0 ? (
            <div className="stock-list">
              <div className="stock-list-header">
                <span className="col-rank">No.</span>
                <span className="col-name">종목명</span>
                <span className="col-price">현재가</span>
                <span className="col-change">등락률</span>
              </div>
              {watchlistData.items.map((item, index) => (
                <Link
                  key={item.stock_code}
                  to={`/stock/${item.stock_code}`}
                  className="stock-list-item"
                >
                  <span className="col-rank">
                    <span className="rank-badge">{index + 1}</span>
                  </span>
                  <span className="col-name">
                    <span className="stock-name">{item.stock_name}</span>
                    <span className="stock-code">{item.stock_code}</span>
                  </span>
                  <span className="col-price">
                    {item.current_price ? formatNumber(item.current_price) + '원' : '-'}
                  </span>
                  <span className={`col-change ${getPriceChangeClass(item.change_rate)}`}>
                    {item.change_rate != null
                      ? `${item.change_rate > 0 ? '+' : ''}${item.change_rate.toFixed(2)}%`
                      : '-'}
                  </span>
                </Link>
              ))}
            </div>
          ) : (
            <div className="empty-holdings">
              <div className="empty-icon">{'\u2606'}</div>
              <p>관심종목이 없습니다</p>
              <p style={{fontSize: '0.8rem', color: 'rgba(255,255,255,0.3)', marginTop: '0.5rem'}}>
                종목 상세에서 ☆ 버튼을 눌러 추가해보세요
              </p>
            </div>
          )
        )}
      </section>
    </div>
  )
}

export default Home
