"""포트폴리오 API 엔드포인트"""
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_
from typing import List
from datetime import datetime
import logging

from app.schemas.portfolio import (
    PortfolioSummary,
    StockHolding,
    OrderRequest,
    OrderResponse,
    OrderHistory,
    StockPriceInfo,
    StockDetailInfo,
    StockBasicInfo,
    ChartData,
    PredictionResult,
    TopStock,
    TopStocksResponse,
    StockSearchItem,
    StockSearchResponse
)
from app.services.kis_api import get_kis_client, KISAPIClient
from app.services.prediction import get_prediction_service, PredictionService
from app.services.redis_client import get_redis_client
from app.database import get_db
from app.models.stock import Stock
import json

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/portfolio", tags=["Portfolio"])


@router.get("/search", response_model=StockSearchResponse)
async def search_stocks(
    q: str = Query(..., min_length=1, description="검색어 (종목명 또는 코드)"),
    limit: int = Query(default=10, ge=1, le=30, description="최대 결과 수"),
    db: AsyncSession = Depends(get_db)
):
    """종목명 또는 코드로 검색 (DB 기반)"""
    try:
        search_term = q.strip()

        # 종목코드 또는 종목명으로 검색 (부분 일치)
        query = select(Stock).where(
            or_(
                Stock.stock_code.contains(search_term),
                Stock.stock_name.contains(search_term)
            )
        ).limit(limit)

        result = await db.execute(query)
        stocks = result.scalars().all()

        # 정렬: 완전 일치 우선, 시작 일치 그 다음
        def sort_key(stock):
            if stock.stock_code == search_term:
                return (0, stock.stock_name)
            elif stock.stock_name == search_term:
                return (1, stock.stock_name)
            elif stock.stock_code.startswith(search_term):
                return (2, stock.stock_name)
            elif stock.stock_name.startswith(search_term):
                return (3, stock.stock_name)
            else:
                return (4, stock.stock_name)

        sorted_stocks = sorted(stocks, key=sort_key)

        results = [
            StockSearchItem(
                stock_code=stock.stock_code,
                stock_name=stock.stock_name,
                market=stock.market
            )
            for stock in sorted_stocks
        ]

        return StockSearchResponse(results=results, total=len(results))

    except Exception as e:
        logger.error(f"Stock search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/balance", response_model=PortfolioSummary)
async def get_portfolio_balance(
    client: KISAPIClient = Depends(get_kis_client)
):
    """계좌 잔고 및 포트폴리오 조회"""
    try:
        # 2. KIS API 호출
        result = client.get_balance()

        if result.get("rt_cd") != "0":
            raise HTTPException(
                status_code=400,
                detail=f"잔고 조회 실패: {result.get('msg1', 'Unknown error')}"
            )

        output1 = result.get("output1", [])
        output2 = result.get("output2", [{}])[0]

        # 디버깅: 수익률 원본 값 로그
        tot_evlu_pfls_rt = output2.get("tot_evlu_pfls_rt", 0)
        logger.info(f"Portfolio total_profit_rate raw value: {tot_evlu_pfls_rt}")

        holdings = []
        for item in output1:
            holding = StockHolding(
                stock_code=item.get("pdno", ""),
                stock_name=item.get("prdt_name", ""),
                quantity=int(item.get("hldg_qty", 0)),
                avg_price=float(item.get("pchs_avg_pric", 0)),
                current_price=float(item.get("prpr", 0)),
                eval_amount=float(item.get("evlu_amt", 0)),
                profit_loss=float(item.get("evlu_pfls_amt", 0)),
                profit_rate=float(item.get("evlu_pfls_rt", 0))
            )
            holdings.append(holding)

        # 수익률 계산
        evlu_pfls_smtl_amt=float(output2.get("evlu_pfls_smtl_amt", 0))
        pchs_amt_smtl_amt=float(output2.get("pchs_amt_smtl_amt", 0))

        summary = PortfolioSummary(
            total_asset=float(output2.get("tot_evlu_amt", 0)),
            cash=float(output2.get("dnca_tot_amt", 0)),
            stock_eval_amount=float(output2.get("scts_evlu_amt", 0)),
            total_profit_loss=evlu_pfls_smtl_amt,
            pchs_amt_smtl_amt=pchs_amt_smtl_amt,
            total_profit_rate= round(evlu_pfls_smtl_amt/pchs_amt_smtl_amt*100, 2) if pchs_amt_smtl_amt else 0,
            holdings=holdings
        )
        
        return summary

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stock/{stock_code}", response_model=StockPriceInfo)
async def get_stock_price(
    stock_code: str,
    client: KISAPIClient = Depends(get_kis_client)
):
    """주식 현재가 조회"""
    try:
        result = client.get_stock_price(stock_code)

        if result.get("rt_cd") != "0":
            raise HTTPException(
                status_code=400,
                detail=f"현재가 조회 실패: {result.get('msg1', 'Unknown error')}"
            )

        output = result.get("output", {})

        price_info = StockPriceInfo(
            stock_code=stock_code,
            stock_name=output.get("hts_kor_isnm", ""),
            current_price=int(output.get("stck_prpr", 0)),
            change_price=int(output.get("prdy_vrss", 0)),
            change_rate=float(output.get("prdy_ctrt", 0)),
            volume=int(output.get("acml_vol", 0)),
            high_price=int(output.get("stck_hgpr", 0)),
            low_price=int(output.get("stck_lwpr", 0)),
            open_price=int(output.get("stck_oprc", 0))
        )
        return price_info

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stock/{stock_code}/detail", response_model=StockDetailInfo)
async def get_stock_detail(
    stock_code: str,
    period: str = Query(default="D", description="차트 기간 (D:일봉, W:주봉, M:월봉)"),
    client: KISAPIClient = Depends(get_kis_client),
    predictor: PredictionService = Depends(get_prediction_service)
):
    """종목 상세 정보 조회 (기본정보 + 차트 + 예측)"""
    try:
        # 1. 현재가 정보 조회
        price_result = client.get_stock_price(stock_code)
        if price_result.get("rt_cd") != "0":
            raise HTTPException(
                status_code=400,
                detail=f"현재가 조회 실패: {price_result.get('msg1', 'Unknown error')}"
            )

        price_output = price_result.get("output", {})
        stock_name = price_output.get("hts_kor_isnm", "")

        # 디버깅: 주식명 로그
        logger.info(f"Stock detail for {stock_code}: stock_name='{stock_name}', price_output keys={list(price_output.keys())}")

        # 기본 정보 구성
        basic_info = StockBasicInfo(
            stock_code=stock_code,
            stock_name=stock_name,
            market=price_output.get("bstp_kor_isnm", "KOSPI"),
            current_price=int(price_output.get("stck_prpr", 0)),
            change_price=int(price_output.get("prdy_vrss", 0)),
            change_rate=float(price_output.get("prdy_ctrt", 0)),
            volume=int(price_output.get("acml_vol", 0)),
            market_cap=int(price_output.get("stck_avls", 0)) if price_output.get("stck_avls") else None,
            per=float(price_output.get("per", 0)) if price_output.get("per") else None,
            pbr=float(price_output.get("pbr", 0)) if price_output.get("pbr") else None
        )

        # 2. 차트 데이터 조회
        chart_result = client.get_daily_chart(stock_code, period=period, count=100)
        chart_data = []

        if chart_result.get("rt_cd") == "0":
            output2 = chart_result.get("output2", [])
            for item in output2[:30]:  # 최근 30개 데이터
                chart_data.append(ChartData(
                    date=item.get("stck_bsop_date", ""),
                    open=int(item.get("stck_oprc", 0)),
                    high=int(item.get("stck_hgpr", 0)),
                    low=int(item.get("stck_lwpr", 0)),
                    close=int(item.get("stck_clpr", 0)),
                    volume=int(item.get("acml_vol", 0)),
                ))
                
            output1 = chart_result.get("output1", {})
            stock_name = output1.get("hts_kor_isnm", "")
            basic_info.stock_name = stock_name  # 업데이트된 주식명 반영

        # 3. 주가 예측
        prediction_data = None
        if chart_result.get("rt_cd") == "0":
            output2 = chart_result.get("output2", [])
            logger.info(f"Chart data for prediction - stock: {stock_code}, data_count: {len(output2)}")
            if output2:
                logger.info(f"First chart item keys: {list(output2[0].keys())}")
                logger.info(f"First chart item sample: stck_clpr={output2[0].get('stck_clpr')}, stck_oprc={output2[0].get('stck_oprc')}")
            pred_result = predictor.predict_price(stock_code, stock_name, output2)
            logger.info(f"Prediction result: current={pred_result.get('current_price')}, predicted={pred_result.get('predicted_price')}")
            prediction_data = PredictionResult(**pred_result)

        # 통합 응답
        detail_info = StockDetailInfo(
            basic_info=basic_info,
            chart_data=chart_data,
            prediction=prediction_data
        )

        return detail_info

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/buy", response_model=OrderResponse)
async def buy_stock(
    order: OrderRequest,
    client: KISAPIClient = Depends(get_kis_client)
):
    """주식 매수"""
    try:
        result = client.buy_stock(
            stock_code=order.stock_code,
            quantity=order.quantity,
            price=order.price,
            order_type=order.order_type
        )

        if result.get("rt_cd") != "0":
            raise HTTPException(
                status_code=400,
                detail=f"매수 주문 실패: {result.get('msg1', 'Unknown error')}"
            )

        output = result.get("output", {})

        response = OrderResponse(
            order_id=output.get("ODNO", ""),
            stock_code=order.stock_code,
            stock_name=None,
            order_type="buy",
            quantity=order.quantity,
            price=order.price,
            order_time=datetime.now(),
            status="접수",
            message=result.get("msg1", "매수 주문이 접수되었습니다")
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sell", response_model=OrderResponse)
async def sell_stock(
    order: OrderRequest,
    client: KISAPIClient = Depends(get_kis_client)
):
    """주식 매도"""
    try:
        result = client.sell_stock(
            stock_code=order.stock_code,
            quantity=order.quantity,
            price=order.price,
            order_type=order.order_type
        )

        if result.get("rt_cd") != "0":
            raise HTTPException(
                status_code=400,
                detail=f"매도 주문 실패: {result.get('msg1', 'Unknown error')}"
            )

        output = result.get("output", {})

        response = OrderResponse(
            order_id=output.get("ODNO", ""),
            stock_code=order.stock_code,
            stock_name=None,
            order_type="sell",
            quantity=order.quantity,
            price=order.price,
            order_time=datetime.now(),
            status="접수",
            message=result.get("msg1", "매도 주문이 접수되었습니다")
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orders", response_model=List[OrderHistory])
async def get_order_history(
    client: KISAPIClient = Depends(get_kis_client)
):
    """주문 내역 조회"""
    try:
        result = client.get_order_history()

        if result.get("rt_cd") != "0":
            raise HTTPException(
                status_code=400,
                detail=f"주문 내역 조회 실패: {result.get('msg1', 'Unknown error')}"
            )

        output = result.get("output", [])

        orders = []
        for item in output:
            order = OrderHistory(
                order_date=item.get("ord_dt", ""),
                order_id=item.get("odno", ""),
                stock_code=item.get("pdno", ""),
                stock_name=item.get("prdt_name", ""),
                order_type=item.get("sll_buy_dvsn_cd_name", ""),
                order_quantity=int(item.get("ord_qty", 0)),
                order_price=int(item.get("ord_unpr", 0)),
                executed_quantity=int(item.get("tot_ccld_qty", 0)),
                executed_price=int(item.get("avg_prvs", 0)),
                status=item.get("ord_gno_brno", "")
            )
            orders.append(order)

        return orders

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/market-cap-stocks", response_model=TopStocksResponse)
async def get_market_cap_stocks(
    limit: int = Query(default=10, ge=1, le=30, description="조회할 종목 수"),
    client: KISAPIClient = Depends(get_kis_client)
):
    """시가총액 상위 종목 조회 (Redis 캐시 10분)"""
    try:
        redis = get_redis_client()
        cache_key = f"top_market_cap_stocks:limit_{limit}"

        # KIS API 호출 (시가총액 순위)
        result = client.get_market_cap_ranking(top_n=limit)

        if result.get("rt_cd") != "0":
            raise HTTPException(
                status_code=400,
                detail=f"시가총액 순위 조회 실패: {result.get('msg1', 'Unknown error')}"
            )

        output = result.get("output", [])

        stocks = []
        for idx, item in enumerate(output[:limit], start=1):
            stock = TopStock(
                rank=idx,
                stock_code=item.get("mksc_shrn_iscd", ""),
                stock_name=item.get("hts_kor_isnm", ""),
                current_price=int(item.get("stck_prpr", 0)),
                change_rate=float(item.get("prdy_ctrt", 0)),
                market_cap=int(item.get("stck_avls", 0)) if item.get("stck_avls") else 0
            )
            stocks.append(stock)

        response = TopStocksResponse(stocks=stocks)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/top-stocks", response_model=TopStocksResponse)
async def get_top_stocks(
    limit: int = Query(default=10, ge=1, le=30, description="조회할 종목 수"),
    client: KISAPIClient = Depends(get_kis_client)
):
    """거래량 상위 종목 조회 (Redis 캐시 10분)"""
    try:
        redis = get_redis_client()
        cache_key = f"top_volume_stocks:limit_{limit}"

        # 1. Redis 캐시 확인
        if redis.is_available():
            cached_data = redis.get(cache_key)
            if cached_data:
                try:
                    cached_dict = json.loads(cached_data)
                    return TopStocksResponse(**cached_dict)
                except Exception:
                    pass

        # 2. KIS API 호출 (거래량 순위)
        result = client.get_volume_ranking()

        if result.get("rt_cd") != "0":
            raise HTTPException(
                status_code=400,
                detail=f"거래량 순위 조회 실패: {result.get('msg1', 'Unknown error')}"
            )

        output = result.get("output", [])

        stocks = []
        for idx, item in enumerate(output[:limit], start=1):
            stock = TopStock(
                rank=idx,
                stock_code=item.get("mksc_shrn_iscd", ""),
                stock_name=item.get("hts_kor_isnm", ""),
                current_price=int(item.get("stck_prpr", 0)),
                change_rate=float(item.get("prdy_ctrt", 0)),
                market_cap=int(item.get("data_rank", 0))  # 거래량으로 변경
            )
            stocks.append(stock)

        response = TopStocksResponse(stocks=stocks)

        # 3. Redis에 캐싱 (10분 = 600초)
        if redis.is_available():
            try:
                cache_data = json.dumps(response.model_dump())
                redis.set(cache_key, cache_data, expire=600)
            except Exception:
                pass

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
