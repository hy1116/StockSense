"""포트폴리오 API 엔드포인트"""
import asyncio
from functools import partial
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_
from typing import List
from datetime import datetime
import logging


async def run_sync(func, *args, **kwargs):
    """동기 함수를 스레드풀에서 실행 (이벤트루프 블로킹 방지)"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(func, *args, **kwargs))

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
    ChartCandle,
    MinuteChartData,
    PredictionResult,
    TopStock,
    TopStocksResponse,
    StockSearchItem,
    StockSearchResponse
)
from app.services.kis_api import get_kis_client, KISAPIClient
from app.services.prediction import get_prediction_service, PredictionService
from app.services.redis_client import get_redis_client
from app.api.prediction import _save_prediction_to_db
from app.services.naver_finance import get_naver_finance_client
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
    """종목명 또는 코드로 검색 (DB 기반, DB 결과 없으면 네이버 자동완성 폴백)"""
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

        if stocks:
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
        else:
            # DB에 없으면 네이버 자동완성 API 폴백
            logger.info(f"DB search empty for '{search_term}', falling back to Naver autocomplete")
            naver = get_naver_finance_client()
            naver_results = await naver.search_stocks(search_term, limit)
            results = [StockSearchItem(**item) for item in naver_results]

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
        # 2. KIS API 호출 (스레드풀에서 실행)
        result = await run_sync(client.get_balance)

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
        result = await run_sync(client.get_stock_price, stock_code)

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


@router.get("/stock/{stock_code}/prediction", response_model=PredictionResult)
async def get_stock_prediction_fresh(
    stock_code: str,
    client: KISAPIClient = Depends(get_kis_client),
    predictor: PredictionService = Depends(get_prediction_service)
):
    """종목 AI 예측 (캐시 없음 - 항상 최신 예측 반환)"""
    try:
        price_result = await run_sync(client.get_stock_price, stock_code)
        if price_result.get("rt_cd") != "0":
            raise HTTPException(status_code=400, detail="현재가 조회 실패")

        stock_name = price_result.get("output", {}).get("hts_kor_isnm", "")

        chart_result = await run_sync(client.get_daily_chart, stock_code, period="D", count=100)
        if chart_result.get("rt_cd") != "0":
            raise HTTPException(status_code=400, detail="차트 데이터 조회 실패")

        output2 = chart_result.get("output2", [])
        pred_result = await run_sync(predictor.predict_price, stock_code, stock_name, output2)
        # 예측 결과 DB 저장 (적중률 추적용)
        _save_prediction_to_db(pred_result)
        return PredictionResult(**pred_result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"예측 조회 실패 {stock_code}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stock/{stock_code}/opinion")
async def get_stock_ai_opinion(
    stock_code: str,
    client: KISAPIClient = Depends(get_kis_client),
    predictor: PredictionService = Depends(get_prediction_service),
    db: AsyncSession = Depends(get_db)
):
    """종목 AI 자연어 투자의견 (Gemini 생성, 종목정보+뉴스 포함, Redis 1시간 캐시)"""
    try:
        from app.services.stock_opinion import generate_stock_opinion
        from app.models.stock_news import StockNews
        from datetime import timedelta

        # Redis 캐시 확인 (1시간)
        redis = get_redis_client()
        cache_key = f"ai_opinion:{stock_code}"
        if redis.is_available():
            cached = redis.get(cache_key)
            if cached:
                logger.info(f"AI 의견 캐시 HIT: {stock_code}")
                return json.loads(cached)

        price_result = await run_sync(client.get_stock_price, stock_code)
        stock_name = price_result.get("output", {}).get("hts_kor_isnm", "")

        chart_result = await run_sync(client.get_daily_chart, stock_code, period="D", count=100)
        output2 = chart_result.get("output2", [])

        pred_result = await run_sync(predictor.predict_price, stock_code, stock_name, output2)

        # 최근 7일 뉴스 최대 10건 조회
        start_date = datetime.now() - timedelta(days=7)
        news_query = (
            select(StockNews)
            .where(StockNews.stock_code == stock_code)
            .where(StockNews.published_at >= start_date)
            .order_by(StockNews.published_at.desc())
            .limit(10)
        )
        news_result = await db.execute(news_query)
        news_list = [
            {"title": n.title, "summary": n.summary, "sentiment": n.sentiment}
            for n in news_result.scalars().all()
        ]

        opinion = await generate_stock_opinion(stock_name, pred_result, news_list)

        result = {"opinion": opinion, "stock_code": stock_code, "stock_name": stock_name}

        # 성공한 경우에만 캐시 저장 (1시간)
        if opinion and redis.is_available():
            redis.set(cache_key, json.dumps(result, ensure_ascii=False), expire=3600)

        return result
    except Exception as e:
        logger.error(f"AI 의견 생성 실패 {stock_code}: {e}", exc_info=True)
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
        redis = get_redis_client()
        cache_key = f"stock_detail:{stock_code}:period_{period}"

        # Redis 캐시 확인 (10분)
        if redis.is_available():
            cached_data = redis.get(cache_key)
            if cached_data:
                try:
                    cached_dict = json.loads(cached_data)
                    logger.info(f"✅ [CACHE HIT] Stock detail for {stock_code} (period={period})")
                    return StockDetailInfo(**cached_dict)
                except Exception as e:
                    logger.warning(f"⚠️ [CACHE PARSE ERROR] Failed to parse cached stock detail: {e}")
                    pass
            else:
                logger.info(f"❌ [CACHE MISS] Stock detail for {stock_code} (period={period}) - Fetching from API")
        else:
            logger.warning(f"⚠️ [REDIS UNAVAILABLE] Fetching stock detail from API for {stock_code}")

        # 1. 현재가 정보 조회
        price_result = await run_sync(client.get_stock_price, stock_code)
        if price_result.get("rt_cd") != "0":
            raise HTTPException(
                status_code=400,
                detail=f"현재가 조회 실패: {price_result.get('msg1', 'Unknown error')}"
            )

        price_output = price_result.get("output", {})
        stock_name = price_output.get("hts_kor_isnm", "")

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
            pbr=float(price_output.get("pbr", 0)) if price_output.get("pbr") else None,
            hts_avls=float(price_output.get("hts_avls", 0)) if price_output.get("hts_avls") else None
        )
        
        # 2. 차트 데이터 조회 (항상 일봉, 표시 범위는 프론트에서 제어)
        chart_result = await run_sync(client.get_daily_chart, stock_code, period="D", count=100)
        chart_data = []

        if chart_result.get("rt_cd") == "0":
            output2 = chart_result.get("output2", [])
            for item in output2:  # 전체 차트 데이터
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
            pred_result = await run_sync(predictor.predict_price, stock_code, stock_name, output2)
            logger.info(f"Prediction result: current={pred_result.get('current_price')}, predicted={pred_result.get('predicted_price')}")
            prediction_data = PredictionResult(**pred_result)

        # 통합 응답
        detail_info = StockDetailInfo(
            basic_info=basic_info,
            chart_data=chart_data,
            prediction=prediction_data
        )

        # Redis 캐싱 (10분 = 600초)
        if redis.is_available():
            try:
                cache_data = json.dumps(detail_info.model_dump())
                redis.set(cache_key, cache_data, expire=600)
                logger.info(f"💾 [CACHE SAVED] Stock detail for {stock_code} (period={period}) cached for 10 min")
            except Exception as e:
                logger.error(f"❌ [CACHE SAVE ERROR] Failed to cache stock detail: {e}")
                pass

        logger.info(f"✅ [API SUCCESS] Returning stock detail for {stock_code} with {len(chart_data)} candles")
        return detail_info

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stock/{stock_code}/intraday", response_model=list[MinuteChartData])
async def get_stock_intraday(
    stock_code: str,
    interval: int = Query(default=1, ge=1, le=60, description="분봉 간격 (분 단위: 1/5/10/30/60)"),
    client: KISAPIClient = Depends(get_kis_client)
):
    """종목 당일 분봉 데이터 조회 (간격 조절 가능)"""
    try:
        redis = get_redis_client()
        cache_key = f"intraday:{stock_code}:interval_{interval}"

        # Redis 캐시 확인 (10분)
        if redis.is_available():
            cached_data = redis.get(cache_key)
            if cached_data:
                try:
                    cached_list = json.loads(cached_data)
                    logger.info(f"✅ [CACHE HIT] Intraday data for {stock_code} (interval={interval}) - {len(cached_list)} candles")
                    return [MinuteChartData(**item) for item in cached_list]
                except Exception as e:
                    logger.warning(f"⚠️ [CACHE PARSE ERROR] Failed to parse cached intraday data: {e}")
                    pass
            else:
                logger.info(f"❌ [CACHE MISS] Intraday data for {stock_code} (interval={interval}) - Fetching from API")
        else:
            logger.warning(f"⚠️ [REDIS UNAVAILABLE] Fetching intraday data from API for {stock_code}")

        result = await run_sync(client.get_minute_chart, stock_code, interval=interval)
        if result.get("rt_cd") != "0":
            raise HTTPException(
                status_code=400,
                detail=f"분봉 조회 실패: {result.get('msg1', 'Unknown error')}"
            )

        output2 = result.get("output2", [])
        chart_data = []
        seen_times = set()

        for item in output2:
            raw_time = item.get("stck_cntg_hour", "")
            if not raw_time or len(raw_time) < 4:
                continue

            time_str = f"{raw_time[:2]}:{raw_time[2:4]}"
            if time_str in seen_times:
                continue
            seen_times.add(time_str)

            chart_data.append(MinuteChartData(
                time=time_str,
                open=int(item.get("stck_oprc", 0)),
                high=int(item.get("stck_hgpr", 0)),
                low=int(item.get("stck_lwpr", 0)),
                close=int(item.get("stck_prpr", 0)),
                volume=int(item.get("cntg_vol", 0)),
            ))

        # 시간순 정렬 (API는 역순)
        chart_data.sort(key=lambda x: x.time)

        # Redis 캐싱 (10분 = 600초)
        if redis.is_available() and chart_data:
            try:
                cache_data = json.dumps([item.model_dump() for item in chart_data])
                redis.set(cache_key, cache_data, expire=600)
                logger.info(f"💾 [CACHE SAVED] Intraday data for {stock_code} (interval={interval}) - {len(chart_data)} candles cached for 10 min")
            except Exception as e:
                logger.error(f"❌ [CACHE SAVE ERROR] Failed to cache intraday data: {e}")
                pass

        logger.info(f"✅ [API SUCCESS] Returning {len(chart_data)} intraday candles for {stock_code}")
        return chart_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"분봉 조회 실패: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stock/{stock_code}/chart", response_model=list[ChartCandle])
async def get_stock_chart(
    stock_code: str,
    period: str = Query(default="1D", description="1D/1W/1M/3M/1Y"),
    client: KISAPIClient = Depends(get_kis_client),
):
    """차트 데이터 조회 — 1D: KIS 10분봉 / 1W·1M·3M·1Y: Naver fchart 일봉·주봉"""
    if period not in ("1D", "1W", "1M", "3M", "1Y"):
        raise HTTPException(status_code=400, detail="Invalid period")
    try:
        redis = get_redis_client()
        cache_key = f"chart:{stock_code}:{period}"
        ttl = {"1D": 300, "1W": 600, "1M": 3600, "3M": 3600, "1Y": 3600}[period]

        if redis.is_available():
            cached = redis.get(cache_key)
            if cached:
                try:
                    return [ChartCandle(**i) for i in json.loads(cached)]
                except Exception:
                    pass

        if period == "1D":
            # KIS API 10분봉 (당일)
            from datetime import date
            from zoneinfo import ZoneInfo
            kst = ZoneInfo("Asia/Seoul")
            now_kst = datetime.now(kst)
            today = now_kst.strftime("%Y%m%d")
            # 현재 시간(KST) 기준, 장 마감(15:30) 이후면 15:30으로 제한
            current_hhmm = now_kst.strftime("%H%M")
            max_hhmm = min(current_hhmm, "1530")

            raw = await run_sync(client.get_minute_chart, stock_code, interval=10)
            candles = []
            if raw.get("rt_cd") == "0":
                seen = set()
                for item in raw.get("output2", []):
                    t = item.get("stck_cntg_hour", "")
                    if not t or len(t) < 4:
                        continue
                    hhmm = t[:4]
                    # 장 시간(09:00~현재시간, 최대 15:30) 외 캔들 제거
                    if hhmm < "0900" or hhmm > max_hhmm:
                        continue
                    if hhmm in seen:
                        continue
                    seen.add(hhmm)
                    o = int(item.get("stck_oprc", 0))
                    c = int(item.get("stck_prpr", 0))
                    h = int(item.get("stck_hgpr", 0))
                    l = int(item.get("stck_lwpr", 0))
                    # OHLCV 기본 유효성 검사 (0 또는 비정상 OHLCV 제거)
                    if o <= 0 or c <= 0 or h <= 0 or l <= 0:
                        continue
                    if h < max(o, c) or l > min(o, c):
                        continue
                    candles.append(ChartCandle(
                        dt=today + hhmm,
                        open=o,
                        high=h,
                        low=l,
                        close=c,
                        volume=int(item.get("cntg_vol", 0)),
                    ))
                candles.sort(key=lambda x: x.dt)
            result = candles
        else:
            # Naver fchart (1W/1M/3M/1Y)
            naver = get_naver_finance_client()
            data = await naver.get_chart(stock_code, period)
            result = [ChartCandle(**i) for i in data]

        if redis.is_available() and result:
            redis.set(cache_key, json.dumps([i.model_dump() for i in result]), expire=ttl)

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"차트 조회 실패 {stock_code}/{period}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/buy", response_model=OrderResponse)
async def buy_stock(
    order: OrderRequest,
    client: KISAPIClient = Depends(get_kis_client)
):
    """주식 매수"""
    try:
        result = await run_sync(
            client.buy_stock,
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
        result = await run_sync(
            client.sell_stock,
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
        result = await run_sync(client.get_order_history)

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

        # 1. Redis 캐시 확인
        if redis.is_available():
            cached_data = redis.get(cache_key)
            if cached_data:
                try:
                    cached_dict = json.loads(cached_data)
                    return TopStocksResponse(**cached_dict)
                except Exception:
                    pass

        # 2. KIS API 호출 (시가총액 순위)
        result = await run_sync(client.get_market_cap_ranking, top_n=limit)

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
        result = await run_sync(client.get_volume_ranking)

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

@router.get("/fluctuation-stocks", response_model=TopStocksResponse)
async def get_fluctuation_stocks(
    limit: int = Query(default=10, ge=1, le=30, description="조회할 종목 수"),
    sort: int = 0,
    client: KISAPIClient = Depends(get_kis_client)
):
    """등락률 상위 종목 조회 (Redis 캐시 10분)"""
    try:
        redis = get_redis_client()
        cache_key = f"top_fluctuation_stocks:limit_{limit}:sort_{sort}"

        # 1. Redis 캐시 확인
        if redis.is_available():
            cached_data = redis.get(cache_key)
            if cached_data:
                try:
                    cached_dict = json.loads(cached_data)
                    return TopStocksResponse(**cached_dict)
                except Exception:
                    pass

        # 2. KIS API 호출 (등락률 순위)
        result = await run_sync(client.get_fluctuation_ranking, top_n=limit, sort_code=sort)

        if result.get("rt_cd") != "0":
            raise HTTPException(
                status_code=400,
                detail=f"등락률 순위 조회 실패: {result.get('msg1', 'Unknown error')}"
            )

        output = result.get("output", [])

        stocks = []
        for idx, item in enumerate(output[:limit], start=1):
            stock = TopStock(
                rank=idx,
                stock_code=item.get("stck_shrn_iscd") or item.get("mksc_shrn_iscd", ""),
                stock_name=item.get("hts_kor_isnm", ""),
                current_price=int(item.get("stck_prpr", 0)),
                change_rate=float(item.get("prdy_ctrt", 0)),
                market_cap=int(item.get("stck_avls", 0)) if item.get("stck_avls") else 0
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


