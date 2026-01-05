"""포트폴리오 API 엔드포인트"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List
from datetime import datetime

from app.schemas.portfolio import (
    PortfolioSummary,
    StockHolding,
    OrderRequest,
    OrderResponse,
    OrderHistory,
    StockPriceInfo
)
from app.services.kis_api import get_kis_client, KISAPIClient

router = APIRouter(prefix="/api/portfolio", tags=["Portfolio"])


@router.get("/balance", response_model=PortfolioSummary)
async def get_portfolio_balance(
    client: KISAPIClient = Depends(get_kis_client)
):
    """계좌 잔고 및 포트폴리오 조회"""
    try:
        result = client.get_balance()

        if result.get("rt_cd") != "0":
            raise HTTPException(
                status_code=400,
                detail=f"잔고 조회 실패: {result.get('msg1', 'Unknown error')}"
            )

        output1 = result.get("output1", [])
        output2 = result.get("output2", [{}])[0]

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

        summary = PortfolioSummary(
            total_asset=float(output2.get("tot_evlu_amt", 0)),
            cash=float(output2.get("dnca_tot_amt", 0)),
            stock_eval_amount=float(output2.get("scts_evlu_amt", 0)),
            total_profit_loss=float(output2.get("evlu_pfls_smtl_amt", 0)),
            total_profit_rate=float(output2.get("tot_evlu_pfls_rt", 0)),
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
