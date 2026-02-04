"""ML 모델 관리 API 라우터"""
import json
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, update

from app.database import get_db
from app.models.ml_model import ModelTrainingHistory
from app.schemas.ml_model import (
    ModelTrainingHistoryResponse,
    ModelTrainingListResponse,
    ModelActivateRequest,
    ActiveModelResponse,
    ModelComparisonResponse
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ml-models", tags=["ml-models"])


def parse_json_field(value: Optional[str]) -> Optional[dict | list]:
    """JSON 문자열을 파싱"""
    if value is None:
        return None
    try:
        return json.loads(value)
    except:
        return None


@router.get("/history", response_model=ModelTrainingListResponse)
async def get_training_history(
    page: int = Query(default=1, ge=1, description="페이지 번호"),
    page_size: int = Query(default=10, ge=1, le=50, description="페이지당 개수"),
    db: AsyncSession = Depends(get_db)
):
    """모델 학습 이력 목록 조회"""
    try:
        offset = (page - 1) * page_size

        # 총 개수 조회
        count_query = select(func.count(ModelTrainingHistory.id))
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0

        # 목록 조회 (최신순)
        query = (
            select(ModelTrainingHistory)
            .order_by(ModelTrainingHistory.trained_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        result = await db.execute(query)
        models = result.scalars().all()

        return ModelTrainingListResponse(
            models=[
                ModelTrainingHistoryResponse(
                    id=m.id,
                    model_name=m.model_name,
                    model_type=m.model_type,
                    model_version=m.model_version,
                    hyperparameters=parse_json_field(m.hyperparameters),
                    feature_columns=parse_json_field(m.feature_columns),
                    scaler_type=m.scaler_type or "MinMaxScaler",
                    train_samples=m.train_samples,
                    test_samples=m.test_samples,
                    total_samples=m.total_samples,
                    train_score=m.train_score,
                    test_score=m.test_score,
                    mae=m.mae,
                    rmse=m.rmse,
                    mape=m.mape,
                    is_active=m.is_active,
                    is_production=m.is_production,
                    trained_by=m.trained_by or "batch",
                    training_duration_sec=m.training_duration_sec,
                    notes=m.notes,
                    trained_at=m.trained_at,
                    created_at=m.created_at
                )
                for m in models
            ],
            total=total,
            page=page,
            page_size=page_size
        )

    except Exception as e:
        logger.error(f"Error fetching training history: {e}")
        raise HTTPException(status_code=500, detail="학습 이력 조회 중 오류가 발생했습니다")


@router.get("/active", response_model=Optional[ActiveModelResponse])
async def get_active_model(db: AsyncSession = Depends(get_db)):
    """현재 활성화된 모델 조회"""
    try:
        query = (
            select(ModelTrainingHistory)
            .where(ModelTrainingHistory.is_active == True)
            .order_by(ModelTrainingHistory.trained_at.desc())
            .limit(1)
        )
        result = await db.execute(query)
        model = result.scalar_one_or_none()

        if not model:
            return None

        return ActiveModelResponse(
            id=model.id,
            model_name=model.model_name,
            model_type=model.model_type,
            model_version=model.model_version,
            test_score=model.test_score,
            trained_at=model.trained_at,
            hyperparameters=parse_json_field(model.hyperparameters),
            feature_columns=parse_json_field(model.feature_columns)
        )

    except Exception as e:
        logger.error(f"Error fetching active model: {e}")
        raise HTTPException(status_code=500, detail="활성 모델 조회 중 오류가 발생했습니다")


@router.get("/{model_id}", response_model=ModelTrainingHistoryResponse)
async def get_model_detail(
    model_id: int,
    db: AsyncSession = Depends(get_db)
):
    """특정 모델 상세 조회"""
    try:
        query = select(ModelTrainingHistory).where(ModelTrainingHistory.id == model_id)
        result = await db.execute(query)
        model = result.scalar_one_or_none()

        if not model:
            raise HTTPException(status_code=404, detail="모델을 찾을 수 없습니다")

        return ModelTrainingHistoryResponse(
            id=model.id,
            model_name=model.model_name,
            model_type=model.model_type,
            model_version=model.model_version,
            hyperparameters=parse_json_field(model.hyperparameters),
            feature_columns=parse_json_field(model.feature_columns),
            scaler_type=model.scaler_type or "MinMaxScaler",
            train_samples=model.train_samples,
            test_samples=model.test_samples,
            total_samples=model.total_samples,
            train_score=model.train_score,
            test_score=model.test_score,
            mae=model.mae,
            rmse=model.rmse,
            mape=model.mape,
            is_active=model.is_active,
            is_production=model.is_production,
            trained_by=model.trained_by or "batch",
            training_duration_sec=model.training_duration_sec,
            notes=model.notes,
            trained_at=model.trained_at,
            created_at=model.created_at
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching model detail: {e}")
        raise HTTPException(status_code=500, detail="모델 상세 조회 중 오류가 발생했습니다")


@router.post("/activate")
async def activate_model(
    request: ModelActivateRequest,
    db: AsyncSession = Depends(get_db)
):
    """특정 모델 활성화"""
    try:
        # 모델 존재 확인
        query = select(ModelTrainingHistory).where(ModelTrainingHistory.id == request.model_id)
        result = await db.execute(query)
        model = result.scalar_one_or_none()

        if not model:
            raise HTTPException(status_code=404, detail="모델을 찾을 수 없습니다")

        # 기존 활성 모델 비활성화
        await db.execute(
            update(ModelTrainingHistory)
            .where(ModelTrainingHistory.is_active == True)
            .values(is_active=False)
        )

        # 새 모델 활성화
        model.is_active = True
        await db.commit()

        logger.info(f"Model activated: ID={model.id}, Version={model.model_version}")

        return {
            "success": True,
            "message": f"모델 {model.model_version}이 활성화되었습니다",
            "model_id": model.id,
            "model_version": model.model_version
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error activating model: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="모델 활성화 중 오류가 발생했습니다")


@router.get("/compare/latest", response_model=ModelComparisonResponse)
async def compare_with_latest(db: AsyncSession = Depends(get_db)):
    """현재 활성 모델과 최신 모델 비교"""
    try:
        # 현재 활성 모델
        active_query = (
            select(ModelTrainingHistory)
            .where(ModelTrainingHistory.is_active == True)
            .limit(1)
        )
        active_result = await db.execute(active_query)
        active_model = active_result.scalar_one_or_none()

        # 최신 모델 (활성화 여부 무관)
        latest_query = (
            select(ModelTrainingHistory)
            .order_by(ModelTrainingHistory.trained_at.desc())
            .limit(1)
        )
        latest_result = await db.execute(latest_query)
        latest_model = latest_result.scalar_one_or_none()

        current = None
        if active_model:
            current = ActiveModelResponse(
                id=active_model.id,
                model_name=active_model.model_name,
                model_type=active_model.model_type,
                model_version=active_model.model_version,
                test_score=active_model.test_score,
                trained_at=active_model.trained_at,
                hyperparameters=parse_json_field(active_model.hyperparameters),
                feature_columns=parse_json_field(active_model.feature_columns)
            )

        latest = None
        if latest_model:
            latest = ModelTrainingHistoryResponse(
                id=latest_model.id,
                model_name=latest_model.model_name,
                model_type=latest_model.model_type,
                model_version=latest_model.model_version,
                hyperparameters=parse_json_field(latest_model.hyperparameters),
                feature_columns=parse_json_field(latest_model.feature_columns),
                scaler_type=latest_model.scaler_type or "MinMaxScaler",
                train_samples=latest_model.train_samples,
                test_samples=latest_model.test_samples,
                total_samples=latest_model.total_samples,
                train_score=latest_model.train_score,
                test_score=latest_model.test_score,
                mae=latest_model.mae,
                rmse=latest_model.rmse,
                mape=latest_model.mape,
                is_active=latest_model.is_active,
                is_production=latest_model.is_production,
                trained_by=latest_model.trained_by or "batch",
                training_duration_sec=latest_model.training_duration_sec,
                notes=latest_model.notes,
                trained_at=latest_model.trained_at,
                created_at=latest_model.created_at
            )

        improvement = None
        if active_model and latest_model:
            improvement = latest_model.test_score - active_model.test_score

        return ModelComparisonResponse(
            current_model=current,
            latest_model=latest,
            improvement=improvement
        )

    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        raise HTTPException(status_code=500, detail="모델 비교 중 오류가 발생했습니다")
