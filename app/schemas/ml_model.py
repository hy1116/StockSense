"""ML 모델 관련 Pydantic 스키마"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class ModelTrainingHistoryResponse(BaseModel):
    """모델 학습 이력 응답"""
    id: int = Field(..., description="학습 이력 ID")
    model_name: str = Field(..., description="모델 파일명")
    model_type: str = Field(..., description="모델 타입")
    model_version: str = Field(..., description="모델 버전")

    # 하이퍼파라미터
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="하이퍼파라미터")

    # 피처 정보
    feature_columns: Optional[List[str]] = Field(None, description="피처 컬럼 목록")
    scaler_type: str = Field(..., description="스케일러 타입")

    # 학습 데이터 정보
    train_samples: int = Field(..., description="학습 샘플 수")
    test_samples: int = Field(..., description="테스트 샘플 수")
    total_samples: int = Field(..., description="전체 샘플 수")

    # 성능 지표
    train_score: float = Field(..., description="학습 R² 점수")
    test_score: float = Field(..., description="테스트 R² 점수")
    mae: Optional[float] = Field(None, description="MAE")
    rmse: Optional[float] = Field(None, description="RMSE")
    mape: Optional[float] = Field(None, description="MAPE (%)")

    # 상태
    is_active: bool = Field(..., description="현재 활성화 여부")
    is_production: bool = Field(..., description="프로덕션 배포 여부")

    # 메타 정보
    trained_by: str = Field(..., description="학습 주체 (batch/manual)")
    training_duration_sec: Optional[float] = Field(None, description="학습 소요 시간 (초)")
    notes: Optional[str] = Field(None, description="메모")

    trained_at: datetime = Field(..., description="학습 시간")
    created_at: datetime = Field(..., description="생성 시간")

    class Config:
        from_attributes = True


class ModelTrainingListResponse(BaseModel):
    """모델 학습 이력 목록 응답"""
    models: List[ModelTrainingHistoryResponse] = Field(default=[], description="모델 목록")
    total: int = Field(..., description="전체 개수")
    page: int = Field(..., description="현재 페이지")
    page_size: int = Field(..., description="페이지당 개수")


class ModelActivateRequest(BaseModel):
    """모델 활성화 요청"""
    model_id: int = Field(..., description="활성화할 모델 ID")


class ActiveModelResponse(BaseModel):
    """현재 활성 모델 정보"""
    id: int = Field(..., description="모델 ID")
    model_name: str = Field(..., description="모델 파일명")
    model_type: str = Field(..., description="모델 타입")
    model_version: str = Field(..., description="모델 버전")
    test_score: float = Field(..., description="테스트 R² 점수")
    trained_at: datetime = Field(..., description="학습 시간")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="하이퍼파라미터")
    feature_columns: Optional[List[str]] = Field(None, description="피처 컬럼 목록")


class ModelComparisonResponse(BaseModel):
    """모델 비교 응답"""
    current_model: Optional[ActiveModelResponse] = Field(None, description="현재 활성 모델")
    latest_model: Optional[ModelTrainingHistoryResponse] = Field(None, description="최신 학습 모델")
    improvement: Optional[float] = Field(None, description="성능 개선율")
