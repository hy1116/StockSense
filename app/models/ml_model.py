"""ML 모델 학습 이력 모델"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, LargeBinary
from sqlalchemy.sql import func

from app.database import Base


class ModelTrainingHistory(Base):
    """모델 학습 이력"""
    __tablename__ = "model_training_history"

    id = Column(Integer, primary_key=True, index=True)

    # 모델 정보
    model_name = Column(String(100), nullable=False)  # 모델 파일명
    model_type = Column(String(50), nullable=False)  # GradientBoosting, RandomForest 등
    model_version = Column(String(20), nullable=False)  # v1, v2, ...

    # 하이퍼파라미터 (JSON)
    hyperparameters = Column(Text, nullable=True)  # {"n_estimators": 100, ...}

    # 피처 정보
    feature_columns = Column(Text, nullable=True)  # JSON array
    scaler_type = Column(String(50), default="MinMaxScaler")

    # 학습 데이터 정보
    train_samples = Column(Integer, nullable=False)
    test_samples = Column(Integer, nullable=False)
    total_samples = Column(Integer, nullable=False)

    # 성능 지표
    train_score = Column(Float, nullable=False)  # R² score
    test_score = Column(Float, nullable=False)   # R² score
    mae = Column(Float, nullable=True)  # Mean Absolute Error
    rmse = Column(Float, nullable=True)  # Root Mean Square Error
    mape = Column(Float, nullable=True)  # Mean Absolute Percentage Error

    # 모델 바이너리 (pkl 파일을 직접 저장)
    model_binary = Column(LargeBinary, nullable=True)
    scaler_binary = Column(LargeBinary, nullable=True)

    # 상태
    is_active = Column(Boolean, default=False)  # 현재 사용 중인 모델 여부
    is_production = Column(Boolean, default=False)  # 프로덕션 배포 여부

    # 메타 정보
    trained_by = Column(String(50), default="batch")  # batch, manual
    training_duration_sec = Column(Float, nullable=True)  # 학습 소요 시간
    notes = Column(Text, nullable=True)  # 메모

    # 타임스탬프
    trained_at = Column(DateTime(timezone=True), server_default=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<ModelTrainingHistory(id={self.id}, model_name={self.model_name}, test_score={self.test_score:.4f})>"
