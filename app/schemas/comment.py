"""댓글 관련 Pydantic 스키마"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class CommentCreate(BaseModel):
    """댓글 생성 요청"""
    content: str = Field(..., description="댓글 내용", min_length=1, max_length=1000)


class CommentUpdate(BaseModel):
    """댓글 수정 요청"""
    content: str = Field(..., description="댓글 내용", min_length=1, max_length=1000)


class CommentResponse(BaseModel):
    """댓글 응답"""
    id: int = Field(..., description="댓글 ID")
    stock_code: str = Field(..., description="종목코드")
    user_id: int = Field(..., description="작성자 ID")
    username: str = Field(..., description="작성자 이름")
    content: str = Field(..., description="댓글 내용")
    created_at: datetime = Field(..., description="작성일시")
    updated_at: Optional[datetime] = Field(None, description="수정일시")
    is_mine: bool = Field(default=False, description="본인 작성 여부")

    class Config:
        from_attributes = True


class CommentListResponse(BaseModel):
    """댓글 목록 응답"""
    comments: List[CommentResponse] = Field(default=[], description="댓글 리스트")
    total: int = Field(..., description="총 댓글 수")
    page: int = Field(..., description="현재 페이지")
    page_size: int = Field(..., description="페이지당 댓글 수")
    has_more: bool = Field(..., description="더 많은 댓글 존재 여부")
