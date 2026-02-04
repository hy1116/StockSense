"""댓글 API 라우터"""
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Query, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete

from app.database import get_db
from app.models.comment import Comment
from app.models.user import User
from app.schemas.comment import (
    CommentCreate, CommentUpdate, CommentResponse, CommentListResponse
)
from app.services.auth import verify_token, get_session_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/comments", tags=["comments"])
security = HTTPBearer(auto_error=False)


async def get_current_user_id(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[int]:
    """현재 인증된 사용자 ID 조회"""
    token = None

    if credentials:
        token = credentials.credentials
    if not token:
        token = request.cookies.get("access_token")

    if not token:
        return None

    payload = verify_token(token)
    if not payload:
        return None

    session_id = payload.get("session_id")
    if not session_id:
        return None

    session_manager = get_session_manager()
    session = session_manager.get_session(session_id)

    if not session:
        return None

    return session.get("user_id")


async def require_auth_user_id(
    user_id: Optional[int] = Depends(get_current_user_id)
) -> int:
    """인증 필수 - 사용자 ID 반환"""
    if not user_id:
        raise HTTPException(status_code=401, detail="인증이 필요합니다")
    return user_id


@router.get("/{stock_code}", response_model=CommentListResponse)
async def get_comments(
    stock_code: str,
    page: int = Query(default=1, ge=1, description="페이지 번호"),
    page_size: int = Query(default=20, ge=1, le=100, description="페이지당 댓글 수"),
    db: AsyncSession = Depends(get_db),
    current_user_id: Optional[int] = Depends(get_current_user_id)
):
    """종목별 댓글 목록 조회"""
    try:
        offset = (page - 1) * page_size

        # 총 댓글 수 조회
        count_query = select(func.count(Comment.id)).where(Comment.stock_code == stock_code)
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0

        # 댓글 목록 조회 (최신순)
        query = (
            select(Comment, User.nickname)
            .join(User, Comment.user_id == User.id)
            .where(Comment.stock_code == stock_code)
            .order_by(Comment.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        result = await db.execute(query)
        rows = result.all()

        comments = []
        for comment, nickname in rows:
            comments.append(CommentResponse(
                id=comment.id,
                stock_code=comment.stock_code,
                user_id=comment.user_id,
                username=nickname,  # 닉네임을 username 필드에 저장
                content=comment.content,
                created_at=comment.created_at,
                updated_at=comment.updated_at,
                is_mine=current_user_id == comment.user_id if current_user_id else False
            ))

        has_more = (page * page_size) < total

        return CommentListResponse(
            comments=comments,
            total=total,
            page=page,
            page_size=page_size,
            has_more=has_more
        )

    except Exception as e:
        logger.error(f"Error fetching comments: {e}")
        raise HTTPException(status_code=500, detail="댓글 조회 중 오류가 발생했습니다")


@router.post("/{stock_code}", response_model=CommentResponse)
async def create_comment(
    stock_code: str,
    comment_data: CommentCreate,
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(require_auth_user_id)
):
    """댓글 작성"""
    try:
        # 사용자 정보 조회
        user_result = await db.execute(
            select(User).where(User.id == current_user_id)
        )
        user = user_result.scalar_one_or_none()

        if not user:
            raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다")

        # 댓글 생성
        new_comment = Comment(
            stock_code=stock_code,
            user_id=current_user_id,
            content=comment_data.content
        )

        db.add(new_comment)
        await db.commit()
        await db.refresh(new_comment)

        logger.info(f"Comment created: stock={stock_code}, user={user.nickname}")

        return CommentResponse(
            id=new_comment.id,
            stock_code=new_comment.stock_code,
            user_id=new_comment.user_id,
            username=user.nickname,  # 닉네임 사용
            content=new_comment.content,
            created_at=new_comment.created_at,
            updated_at=new_comment.updated_at,
            is_mine=True
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating comment: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="댓글 작성 중 오류가 발생했습니다")


@router.put("/{comment_id}", response_model=CommentResponse)
async def update_comment(
    comment_id: int,
    comment_data: CommentUpdate,
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(require_auth_user_id)
):
    """댓글 수정"""
    try:
        # 댓글 조회
        query = (
            select(Comment, User.nickname)
            .join(User, Comment.user_id == User.id)
            .where(Comment.id == comment_id)
        )
        result = await db.execute(query)
        row = result.first()

        if not row:
            raise HTTPException(status_code=404, detail="댓글을 찾을 수 없습니다")

        comment, nickname = row

        # 작성자 확인
        if comment.user_id != current_user_id:
            raise HTTPException(status_code=403, detail="본인의 댓글만 수정할 수 있습니다")

        # 댓글 수정
        comment.content = comment_data.content
        await db.commit()
        await db.refresh(comment)

        logger.info(f"Comment updated: id={comment_id}")

        return CommentResponse(
            id=comment.id,
            stock_code=comment.stock_code,
            user_id=comment.user_id,
            username=nickname,  # 닉네임 사용
            content=comment.content,
            created_at=comment.created_at,
            updated_at=comment.updated_at,
            is_mine=True
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating comment: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="댓글 수정 중 오류가 발생했습니다")


@router.delete("/{comment_id}")
async def delete_comment(
    comment_id: int,
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(require_auth_user_id)
):
    """댓글 삭제"""
    try:
        # 댓글 조회
        result = await db.execute(
            select(Comment).where(Comment.id == comment_id)
        )
        comment = result.scalar_one_or_none()

        if not comment:
            raise HTTPException(status_code=404, detail="댓글을 찾을 수 없습니다")

        # 작성자 확인
        if comment.user_id != current_user_id:
            raise HTTPException(status_code=403, detail="본인의 댓글만 삭제할 수 있습니다")

        # 댓글 삭제
        await db.delete(comment)
        await db.commit()

        logger.info(f"Comment deleted: id={comment_id}")

        return {"success": True, "message": "댓글이 삭제되었습니다"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting comment: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="댓글 삭제 중 오류가 발생했습니다")
