"""한국어 뉴스 감성 분석 서비스

키워드 기반 분석 (기본, 빠름) + Transformer 기반 분석 (옵션, 정확)

Usage:
    from app.services.news_sentiment import NewsSentimentAnalyzer

    analyzer = NewsSentimentAnalyzer()
    score, label = analyzer.analyze("삼성전자 실적 호조로 주가 급등")
    # score: 80, label: "positive"

    # Transformer 모드
    analyzer = NewsSentimentAnalyzer(use_transformer=True)
"""
import logging
import os
import sys
from pathlib import Path
from typing import Tuple, Optional, List, Dict

# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


class NewsSentimentAnalyzer:
    """한국 주식 뉴스 감성 분석기"""

    # 긍정 키워드 (키워드, 가중치)
    POSITIVE_KEYWORDS = {
        # 강한 긍정 (가중치 높음)
        '급등': 15, '폭등': 15, '신고가': 15, '사상최고': 15,
        '흑자전환': 15, '어닝서프라이즈': 15,
        # 일반 긍정
        '상승': 10, '호재': 10, '매수': 8, '성장': 10,
        '최고': 8, '돌파': 10, '반등': 10, '호실적': 12,
        '흑자': 10, '개선': 8, '확대': 8, '증가': 8,
        '상향': 10, '목표가': 5, '호전': 8, '수혜': 8,
        '기대': 5, '강세': 10, '회복': 8, '안정': 5,
        '수주': 8, '투자': 5, '배당': 5, '인수': 5,
        '협력': 5, '계약': 5, '승인': 8, '특허': 8,
        '혁신': 5, '선도': 5, '1위': 8, '최대': 5,
        '기록': 5, '달성': 5, '초과': 8, '호황': 10,
        '랠리': 10, '매력': 5, '유망': 5, '추천': 5,
    }

    # 부정 키워드 (키워드, 가중치)
    NEGATIVE_KEYWORDS = {
        # 강한 부정 (가중치 높음)
        '급락': 15, '폭락': 15, '신저가': 15, '사상최저': 15,
        '적자전환': 15, '어닝쇼크': 15,
        # 일반 부정
        '하락': 10, '악재': 10, '매도': 8, '감소': 8,
        '최저': 8, '부진': 10, '적자': 10, '손실': 10,
        '약세': 10, '위기': 10, '축소': 8, '하향': 10,
        '우려': 8, '리스크': 8, '불안': 8, '둔화': 8,
        '하회': 8, '미달': 8, '실패': 10, '중단': 8,
        '철수': 8, '소송': 8, '과징금': 10, '벌금': 10,
        '제재': 10, '규제': 5, '파산': 15, '부도': 15,
        '디폴트': 15, '약화': 8, '악화': 10, '침체': 10,
        '불황': 10, '공매도': 5, '투매': 10, '반도체한파': 10,
    }

    def __init__(self, use_transformer: bool = False):
        """
        Args:
            use_transformer: Transformer 모델 사용 여부
        """
        self.use_transformer = use_transformer
        self._transformer_pipeline = None

        if self.use_transformer:
            self._load_transformer()

    def _load_transformer(self):
        """Transformer 모델 로드"""
        try:
            from transformers import pipeline
            import torch

            device = 0 if torch.cuda.is_available() else -1
            self._transformer_pipeline = pipeline(
                "sentiment-analysis",
                model="snunlp/KR-FinBert-SC",
                device=device,
                truncation=True,
                max_length=512
            )
            logger.info(f"Transformer model loaded (device={'GPU' if device == 0 else 'CPU'})")
        except ImportError:
            logger.warning("transformers/torch not installed. Falling back to keyword-based analysis.")
            self.use_transformer = False
        except Exception as e:
            logger.warning(f"Failed to load transformer model: {e}. Falling back to keyword-based.")
            self.use_transformer = False

    def analyze(self, text: str) -> Tuple[int, str]:
        """뉴스 텍스트 감성 분석

        Args:
            text: 뉴스 제목 + 내용

        Returns:
            (sentiment_score, sentiment_label)
            score: -100 ~ +100
            label: "positive", "negative", "neutral"
        """
        if not text or not text.strip():
            return 0, "neutral"

        if self.use_transformer and self._transformer_pipeline is not None:
            return self._analyze_with_transformer(text)

        return self._analyze_with_keywords(text)

    def _analyze_with_keywords(self, text: str) -> Tuple[int, str]:
        """키워드 기반 감성 분석

        Args:
            text: 분석할 텍스트

        Returns:
            (score, label) - score는 -100~100 범위
        """
        positive_score = 0
        negative_score = 0

        for keyword, weight in self.POSITIVE_KEYWORDS.items():
            count = text.count(keyword)
            if count > 0:
                positive_score += weight * count

        for keyword, weight in self.NEGATIVE_KEYWORDS.items():
            count = text.count(keyword)
            if count > 0:
                negative_score += weight * count

        total = positive_score + negative_score
        if total == 0:
            return 0, "neutral"

        # -100 ~ +100 범위로 정규화
        raw_score = positive_score - negative_score
        # 최대 가능 점수 기준 정규화 (대략 50점을 100으로 매핑)
        score = max(-100, min(100, int(raw_score * 2)))

        if score > 10:
            label = "positive"
        elif score < -10:
            label = "negative"
        else:
            label = "neutral"

        return score, label

    def _analyze_with_transformer(self, text: str) -> Tuple[int, str]:
        """Transformer 기반 감성 분석

        KR-FinBert-SC 모델 출력:
        - label: "positive", "negative", "neutral"
        - score: 0~1 (확률)
        """
        try:
            # 텍스트 길이 제한 (512 토큰)
            truncated = text[:500]
            result = self._transformer_pipeline(truncated)[0]

            transformer_label = result['label'].lower()
            transformer_confidence = result['score']

            # label → score 변환 (-100 ~ +100)
            if transformer_label == 'positive':
                score = int(transformer_confidence * 100)
            elif transformer_label == 'negative':
                score = int(-transformer_confidence * 100)
            else:
                score = 0

            # neutral 임계값 보정
            if abs(score) <= 20:
                label = "neutral"
            elif score > 0:
                label = "positive"
            else:
                label = "negative"

            return score, label

        except Exception as e:
            logger.warning(f"Transformer analysis failed: {e}, falling back to keywords")
            return self._analyze_with_keywords(text)

    def analyze_batch(self, texts: List[str]) -> List[Tuple[int, str]]:
        """여러 뉴스 일괄 분석

        Args:
            texts: 뉴스 텍스트 리스트

        Returns:
            [(score, label), ...] 리스트
        """
        return [self.analyze(text) for text in texts]

    def process_unprocessed_news(self, use_transformer: bool = False):
        """DB에서 미처리 뉴스를 조회하여 감성 분석 수행

        Args:
            use_transformer: Transformer 모델 사용 여부
        """
        from dotenv import load_dotenv
        from sqlalchemy import create_engine, select
        from sqlalchemy.orm import sessionmaker
        from app.models.stock_news import StockNews

        load_dotenv()

        db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/stocksense")
        db_url = db_url.replace("+asyncpg", "")
        engine = create_engine(db_url)
        Session = sessionmaker(bind=engine)

        # Transformer 옵션이 지정되면 동적으로 변경
        if use_transformer and not self.use_transformer:
            self.use_transformer = True
            self._load_transformer()

        with Session() as session:
            # 미처리 뉴스 조회
            result = session.execute(
                select(StockNews)
                .where(StockNews.is_processed == False)
                .order_by(StockNews.crawled_at.desc())
            )
            news_list = result.scalars().all()

            if not news_list:
                print("No unprocessed news found.")
                return 0

            print(f"Processing {len(news_list)} unprocessed news articles...")

            processed_count = 0
            for news in news_list:
                # 제목 + 내용 결합
                text = news.title or ""
                if news.content:
                    text += " " + news.content

                score, label = self.analyze(text)

                news.sentiment_score = score
                news.sentiment_label = label
                news.is_processed = True
                processed_count += 1

            session.commit()
            print(f"Processed {processed_count} news articles.")
            return processed_count


def main():
    """CLI로 미처리 뉴스 감성 분석 실행"""
    import argparse

    parser = argparse.ArgumentParser(description="뉴스 감성 분석")
    parser.add_argument('--use-transformer', action='store_true',
                        help='Transformer 모델 사용 (기본: 키워드)')
    parser.add_argument('--test', type=str, default=None,
                        help='테스트 텍스트 분석')
    args = parser.parse_args()

    analyzer = NewsSentimentAnalyzer(use_transformer=args.use_transformer)

    if args.test:
        score, label = analyzer.analyze(args.test)
        print(f"Text: {args.test}")
        print(f"Score: {score}, Label: {label}")
    else:
        count = analyzer.process_unprocessed_news(use_transformer=args.use_transformer)
        print(f"Total processed: {count}")


if __name__ == "__main__":
    main()
