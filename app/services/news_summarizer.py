"""뉴스 한줄요약 서비스

한국어 금융 뉴스에서 핵심 내용을 한 문장으로 추출합니다.

Usage:
    from app.services.news_summarizer import NewsSummarizer

    summarizer = NewsSummarizer()
    summary = summarizer.summarize(title="삼성전자, 4분기 영업이익 6조원 달성", content="...")
    # "삼성전자가 4분기 영업이익 6조원을 달성하며 시장 기대치를 상회했다."
"""
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# 최대 요약 길이 (글자 수)
MAX_SUMMARY_LENGTH = 80


class NewsSummarizer:
    """한국어 금융 뉴스 한줄요약 생성기 (추출형)"""

    # 불필요한 접두어 패턴 (제거 대상)
    NOISE_PATTERNS = [
        r"^\[.*?\]\s*",           # [속보], [단독], [마감] 등
        r"^【.*?】\s*",
        r"^\(.*?\)\s*",           # (종합), (1보) 등
        r"^<.*?>\s*",
        r"\s*\[.*?사진\]",
        r"\s*\/\s*사진\s*=.*$",
        r"\s*▶.*$",              # ▶ 관련기사
        r"\s*©.*$",              # 저작권
        r"\(끝\)\s*$",
    ]

    # 문장 종결 패턴
    SENTENCE_ENDERS = re.compile(r"[.다요함됨임음습니다까]\s|[.다요함됨임음습니다까]$")

    def summarize(self, title: str, content: Optional[str] = None) -> str:
        """뉴스 한줄요약 생성

        Args:
            title: 뉴스 제목
            content: 뉴스 본문 (없으면 제목 기반)

        Returns:
            한줄요약 문자열
        """
        # 본문이 있으면 첫 핵심 문장 추출
        if content and len(content.strip()) > 20:
            summary = self._extract_from_content(title, content)
            if summary:
                return summary

        # 본문이 없거나 추출 실패 시 제목 정제
        return self._clean_title(title)

    def _extract_from_content(self, title: str, content: str) -> Optional[str]:
        """본문에서 핵심 문장 추출"""
        cleaned = self._clean_text(content)
        if not cleaned:
            return None

        sentences = self._split_sentences(cleaned)
        if not sentences:
            return None

        # 첫 문장이 제목과 거의 동일하면 두 번째 문장 사용
        first = sentences[0]
        if self._is_similar(first, title) and len(sentences) > 1:
            best = sentences[1]
        else:
            best = first

        # 길이 제한
        if len(best) > MAX_SUMMARY_LENGTH:
            best = best[:MAX_SUMMARY_LENGTH - 1] + "…"

        return best

    def _clean_title(self, title: str) -> str:
        """제목 정제 (노이즈 제거)"""
        cleaned = title.strip()
        for pattern in self.NOISE_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned)
        cleaned = cleaned.strip()

        if len(cleaned) > MAX_SUMMARY_LENGTH:
            cleaned = cleaned[:MAX_SUMMARY_LENGTH - 1] + "…"

        return cleaned

    def _clean_text(self, text: str) -> str:
        """본문 텍스트 정제"""
        cleaned = text.strip()

        # 노이즈 패턴 제거
        for pattern in self.NOISE_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned)

        # 연속 공백 → 단일 공백
        cleaned = re.sub(r"\s+", " ", cleaned)

        # 기자명/이메일 줄 제거
        cleaned = re.sub(r"\S+@\S+\.\S+", "", cleaned)
        cleaned = re.sub(r"\S+\s기자\s*$", "", cleaned)

        return cleaned.strip()

    def _split_sentences(self, text: str) -> list:
        """한국어 문장 분리"""
        # 마침표/종결어미 기준 분리
        raw_splits = re.split(r"(?<=[.다요함됨임음])\s+", text)

        sentences = []
        for s in raw_splits:
            s = s.strip()
            if len(s) >= 10:  # 너무 짧은 문장 제외
                sentences.append(s)

        return sentences

    def _is_similar(self, a: str, b: str) -> bool:
        """두 문장의 유사도 판단 (단순 겹침 비율)"""
        a_set = set(a.replace(" ", ""))
        b_set = set(b.replace(" ", ""))

        if not a_set or not b_set:
            return False

        overlap = len(a_set & b_set) / min(len(a_set), len(b_set))
        return overlap > 0.7

    def summarize_batch(self, items: list) -> list:
        """여러 뉴스를 일괄 요약

        Args:
            items: [{'title': str, 'content': str|None}, ...]

        Returns:
            요약 문자열 리스트
        """
        return [
            self.summarize(item.get("title", ""), item.get("content"))
            for item in items
        ]
