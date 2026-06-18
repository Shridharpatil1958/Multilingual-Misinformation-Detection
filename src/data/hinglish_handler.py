# src/data/hinglish_handler.py

"""
Handles code-mixed Hindi-English ("Hinglish") text — a major real-world gap.
Strategies:
  1. Detect if text is Hinglish (character n-gram heuristic)
  2. Transliterate Roman-script Hindi → Devanagari (for model input)
  3. Segment tokens by language for targeted encoding
"""

import re
from typing import List, Tuple, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Devanagari Unicode block: U+0900–U+097F
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
# Tamil Unicode block: U+0B80–U+0BFF
TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")

# Common Hinglish romanized Hindi words (partial list for detection)
HINGLISH_SEED_WORDS = {
    "hai", "nahi", "kya", "aur", "mein", "ka", "ki", "ke", "se",
    "ho", "hota", "bhi", "toh", "yeh", "woh", "unka", "inka",
    "sarkaar", "sarkar", "desh", "log", "matlab", "theek", "sach",
    "jhooth", "jhoot", "fake", "viral", "forward", "share",
}


class HinglishDetector:
    """
    Heuristic detector for code-mixed Hindi-English text.
    Returns a confidence score [0, 1] — higher = more Hinglish.
    """

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

    def score(self, text: str) -> float:
        if not text:
            return 0.0
        tokens = text.lower().split()
        if not tokens:
            return 0.0

        # Check for Devanagari presence (already handled by multilingual model)
        if DEVANAGARI_RE.search(text):
            return 0.0   # pure Hindi, not code-mixed Roman

        # Count tokens that look like romanized Hindi
        hindi_count = sum(1 for t in tokens if t in HINGLISH_SEED_WORDS)
        # Count tokens that look like English (simple heuristic: ASCII + length > 3)
        english_count = sum(1 for t in tokens if t.isascii() and len(t) > 3 and t not in HINGLISH_SEED_WORDS)

        total = len(tokens)
        if total == 0:
            return 0.0

        # Hinglish score: ratio of Hindi words in otherwise Roman-script text
        hindi_ratio = hindi_count / total
        return min(hindi_ratio * 2.5, 1.0)  # scale so 12%+ triggers detection

    def is_hinglish(self, text: str) -> bool:
        return self.score(text) >= self.threshold


class HinglishNormalizer:
    """
    Normalizes Hinglish text for better model input.
    Approach: prepend a [HINGLISH] marker so the model can learn
    to handle this distribution, and expand common abbreviations.
    """

    # Common Hinglish shortenings seen in WhatsApp forwards
    EXPANSIONS = {
        r"\bnhi\b": "nahi",
        r"\bkr\b": "kar",
        r"\bhn\b": "haan",
        r"\bbhai\b": "bhai",    # already correct, kept for demo
        r"\bkuch\b": "kuch",
        r"\bbc\b": "",           # common abuse abbreviation — remove
        r"\bbtw\b": "by the way",
        r"\bifykyk\b": "if you know you know",
    }

    def __init__(self):
        self._patterns = [
            (re.compile(pat, re.IGNORECASE), repl)
            for pat, repl in self.EXPANSIONS.items()
        ]

    def normalize(self, text: str, add_marker: bool = True) -> str:
        for pat, repl in self._patterns:
            text = pat.sub(repl, text)
        text = re.sub(r"\s+", " ", text).strip()
        if add_marker:
            text = "[HINGLISH] " + text
        return text


class MultilingualSegmenter:
    """
    Segment a mixed-script text into (language, segment) pairs.
    Useful for targeted encoding and analysis.

    Example:
        "यह news सच है या fake?" →
        [("hi", "यह"), ("en", "news"), ("hi", "सच है या"), ("en", "fake")]
    """

    def segment(self, text: str) -> List[Tuple[str, str]]:
        segments = []
        current_lang: Optional[str] = None
        current_tokens: List[str] = []

        for token in text.split():
            if DEVANAGARI_RE.search(token):
                lang = "hi"
            elif TAMIL_RE.search(token):
                lang = "ta"
            else:
                lang = "en"

            if lang != current_lang:
                if current_tokens:
                    segments.append((current_lang, " ".join(current_tokens)))
                current_tokens = [token]
                current_lang = lang
            else:
                current_tokens.append(token)

        if current_tokens:
            segments.append((current_lang, " ".join(current_tokens)))

        return segments

    def dominant_language(self, text: str) -> str:
        segments = self.segment(text)
        counts: dict = {}
        for lang, seg in segments:
            counts[lang] = counts.get(lang, 0) + len(seg.split())
        return max(counts, key=counts.get) if counts else "en"


class HinglishPipeline:
    """Convenience wrapper: detect → normalize → segment."""

    def __init__(self, threshold: float = 0.3):
        self.detector = HinglishDetector(threshold)
        self.normalizer = HinglishNormalizer()
        self.segmenter = MultilingualSegmenter()

    def process(self, text: str) -> Dict:
        is_hinglish = self.detector.is_hinglish(text)
        normalized = self.normalizer.normalize(text) if is_hinglish else text
        segments = self.segmenter.segment(text)
        dominant = self.segmenter.dominant_language(text)

        return {
            "original": text,
            "is_hinglish": is_hinglish,
            "hinglish_score": self.detector.score(text),
            "normalized": normalized,
            "segments": segments,
            "dominant_language": dominant,
        }


# Fix missing import
from typing import Dict
