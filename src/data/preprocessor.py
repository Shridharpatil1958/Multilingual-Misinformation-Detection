# src/data/preprocessor.py

"""
Text cleaning, language detection, and tokenization for multilingual claims.
Handles: Unicode normalization, noise removal, language detection, tokenization.
"""

import re
import unicodedata
from typing import List, Dict, Optional, Tuple

from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from transformers import AutoTokenizer
from src.utils.logger import get_logger

DetectorFactory.seed = 42  # reproducible lang detection
logger = get_logger(__name__)

SUPPORTED_LANGS = {"en", "hi", "ta", "bn"}

# Common noise patterns in WhatsApp forwards
NOISE_PATTERNS = [
    r"(forward(ed)?|फ़ॉरवर्ड|பகிர்)",          # forward markers
    r"https?://\S+",                               # URLs
    r"\+\d[\d\s\-]{8,}",                          # phone numbers
    r"[📲📢📣🔴🔵⚠️❗]+",                          # common spam emojis
    r"[\u200b-\u200f\ufeff]",                      # zero-width chars
]


class TextPreprocessor:
    def __init__(self, model_name: str = "xlm-roberta-large", max_length: int = 256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self._noise_re = re.compile("|".join(NOISE_PATTERNS), re.IGNORECASE)

    # ── Cleaning ────────────────────────────────────────────────────

    def clean(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        # Unicode NFC normalization
        text = unicodedata.normalize("NFC", text)
        # Remove noise
        text = self._noise_re.sub(" ", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def batch_clean(self, texts: List[str]) -> List[str]:
        return [self.clean(t) for t in texts]

    # ── Language Detection ──────────────────────────────────────────

    def detect_language(self, text: str) -> str:
        """Returns ISO 639-1 code. Falls back to 'en' on failure."""
        if not text or len(text.strip()) < 10:
            return "en"
        try:
            lang = detect(text)
            return lang if lang in SUPPORTED_LANGS else "en"
        except LangDetectException:
            return "en"

    def detect_languages_batch(self, texts: List[str]) -> List[str]:
        return [self.detect_language(t) for t in texts]

    # ── Tokenization ────────────────────────────────────────────────

    def tokenize(self, texts: List[str], **kwargs) -> Dict:
        """Tokenize for XLM-R / mDeBERTa input."""
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            **kwargs,
        )

    def tokenize_pairs(
        self, claims: List[str], evidences: List[str]
    ) -> Dict:
        """Tokenize claim-evidence pairs for NLI."""
        return self.tokenizer(
            claims,
            evidences,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    # ── Full Pipeline ───────────────────────────────────────────────

    def process_record(self, record: Dict) -> Dict:
        """Clean + detect language for a single record."""
        claim = self.clean(record.get("claim", ""))
        lang = record.get("language") or self.detect_language(claim)
        return {**record, "claim": claim, "language": lang}

    def process_batch(self, records: List[Dict]) -> List[Dict]:
        return [self.process_record(r) for r in records]


class CorpusChunker:
    """
    Split long Wikipedia / news articles into overlapping chunks
    suitable for dense retrieval indexing.
    """

    def __init__(self, chunk_size: int = 200, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, doc_id: str) -> List[Dict]:
        words = text.split()
        chunks = []
        step = self.chunk_size - self.overlap
        for i in range(0, len(words), step):
            chunk_words = words[i: i + self.chunk_size]
            if len(chunk_words) < 20:          # skip tiny trailing chunks
                continue
            chunks.append({
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_{i}",
                "text": " ".join(chunk_words),
                "start_word": i,
            })
        return chunks

    def chunk_corpus(self, documents: List[Dict]) -> List[Dict]:
        """documents: list of {id, text, ...}"""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_text(doc["text"], doc["id"])
            for chunk in chunks:
                chunk.update({k: v for k, v in doc.items() if k not in ("id", "text")})
            all_chunks.extend(chunks)
        logger.info(f"Chunked {len(documents)} docs → {len(all_chunks)} chunks")
        return all_chunks
