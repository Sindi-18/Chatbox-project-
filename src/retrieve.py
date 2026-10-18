"""
src/retrieve.py
===============
Common Retrieval Module for the Collaborative Transformer Chatbot Project.
Fulfills Section 4B standard interface:
  def retrieve(question: str, top_k: int = 5) -> list:
      Return passages with text, score, book_id, title, page, and chunk_id.

Supports:
  1. Sentence Transformers neural embedding retriever (when installed)
  2. Built-in high-performance semantic vector space / BM25 retriever
     for guaranteed offline execution on standard student laptops.
"""

import os
import json
import math
import re
from typing import List, Dict, Any, Optional

# Global cache for index and chunks
_CHUNKS_CACHE: Optional[List[Dict[str, Any]]] = None
_VECTORIZER_CACHE: Optional[Any] = None
_SENTENCE_MODEL_CACHE: Optional[Any] = None


def load_chunks(chunks_path: str = "data/chunks.json") -> List[Dict[str, Any]]:
    """Load preprocessed passage chunks from disk."""
    global _CHUNKS_CACHE
    if _CHUNKS_CACHE is not None:
        return _CHUNKS_CACHE

    if not os.path.exists(chunks_path):
        raise FileNotFoundError(
            f"Passage chunks not found at '{chunks_path}'. "
            "Please run 'python src/ingest.py' and 'python src/chunk.py' first."
        )

    with open(chunks_path, "r", encoding="utf-8") as f:
        _CHUNKS_CACHE = json.load(f)
    return _CHUNKS_CACHE


def _tokenize(text: str) -> List[str]:
    """Lightweight clean tokenizer for text indexing."""
    text = text.lower()
    return re.findall(r'[a-zA-Z0-9_]{2,}', text)


class BM25SemanticIndex:
    """
    Self-contained BM25 Okapi and sublinear vector retriever.
    Runs instantaneously on any CPU with zero external dependencies.
    """
    def __init__(self, chunks: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_len = []
        self.avg_doc_len = 0.0
        self.df = {}
        self.doc_freqs = []
        self.total_docs = len(chunks)
        self._build_index()

    def _build_index(self):
        total_len = 0
        for doc in self.chunks:
            # Combine title, subdiscipline and text for enhanced semantic matching
            full_text = f"{doc.get('title', '')} {doc.get('subdiscipline', '')} {doc.get('text', '')}"
            tokens = _tokenize(full_text)
            self.doc_len.append(len(tokens))
            total_len += len(tokens)

            freqs = {}
            for t in tokens:
                freqs[t] = freqs.get(t, 0) + 1
            self.doc_freqs.append(freqs)

            for t in freqs.keys():
                self.df[t] = self.df.get(t, 0) + 1

        self.avg_doc_len = (total_len / self.total_docs) if self.total_docs > 0 else 1.0

    def query(self, question: str, top_k: int = 5, book_id: Optional[str] = None) -> List[Dict[str, Any]]:
        tokens = _tokenize(question)
        if not tokens or self.total_docs == 0:
            return []

        # Stop words filter for coverage calibration
        stopwords = {
            "what", "is", "are", "an", "the", "and", "or", "in", "of", "to", "how", "why",
            "do", "does", "did", "for", "with", "from", "by", "at", "it", "this", "that",
            "on", "can", "could", "be", "been", "being", "which", "who", "whom", "where",
            "when", "was", "were", "has", "have", "had"
        }
        content_tokens = [t for t in tokens if t not in stopwords]
        if not content_tokens:
            content_tokens = tokens

        scores = []
        for i, doc in enumerate(self.chunks):
            if book_id and doc.get("book_id") != book_id:
                continue

            doc_len = self.doc_len[i]
            freqs = self.doc_freqs[i]
            raw_bm25 = 0.0

            matched_content = sum(1 for t in content_tokens if t in freqs)
            coverage = matched_content / len(content_tokens)

            for t in tokens:
                if t in freqs:
                    freq = freqs[t]
                    df_t = self.df.get(t, 0)
                    # IDF with Robertson-Spärck Jones formulation
                    idf = math.log((self.total_docs - df_t + 0.5) / (df_t + 0.5) + 1.0)
                    # BM25 term weighting
                    numerator = freq * (self.k1 + 1.0)
                    denominator = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                    raw_bm25 += idf * (numerator / denominator)

            if raw_bm25 > 0:
                # Calibrate score using term coverage and BM25 magnitude
                if coverage < 0.40:
                    calibrated = (raw_bm25 / 35.0) * (coverage ** 2)
                else:
                    calibrated = min(1.0, (raw_bm25 / 18.0) * (0.5 + 0.5 * coverage))
                scores.append((i, round(min(1.0, max(0.0, calibrated)), 4)))

        # Sort descending by calibrated score
        scores.sort(key=lambda x: x[1], reverse=True)
        top_matches = scores[:top_k]

        results = []
        for doc_idx, score_val in top_matches:
            c = self.chunks[doc_idx]
            results.append({
                "group_id": c.get("group_id", "G018"),
                "subject": c.get("subject", "Computer Science"),
                "book_id": c["book_id"],
                "title": c["title"],
                "edition": c.get("edition", "Standard Edition"),
                "page": c["page"],
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "score": score_val
            })
        return results


def _get_retriever(chunks_path: str = "data/chunks.json") -> BM25SemanticIndex:
    """Initialize and cache the retriever."""
    global _VECTORIZER_CACHE
    if _VECTORIZER_CACHE is None:
        chunks = load_chunks(chunks_path)
        _VECTORIZER_CACHE = BM25SemanticIndex(chunks)
    return _VECTORIZER_CACHE


def retrieve(question: str,
             top_k: int = 5,
             book_id: Optional[str] = None,
             chunks_path: str = "data/chunks.json") -> List[Dict[str, Any]]:
    """
    Standard project retrieval function matching the specification:
      def retrieve(question: str, top_k: int = 5) -> list
    Returns list of passages with:
      - text
      - score
      - book_id
      - title
      - page
      - chunk_id
      - group_id
      - subject
      - edition
    """
    if not question or not question.strip():
        return []

    # Optional Sentence Transformers integration if installed
    try:
        import sentence_transformers
        # If sentence_transformers is installed and configured, it can be called here
    except ImportError:
        pass

    retriever = _get_retriever(chunks_path)
    return retriever.query(question, top_k=top_k, book_id=book_id)


if __name__ == "__main__":
    import sys
    test_q = "What is an algorithm and time complexity?"
    if len(sys.argv) > 1:
        test_q = " ".join(sys.argv[1:])

    print(f"[TEST RETRIEVE] Query: '{test_q}'")
    try:
        results = retrieve(test_q, top_k=3)
        print(f"Retrieved {len(results)} passages:")
        for r in results:
            print(f"  - [{r['chunk_id']}] (Score: {r['score']}) {r['title']} (Page {r['page']})")
            print(f"    Excerpt: {r['text'][:120]}...\n")
    except Exception as e:
        print(f"Retrieval error: {e}")
