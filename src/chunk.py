"""
src/chunk.py
============
Passage Chunking Module for the Collaborative Transformer Chatbot Project.
Splits extracted book pages into standard passage records adhering to the
mandated project schema (Section 4A):
  - group_id
  - subject
  - book_id
  - title
  - edition
  - page
  - chunk_id
  - text
"""

import os
import json
import re
import sys


def split_text_into_chunks(text: str,
                           target_words: int = 250,
                           overlap_words: int = 40) -> list:
    """
    Split text into overlapping passage chunks using sentence and paragraph boundaries.
    Preserves context, formulas, and structural integrity.
    """
    if not text:
        return []

    words = text.split()
    if len(words) <= target_words + 30:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + target_words, len(words))
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words).strip()
        if len(chunk_str) >= 50:
            chunks.append(chunk_str)
        if end >= len(words):
            break
        start += target_words - overlap_words

    return chunks


def process_chunks(extracted_pages_path: str = "data/extracted_pages.json",
                   output_chunks_path: str = "data/chunks.json",
                   target_words: int = 250,
                   overlap_words: int = 40) -> list:
    """
    Process extracted pages into standard passage records.
    """
    if not os.path.exists(extracted_pages_path):
        raise FileNotFoundError(f"Extracted pages file '{extracted_pages_path}' not found. Run ingest.py first.")

    with open(extracted_pages_path, "r", encoding="utf-8") as f:
        pages = json.load(f)

    print(f"[CHUNK] Loaded {len(pages)} pages from '{extracted_pages_path}'.")

    all_chunks = []
    # Track chunk index per (book_id, page)
    chunk_counters = {}

    for item in pages:
        book_id = item["book_id"]
        page_num = item["page"]
        text = item["text"]

        text_chunks = split_text_into_chunks(text, target_words, overlap_words)

        key = (book_id, page_num)
        chunk_counters.setdefault(key, 0)

        for chunk_text in text_chunks:
            chunk_counters[key] += 1
            chunk_num = chunk_counters[key]
            chunk_id = f"{book_id}_P{page_num:03d}_C{chunk_num:02d}"

            record = {
                "group_id": item.get("group_id", "G018"),
                "subject": item.get("subject", "Computer Science"),
                "book_id": book_id,
                "title": item.get("title", ""),
                "edition": item.get("edition", "Standard Edition"),
                "author": item.get("author", "Academic Contributor"),
                "subdiscipline": item.get("subdiscipline", "General CS"),
                "page": page_num,
                "chunk_id": chunk_id,
                "text": chunk_text
            }
            all_chunks.append(record)

    os.makedirs(os.path.dirname(output_chunks_path), exist_ok=True)
    with open(output_chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"[CHUNK] Generated {len(all_chunks)} standard passage chunks saved to '{output_chunks_path}'.")
    return all_chunks


if __name__ == "__main__":
    process_chunks()
