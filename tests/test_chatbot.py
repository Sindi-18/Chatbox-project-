"""
tests/test_chatbot.py
=====================
Automated Test Suite for Collaborative Transformer Chatbot Project.
Fulfills Section 9 & Section 11 quality gate requirements:
  - Ingestion and Book Manifest validation (>= 15 books)
  - Standard passage record schema compliance (Section 4A)
  - Common retrieval interface and score ordering (Section 4B)
  - Common answer generation schema (Section 4C)
  - 100% citation traceability (every citation resolves to a real passage)
  - Insufficient-evidence detection and abstention behavior
"""

import os
import json
try:
    import pytest
except ImportError:
    pytest = None
from src.retrieve import retrieve, load_chunks
from src.generate import generate_answer


def test_manifest_minimum_scale():
    """Verify that at least 15 books are recorded with complete bibliographic metadata."""
    manifest_path = "data/book_manifest.json"
    assert os.path.exists(manifest_path), f"Manifest file missing at {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Standard requirement: >= 15 books per group
    assert len(manifest) >= 15, f"Expected >= 15 books in manifest, found {len(manifest)}"

    required_fields = ["book_id", "title", "author", "edition", "permission_status", "processing_status"]
    for book in manifest:
        for field in required_fields:
            assert field in book, f"Book {book.get('book_id')} missing required field: {field}"
            assert book[field], f"Book {book.get('book_id')} field '{field}' is empty"


def test_chunk_schema():
    """Verify that passage chunks strictly conform to Section 4A schema."""
    chunks_path = "data/chunks.json"
    assert os.path.exists(chunks_path), f"Chunks file missing at {chunks_path}"

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    assert len(chunks) > 0, "No chunks found in data/chunks.json"

    # Check top sample chunks for schema compliance
    mandatory_keys = {"group_id", "subject", "book_id", "title", "edition", "page", "chunk_id", "text"}
    for c in chunks[:50]:
        missing = mandatory_keys - set(c.keys())
        assert not missing, f"Chunk {c.get('chunk_id')} is missing keys: {missing}"
        assert isinstance(c["page"], int), f"Page must be integer in chunk {c['chunk_id']}"
        assert len(c["text"]) > 20, f"Chunk text suspiciously short in {c['chunk_id']}"


def test_chunk_id_formatting():
    """Verify that chunk IDs follow the required format: {book_id}_P{page}_C{chunk_idx}."""
    chunks = load_chunks()
    assert len(chunks) > 0

    import re
    pattern = re.compile(r"^G\d+_BOOK_\d+_P\d+_C\d+$")
    for c in chunks[:50]:
        cid = c["chunk_id"]
        assert pattern.match(cid), f"Chunk ID '{cid}' does not match standard pattern"


def test_retrieval_interface_and_scoring():
    """Verify Section 4B retrieve() function return schema and score sorting."""
    query = "What is an algorithm and time complexity?"
    results = retrieve(query, top_k=5)

    assert isinstance(results, list), "retrieve() must return a list"
    assert len(results) > 0, "Expected at least 1 retrieved passage for standard query"
    assert len(results) <= 5, "retrieve() returned more than top_k passages"

    # Verify return fields
    for r in results:
        for key in ["text", "score", "book_id", "title", "page", "chunk_id"]:
            assert key in r, f"Retrieved record missing required key: {key}"
        assert 0.0 <= r["score"] <= 1.0, f"Score {r['score']} out of normalized [0, 1] bounds"

    # Verify descending score sorting
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), "Retrieved passages are not sorted by score descending"


def test_answer_generation_schema():
    """Verify Section 4C answer format schema."""
    query = "Explain instruction pipelining in computer architecture."
    response = generate_answer(query, top_k=3)

    assert isinstance(response, dict), "generate_answer() must return a dict"
    for key in ["answer", "sources", "insufficient_evidence"]:
        assert key in response, f"generate_answer() missing required key: {key}"

    assert isinstance(response["answer"], str), "Answer must be a string"
    assert len(response["answer"]) > 10, "Answer content is empty or too short"
    assert isinstance(response["sources"], list), "Sources must be a list"
    assert response["insufficient_evidence"] is False, "Expected sufficient evidence for in-domain query"


def test_citation_traceability():
    """
    CRITICAL QUALITY GATE:
    Verify that 100% of returned citations resolve to an actual stored passage in chunks.json.
    """
    all_chunks = load_chunks()
    chunk_id_registry = {c["chunk_id"]: c for c in all_chunks}

    query = "What are logic gates and digital circuits?"
    response = generate_answer(query, top_k=4)

    sources = response["sources"]
    assert len(sources) > 0, "Expected sources for in-domain query"

    for s in sources:
        cid = s["chunk_id"]
        assert cid in chunk_id_registry, f"Citation '{cid}' does NOT resolve to any real stored passage!"
        # Verify page matches stored passage page
        expected_page = chunk_id_registry[cid]["page"]
        assert s["page"] == expected_page, (
            f"Citation page mismatch for {cid}: cited {s['page']} vs stored {expected_page}"
        )


def test_abstention_on_insufficient_evidence():
    """Verify chatbot abstains with insufficient_evidence=True on out-of-domain questions."""
    unsupported_query = "Who won the 1994 FIFA World Cup soccer championship?"
    response = generate_answer(unsupported_query)

    assert response["insufficient_evidence"] is True, (
        f"Expected insufficient_evidence=True for unsupported query, got False"
    )
    assert len(response["sources"]) == 0, "Abstention response should not cite unrelated sources"
    assert "Insufficient evidence" in response["answer"], (
        "Expected abstention message to inform user of insufficient evidence"
    )
