"""
tests/run_tests.py
==================
Automated Test Runner for the Collaborative Transformer Chatbot Project.
Executes all quality gate validation functions in tests/test_chatbot.py.
"""

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_chatbot import (
    test_manifest_minimum_scale,
    test_chunk_schema,
    test_chunk_id_formatting,
    test_retrieval_interface_and_scoring,
    test_answer_generation_schema,
    test_citation_traceability,
    test_abstention_on_insufficient_evidence
)


def run_all_tests():
    tests = [
        ("Book Manifest Scale (>=15 Books)", test_manifest_minimum_scale),
        ("Passage Chunk Schema (Section 4A)", test_chunk_schema),
        ("Chunk ID Formatting Regex", test_chunk_id_formatting),
        ("Retrieval Interface & Score Order (Section 4B)", test_retrieval_interface_and_scoring),
        ("Answer Generation Schema (Section 4C)", test_answer_generation_schema),
        ("Citation Traceability 100% Quality Gate", test_citation_traceability),
        ("Abstention on Insufficient Evidence", test_abstention_on_insufficient_evidence)
    ]

    print("=" * 65)
    print("  RUNNING AUTOMATED TEST SUITE: TRANSFORMER CHATBOT G018")
    print("=" * 65)

    passed = 0
    failed = 0
    t0 = time.time()

    for name, func in tests:
        try:
            func()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1

    duration = round(time.time() - t0, 3)
    print("=" * 65)
    print(f"  TEST RESULTS: {passed} PASSED, {failed} FAILED ({duration}s)")
    print("=" * 65)

    if failed > 0:
        sys.exit(1)
    else:
        print("  ALL QUALITY GATES SUCCESSFULLY PASSED!")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()
