#!/usr/bin/env python3
"""
Comprehensive Integration Test for Backend AI System

Tests:
1. Dependencies (torch, transformers, fastapi, etc.)
2. Sentiment service (model loading and prediction)
3. Database connection
4. Intent service
5. RAG service with SQL queries
6. End-to-end chat flow simulation
"""

import sys
import json
from pathlib import Path

# Setup path
proj_root = Path.cwd()
backend_dir = proj_root / "backend-ai"
sys.path.insert(0, str(backend_dir))


def test_dependencies():
    """Check if all required packages are installed."""
    print("\n" + "="*60)
    print("TEST 1: Checking Dependencies")
    print("="*60)
    
    deps = {
        'fastapi': 'FastAPI framework',
        'uvicorn': 'ASGI server',
        'pandas': 'Data manipulation',
        'sqlalchemy': 'SQL toolkit',
        'transformers': 'HuggingFace transformers (sentiment)',
        'torch': 'PyTorch (sentiment)',
        'pydantic': 'Data validation',
    }
    
    missing = []
    installed = []
    
    for pkg, desc in deps.items():
        try:
            __import__(pkg)
            installed.append(f"✓ {pkg:20s} ({desc})")
        except ImportError:
            missing.append(f"✗ {pkg:20s} ({desc})")
    
    for line in installed:
        print(line)
    
    if missing:
        print("\nMissing dependencies:")
        for line in missing:
            print(line)
        print("\nInstall with: pip install torch transformers")
        return False
    
    print("\n✓ All dependencies installed!")
    return True


def test_sentiment_service():
    """Test sentiment analysis."""
    print("\n" + "="*60)
    print("TEST 2: Sentiment Service")
    print("="*60)
    
    try:
        from services.sentiment_service import sentiment_service
        
        test_cases = [
            ("This mentor is absolutely fantastic! Highly recommended!", "positive"),
            ("Average experience, nothing special", "neutral"),
            ("Terrible experience, would not recommend", "negative"),
            ("ممتاز جداً! مرشد رائع جداً", "positive"),  # Arabic
            ("سيئ جداً، لن أوصي به", "negative"),  # Arabic
        ]
        
        print("\nTesting single predictions:")
        for text, expected_label in test_cases:
            result = sentiment_service.predict(text)
            status = "✓" if result.label == expected_label else "⚠"
            print(f"{status} Text: '{text[:50]}...'")
            print(f"  → Label: {result.label} | Confidence: {result.confidence:.4f}")
            print(f"  → Scores: {result.scores}")
        
        # Test batch
        print("\nTesting batch prediction:")
        batch_texts = [
            "Great mentor! Very helpful!",
            "Not bad, could be better",
            "Awful experience"
        ]
        results = sentiment_service.predict_batch(batch_texts)
        print(f"✓ Batch predictions: {len(results)} results")
        for i, (text, result) in enumerate(zip(batch_texts, results)):
            print(f"  {i+1}. {text[:40]:40s} → {result.label:8s} ({result.confidence:.4f})")
        
        return True
    
    except Exception as e:
        print(f"✗ Sentiment service failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_database():
    """Test database connection and basic queries."""
    print("\n" + "="*60)
    print("TEST 3: Database Connection")
    print("="*60)
    
    try:
        from database.db import database
        
        # Health check
        is_healthy = database.health_check()
        print(f"{'✓' if is_healthy else '✗'} Database health check: {'OK' if is_healthy else 'FAILED'}")
        
        if not is_healthy:
            return False
        
        # Test some basic tables
        tables_to_check = ['users', 'mentor_profile', 'mentee_profile', 'feedbacks', 'mentorships']
        print("\nChecking tables:")
        for table in tables_to_check:
            exists = database.table_exists(table)
            status = "✓" if exists else "✗"
            print(f"  {status} {table}")
        
        return True
    
    except Exception as e:
        print(f"✗ Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_intent_service():
    """Test intent detection."""
    print("\n" + "="*60)
    print("TEST 4: Intent Service")
    print("="*60)
    
    try:
        from services.intent_service import intent_service
        
        test_cases = [
            ("I need mentor recommendations", "recommendation"),
            ("What courses can I take?", "materials"),
            ("How do I register?", "faq"),
            ("Show me mentor stats", "stats"),
            ("أريد مرشد في مجال البرمجة", "recommendation"),  # Arabic
        ]
        
        print("\nTesting intent detection (rule-based):")
        for message, expected_intent in test_cases:
            intent = intent_service.detect_intent(message)
            status = "✓" if intent == expected_intent else "⚠"
            print(f"{status} '{message[:50]}...'")
            print(f"  → Intent: {intent} (expected: {expected_intent})")
        
        return True
    
    except Exception as e:
        print(f"✗ Intent service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rag_queries():
    """Test RAG SQL query helpers."""
    print("\n" + "="*60)
    print("TEST 5: RAG Service SQL Queries")
    print("="*60)
    
    try:
        from services.rag_service import (
            get_top_mentors_by_rating,
            get_top_mentors_by_feedback,
            get_mentor_counts_by_domain,
            get_open_programs,
        )
        
        print("\nTesting query helpers:")
        
        # Test top mentors by rating
        print("\n1. get_top_mentors_by_rating(limit=3)")
        df = get_top_mentors_by_rating(limit=3)
        print(f"  ✓ Returned {len(df)} rows")
        if len(df) > 0:
            print(f"  Columns: {list(df.columns)}")
            print(f"  Sample: {df.iloc[0].to_dict()}")
        
        # Test top mentors by feedback
        print("\n2. get_top_mentors_by_feedback(limit=3)")
        df = get_top_mentors_by_feedback(limit=3)
        print(f"  ✓ Returned {len(df)} rows")
        
        # Test mentor counts
        print("\n3. get_mentor_counts_by_domain()")
        df = get_mentor_counts_by_domain()
        print(f"  ✓ Returned {len(df)} rows (domains)")
        if len(df) > 0:
            print(f"  Sample: {df.iloc[0].to_dict()}")
        
        # Test open programs
        print("\n4. get_open_programs(limit=5)")
        df = get_open_programs(limit=5)
        print(f"  ✓ Returned {len(df)} rows")
        
        return True
    
    except Exception as e:
        print(f"✗ RAG service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chat_flow():
    """Simulate end-to-end chat flow."""
    print("\n" + "="*60)
    print("TEST 6: End-to-End Chat Flow (Simulation)")
    print("="*60)
    
    try:
        from services.intent_service import intent_service
        from services.rag_service import rag_service
        
        test_messages = [
            ("I need mentor recommendations", "en"),
            ("What programs are available?", "en"),
            ("أخبرني عن المرشدين في البرمجة", "ar"),
        ]
        
        print("\nSimulating chat requests:")
        for message, language in test_messages:
            intent = intent_service.detect_intent(message)
            print(f"\n✓ Message: '{message}'")
            print(f"  Intent: {intent}")
            print(f"  Language: {language}")
            
            # Simulate RAG flow
            if intent in ["faq", "data_query"]:
                answer = rag_service.answer_platform_question(message, language)
                print(f"  Answer: {answer[:100]}...")
        
        return True
    
    except Exception as e:
        print(f"✗ Chat flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "🧪 Backend AI System Integration Tests 🧪".center(60))
    
    results = {
        "Dependencies": test_dependencies(),
        "Sentiment Service": test_sentiment_service(),
        "Database": test_database(),
        "Intent Service": test_intent_service(),
        "RAG Queries": test_rag_queries(),
        "Chat Flow": test_chat_flow(),
    }
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} {test_name}")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Backend AI system is ready.")
        return 0
    else:
        print("\n✗ Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
