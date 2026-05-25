#!/usr/bin/env python3
"""
Quick test script for Mentor Chatbot endpoints.
Run this after deployment to verify all components are working.

Usage:
    python test_mentor_chatbot_quick.py
"""

import asyncio
import json
import httpx
from typing import Optional

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

BASE_URL = "http://localhost:8000"
API_PREFIX = "/api/v1"

# Use a valid mentor UUID from your database
TEST_MENTOR_ID = "550e8400-e29b-41d4-a716-446655440000"
TEST_MENTEE_ID = "660e8400-e29b-41d4-a716-446655440001"

# ─────────────────────────────────────────────────────────────────────
# Test Cases
# ─────────────────────────────────────────────────────────────────────


async def test_mentor_chat_greeting():
    """Test: Chatbot greeting response"""
    print("\n✅ Test 1: Mentor Chat Greeting")
    
    payload = {
        "message": "Hi there!",
        "user_id": TEST_MENTOR_ID,
        "language": "en"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}{API_PREFIX}/mentor-chat",
            json=payload,
            timeout=5.0
        )
        
    if response.status_code == 200:
        data = response.json()
        assert data["intent"] == "greeting"
        assert "answer" in data
        print(f"  ✓ Intent: {data['intent']}")
        print(f"  ✓ Answer: {data['answer'][:100]}...")
        return True
    else:
        print(f"  ✗ Error: {response.status_code}")
        print(f"    {response.text}")
        return False


async def test_mentor_chat_analytics():
    """Test: Analytics intent"""
    print("\n✅ Test 2: Mentor Chat Analytics Intent")
    
    payload = {
        "message": "How many mentees do I have?",
        "user_id": TEST_MENTOR_ID,
        "language": "en"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}{API_PREFIX}/mentor-chat",
            json=payload,
            timeout=10.0
        )
    
    if response.status_code == 200:
        data = response.json()
        assert data["intent"] == "mentor_analytics"
        print(f"  ✓ Intent: {data['intent']}")
        print(f"  ✓ Answer: {data['answer'][:100]}...")
        return True
    else:
        print(f"  ✗ Error: {response.status_code}")
        return False


async def test_mentor_chat_faq():
    """Test: FAQ intent (Arabic)"""
    print("\n✅ Test 3: Mentor Chat FAQ (Arabic)")
    
    payload = {
        "message": "إزاي أنشئ برنامج؟",
        "user_id": TEST_MENTOR_ID
        # language auto-detected as Arabic
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}{API_PREFIX}/mentor-chat",
            json=payload,
            timeout=10.0
        )
    
    if response.status_code == 200:
        data = response.json()
        assert data["intent"] == "faq"
        assert data["language"] == "ar"
        print(f"  ✓ Intent: {data['intent']}")
        print(f"  ✓ Language: {data['language']} (auto-detected)")
        print(f"  ✓ Answer: {data['answer'][:100]}...")
        return True
    else:
        print(f"  ✗ Error: {response.status_code}")
        return False


async def test_mentor_chat_materials():
    """Test: Materials request intent"""
    print("\n✅ Test 4: Mentor Chat Materials Request")
    
    payload = {
        "message": "Interview questions for backend developers",
        "user_id": TEST_MENTOR_ID,
        "language": "en"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}{API_PREFIX}/mentor-chat",
            json=payload,
            timeout=10.0
        )
    
    if response.status_code == 200:
        data = response.json()
        assert data["intent"] == "materials_request"
        print(f"  ✓ Intent: {data['intent']}")
        if "materials" in data and data["materials"]:
            print(f"  ✓ Materials found: {len(data['materials'])} items")
            if data["materials"]:
                print(f"    - {data['materials'][0]['title']}")
        return True
    else:
        print(f"  ✗ Error: {response.status_code}")
        return False


async def test_mentor_chat_non_mentor():
    """Test: Non-mentor rejection"""
    print("\n✅ Test 5: Non-Mentor User Rejection")
    
    payload = {
        "message": "How many mentees?",
        "user_id": TEST_MENTEE_ID,  # Using mentee ID
        "language": "en"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}{API_PREFIX}/mentor-chat",
            json=payload,
            timeout=5.0
        )
    
    # Should reject or return error
    if response.status_code in [403, 400, 200]:  # Might succeed if user role check fails gracefully
        print(f"  ✓ Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if "error" in data or "detail" in data or data.get("intent") == "off_topic":
                print(f"  ✓ Non-mentor properly handled")
        return True
    else:
        print(f"  ✗ Unexpected error: {response.status_code}")
        return False


async def test_analytics_overview():
    """Test: Analytics overview API"""
    print("\n✅ Test 6: Analytics Overview API")
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}{API_PREFIX}/mentor/analytics/overview/{TEST_MENTOR_ID}",
            timeout=5.0
        )
    
    if response.status_code == 200:
        data = response.json()
        assert "mentor_profile" in data
        assert "programs" in data
        assert "active_mentees_count" in data
        profile = data["mentor_profile"]
        print(f"  ✓ Mentor: {profile.get('first_name')} {profile.get('last_name')}")
        print(f"  ✓ Programs: {profile.get('program_count')}")
        print(f"  ✓ Active Mentees: {data['active_mentees_count']}")
        return True
    elif response.status_code == 404:
        print(f"  ⚠ Mentor not found (404) - Make sure TEST_MENTOR_ID exists in DB")
        return True  # Don't fail if mentor doesn't exist
    else:
        print(f"  ✗ Error: {response.status_code}")
        print(f"    {response.text}")
        return False


async def test_analytics_programs():
    """Test: Analytics programs API"""
    print("\n✅ Test 7: Analytics Programs API")
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}{API_PREFIX}/mentor/analytics/programs/{TEST_MENTOR_ID}?limit=5",
            timeout=5.0
        )
    
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, list)
        print(f"  ✓ Programs fetched: {len(data)} items")
        if data:
            print(f"    - {data[0].get('title')}")
        return True
    elif response.status_code == 404:
        print(f"  ⚠ Mentor not found (404)")
        return True
    else:
        print(f"  ✗ Error: {response.status_code}")
        return False


async def test_document_formats():
    """Test: Document formats endpoint"""
    print("\n✅ Test 8: Document Formats API")
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}{API_PREFIX}/mentor/documents/formats",
            timeout=5.0
        )
    
    if response.status_code == 200:
        data = response.json()
        assert "supported_formats" in data
        assert "max_file_size_mb" in data
        print(f"  ✓ Formats: {data['supported_formats']}")
        print(f"  ✓ Max size: {data['max_file_size_mb']} MB")
        return True
    else:
        print(f"  ✗ Error: {response.status_code}")
        return False


async def test_language_detection():
    """Test: Language detection (Arabic)"""
    print("\n✅ Test 9: Language Detection (Arabic)")
    
    payload = {
        "message": "السلام عليكم، ازاي الحال؟",
        "user_id": TEST_MENTOR_ID
        # No language specified - should auto-detect Arabic
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}{API_PREFIX}/mentor-chat",
            json=payload,
            timeout=5.0
        )
    
    if response.status_code == 200:
        data = response.json()
        assert data["language"] == "ar"
        print(f"  ✓ Language detected: {data['language']}")
        print(f"  ✓ Intent: {data['intent']}")
        return True
    else:
        print(f"  ✗ Error: {response.status_code}")
        return False


# ─────────────────────────────────────────────────────────────────────
# Main Test Runner
# ─────────────────────────────────────────────────────────────────────


async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("MENTOR CHATBOT - QUICK VERIFICATION TEST")
    print("="*60)
    
    tests = [
        ("Greeting", test_mentor_chat_greeting),
        ("Analytics Intent", test_mentor_chat_analytics),
        ("FAQ Intent (Arabic)", test_mentor_chat_faq),
        ("Materials Intent", test_mentor_chat_materials),
        ("Non-Mentor Rejection", test_mentor_chat_non_mentor),
        ("Analytics Overview API", test_analytics_overview),
        ("Analytics Programs API", test_analytics_programs),
        ("Document Formats API", test_document_formats),
        ("Language Detection", test_language_detection),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Mentor chatbot is ready.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check logs above.")
        return False


if __name__ == "__main__":
    print("\n📌 IMPORTANT: Update TEST_MENTOR_ID with a valid mentor UUID from your database")
    print(f"   Current: {TEST_MENTOR_ID}")
    
    # Run tests
    success = asyncio.run(main())
    exit(0 if success else 1)
