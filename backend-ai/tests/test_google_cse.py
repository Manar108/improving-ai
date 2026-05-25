#!/usr/bin/env python3
"""Test Google CSE API directly."""

import httpx
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
CSE_ID = os.getenv('CSE_ID')

print(f'API Key: {GOOGLE_API_KEY[:20] if GOOGLE_API_KEY else "MISSING"}...')
print(f'CSE ID: {CSE_ID if CSE_ID else "MISSING"}')

async def test():
    if not GOOGLE_API_KEY or not CSE_ID:
        print("ERROR: Missing credentials!")
        return
    
    # Test 1: Simple query without site filters
    print("\n=== TEST 1: Simple query ===")
    params = {
        'key': GOOGLE_API_KEY,
        'cx': CSE_ID,
        'q': 'machine learning',
        'num': 8,
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get('https://www.googleapis.com/customsearch/v1', params=params)
    
    print(f'Status: {resp.status_code}')
    data = resp.json()
    
    if 'error' in data:
        print(f"ERROR: {data['error']}")
        return
    
    items = data.get('items', [])
    print(f'Total results: {len(items)}')
    for i, item in enumerate(items[:3]):
        print(f'{i+1}. {item.get("title", "N/A")[:60]}')
        print(f'   {item.get("link", "N/A")[:70]}')
    
    # Test 2: Query with site filters (materials sites)
    print("\n=== TEST 2: With site filters ===")
    sites = ["freecodecamp.org", "medium.com", "dev.to", "github.com", "geeksforgeeks.org"]
    site_filter = " OR ".join(f"site:{s}" for s in sites)
    query = f"machine learning {site_filter}"
    
    params = {
        'key': GOOGLE_API_KEY,
        'cx': CSE_ID,
        'q': query,
        'num': 8,
    }
    print(f"Query: {query[:100]}...")
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get('https://www.googleapis.com/customsearch/v1', params=params)
    
    data = resp.json()
    items = data.get('items', [])
    print(f'Total results: {len(items)}')
    for i, item in enumerate(items[:3]):
        print(f'{i+1}. {item.get("title", "N/A")[:60]}')
        print(f'   {item.get("link", "N/A")[:70]}')

asyncio.run(test())
