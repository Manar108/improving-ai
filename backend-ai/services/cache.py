"""Shared TTL cache — lightweight in-memory cache with per-key expiry.

Used by recommendation_service and program_recommendation_service
to avoid duplicating cache logic.
"""

import time as _time
from typing import Any


class TTLCache:
    """Simple in-memory cache with per-key TTL (seconds).

    Usage:
        cache = TTLCache(ttl_seconds=300)  # 5 minutes
        cache.set("key", value)
        result = cache.get("key")  # returns None if expired
    """

    def __init__(self, ttl_seconds: int = 300) -> None:
        self._ttl = ttl_seconds
        self._store: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        timestamp, value = entry
        if _time.monotonic() - timestamp > self._ttl:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (_time.monotonic(), value)

    def clear(self) -> None:
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)
