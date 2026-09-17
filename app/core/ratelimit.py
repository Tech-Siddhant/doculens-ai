"""Minimal in-process rate limiter.

Fixed-window, per-key (client IP) counters, protected by a single lock.
Suitable for a single-worker deployment; deliberately not a shared store.
"""

import threading
import time
from collections import defaultdict

from app.core.config import settings


class InMemoryRateLimiter:
    """Fixed-window per-key limiter.

    ponytail: a global lock and a dict of hit lists. Fine for a single-user
    app on one worker; if throughput or multi-worker fairness matters, move to
    a shared store (e.g. Redis INCR + EXPIRE) behind the same `allow()` API.
    """

    def __init__(self, max_requests: int = 60, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        window_start = now - self.window_seconds
        with self._lock:
            recent = [t for t in self._hits[key] if t > window_start]
            if len(recent) >= self.max_requests:
                self._hits[key] = recent
                return False
            recent.append(now)
            self._hits[key] = recent
            return True

    def clear(self) -> None:
        with self._lock:
            self._hits.clear()


_limiter = InMemoryRateLimiter()


def check_rate_limit(key: str) -> bool:
    """Allow/deny `key` using the current settings-derived limits."""
    _limiter.max_requests = settings.RATE_LIMIT_MAX_REQUESTS
    _limiter.window_seconds = settings.RATE_LIMIT_WINDOW_SECONDS
    return _limiter.allow(key)


def clear_rate_limits() -> None:
    """Reset all counters (used between tests and in manual reset tooling)."""
    _limiter.clear()