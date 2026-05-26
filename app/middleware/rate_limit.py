"""
Rate limiting middleware.
Uses Redis sliding window in production.
Falls back to in-memory counter if Redis is unavailable (dev mode).
"""

import time
from collections import defaultdict
from typing import Callable, Dict, Tuple

import structlog
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings

log = structlog.get_logger()

# In-memory fallback store: {client_ip: [(timestamp, count)]}
_memory_store: Dict[str, list] = defaultdict(list)

# More lenient limits for public endpoints
PUBLIC_LIMIT = 10  # req/min for /predict/public
DEFAULT_LIMIT = settings.RATE_LIMIT_REQUESTS  # req/min for everything else


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        path = request.url.path

        # Skip rate limiting for health/metrics endpoints
        if path in {"/health", "/health/live", "/health/ready", "/metrics"}:
            return await call_next(request)

        limit = PUBLIC_LIMIT if "public" in path else DEFAULT_LIMIT
        allowed, remaining = await self._check_rate_limit(client_ip, limit)

        if not allowed:
            log.warning("rate_limit.exceeded", ip=client_ip, path=path)
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again in a moment.", "type": "RateLimitError"},
                headers={
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "Retry-After": str(settings.RATE_LIMIT_WINDOW_SECONDS),
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response

    async def _check_rate_limit(self, client_ip: str, limit: int) -> Tuple[bool, int]:
        """Sliding window rate limit — prefer Redis, fall back to memory."""
        try:
            return await self._redis_check(client_ip, limit)
        except Exception:
            return self._memory_check(client_ip, limit)

    async def _redis_check(self, client_ip: str, limit: int) -> Tuple[bool, int]:
        import redis.asyncio as aioredis
        client = aioredis.from_url(settings.REDIS_URL, socket_connect_timeout=0.5)
        key = f"ratelimit:{client_ip}"
        window = settings.RATE_LIMIT_WINDOW_SECONDS
        now = time.time()

        async with client.pipeline(transaction=True) as pipe:
            pipe.zremrangebyscore(key, 0, now - window)
            pipe.zcard(key)
            pipe.zadd(key, {str(now): now})
            pipe.expire(key, window)
            results = await pipe.execute()

        count = results[1]
        await client.aclose()
        return count < limit, max(0, limit - count - 1)

    def _memory_check(self, client_ip: str, limit: int) -> Tuple[bool, int]:
        now = time.time()
        window = settings.RATE_LIMIT_WINDOW_SECONDS
        timestamps = _memory_store[client_ip]
        # Remove expired entries
        _memory_store[client_ip] = [t for t in timestamps if now - t < window]
        count = len(_memory_store[client_ip])
        if count >= limit:
            return False, 0
        _memory_store[client_ip].append(now)
        return True, limit - count - 1
