from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
import orjson
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter


class RetryableStatus(Exception):
    pass


class RequestLimiter:
    def __init__(self, rate_per_second: int) -> None:
        self._min_interval = 1.0 / max(rate_per_second, 1)
        self._lock = asyncio.Lock()
        self._last_request = 0.0

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait_for = self._min_interval - (now - self._last_request)
            if wait_for > 0:
                await asyncio.sleep(wait_for)
            self._last_request = time.monotonic()


async def _parse_json(response: httpx.Response) -> Any:
    try:
        return orjson.loads(response.content)
    except orjson.JSONDecodeError:
        return response.json()


def build_request_info(url: str, params: dict[str, Any] | None, cursor: str | None = None) -> dict[str, Any]:
    return {
        "url": url,
        "params": params or {},
        "headers_redacted": {},
        "cursor": cursor,
    }


async def fetch_json(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    params: dict[str, Any] | None,
    limiter: RequestLimiter,
    max_retries: int,
) -> Any:
    @retry(
        retry=retry_if_exception_type((httpx.RequestError, RetryableStatus)),
        wait=wait_exponential_jitter(initial=1, max=30),
        stop=stop_after_attempt(max_retries),
        reraise=True,
    )
    async def _do_request() -> Any:
        await limiter.wait()
        response = await client.request(method, url, params=params)
        if response.status_code in {408, 429} or response.status_code >= 500:
            raise RetryableStatus(f"retryable status {response.status_code}")
        response.raise_for_status()
        return await _parse_json(response)

    return await _do_request()
