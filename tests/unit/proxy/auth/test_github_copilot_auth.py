from typing import Final

import httpx
import pytest
import respx
from fastapi import Request

from litellm.caching.in_memory_cache import InMemoryCache
from litellm.proxy.auth.github_copilot_auth import (
    CopilotAuthFailure,
    CopilotIdentity,
    copilot_auth_endpoint,
    resolve_copilot_identity,
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

_VALIDATION_URL: Final = "https://api.github.com/copilot_internal/user"


class _Clock:
    now: float = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.mark.asyncio
async def test_copilot_validation_forwards_only_identity_headers(respx_mock: respx.MockRouter) -> None:
    headers: Final = {
        "authorization": "Bearer client-oauth",
        "copilot-integration-id": "copilot-cli",
        "editor-version": "test-editor",
        "editor-plugin-version": "test-plugin",
        "copilot-api-version": "test-api",
        "user-agent": "test-client",
        "copilot-session-token": "private-session",
        "x-litellm-api-key": "private-gateway-key",
        "x-github-login": "forged-user",
        "x-copilot-inference-gateway": "true",
    }
    upstream: Final = respx_mock.get(_VALIDATION_URL).respond(json={"login": "  verified-user  ", "ignored": True})
    async with httpx.AsyncClient() as client:
        result: Final = await resolve_copilot_identity(headers, client=client, cache=UserApiKeyCache())
    assert result == CopilotIdentity(login="verified-user")
    sent: Final = upstream.calls.last.request
    for name in (
        "authorization", "copilot-integration-id", "editor-version", "editor-plugin-version",
        "copilot-api-version", "user-agent",
    ):
        assert sent.headers[name] == headers[name]
    assert sent.headers.keys().isdisjoint(
        {"copilot-session-token", "x-litellm-api-key", "x-github-login", "x-copilot-inference-gateway"}
    )
    assert sent.extensions["timeout"] == {"connect": 5.0, "read": 5.0, "write": 5.0, "pool": 5.0}


@pytest.mark.asyncio
async def test_copilot_cache_expires_without_accepting_stale_identity(respx_mock: respx.MockRouter) -> None:
    clock: Final = _Clock()
    cache: Final = UserApiKeyCache(in_memory_cache=InMemoryCache(clock=clock))
    upstream: Final = respx_mock.get(_VALIDATION_URL).respond(json={"login": "verified-user"})
    async with httpx.AsyncClient() as client:
        first: Final = await resolve_copilot_identity({"authorization": "Bearer first"}, client=client, cache=cache)
        clock.now = 299.0
        cached: Final = await resolve_copilot_identity({"authorization": "Bearer first"}, client=client, cache=cache)
        assert first == cached == CopilotIdentity(login="verified-user")
        assert upstream.call_count == 1
        upstream.respond(json={"login": "other-user"})
        other: Final = await resolve_copilot_identity({"authorization": "Bearer second"}, client=client, cache=cache)
        assert other == CopilotIdentity(login="other-user")
        assert upstream.call_count == 2
        clock.now = 301.0
        upstream.respond(401, text="do not expose this upstream body")
        expired: Final = await resolve_copilot_identity({"authorization": "Bearer first"}, client=client, cache=cache)
    assert isinstance(expired, CopilotAuthFailure)
    assert expired.status_code == 401
    assert upstream.call_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,body,expected",
    [
        (401, b"private upstream body", 401),
        (403, b"private upstream body", 401),
        (429, b"private upstream body", 503),
        (500, b"private upstream body", 503),
        (302, b"", 502),
        (200, b"not-json", 502),
        (200, b"{}", 502),
        (200, b'{"login": ""}', 502),
        (200, b'{"login": "   "}', 502),
        (200, b'{"login": 123}', 502),
    ],
)
async def test_copilot_invalid_validation_is_sanitized_and_never_cached(
    respx_mock: respx.MockRouter, status: int, body: bytes, expected: int
) -> None:
    upstream: Final = respx_mock.get(_VALIDATION_URL).respond(
        status, content=body, headers={"location": "https://untrusted.example/collect"}
    )
    cache: Final = UserApiKeyCache()
    async with httpx.AsyncClient(follow_redirects=True) as client:
        first: Final = await resolve_copilot_identity({"authorization": "Bearer private-oauth"}, client=client, cache=cache)
        second: Final = await resolve_copilot_identity({"authorization": "Bearer private-oauth"}, client=client, cache=cache)
    assert isinstance(first, CopilotAuthFailure)
    assert first.status_code == expected
    assert first == second
    assert upstream.call_count == 2
    assert "private" not in repr(first)
    assert all(call.request.url == _VALIDATION_URL for call in respx_mock.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [httpx.ReadTimeout, httpx.ConnectError])
async def test_copilot_transport_failures_are_retryable(
    respx_mock: respx.MockRouter, error: type[httpx.RequestError]
) -> None:
    upstream: Final = respx_mock.get(_VALIDATION_URL).mock(side_effect=error("private transport details"))
    cache: Final = UserApiKeyCache()
    async with httpx.AsyncClient() as client:
        first: Final = await resolve_copilot_identity({"authorization": "Bearer private-oauth"}, client=client, cache=cache)
        second: Final = await resolve_copilot_identity({"authorization": "Bearer private-oauth"}, client=client, cache=cache)
    assert isinstance(first, CopilotAuthFailure)
    assert first.status_code == 503
    assert first == second
    assert "private" not in repr(first)
    assert upstream.call_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("authorization", ["", "Basic abc", "Bearer", "Bearer ", "Bearer two tokens"])
async def test_copilot_requires_original_bearer(
    respx_mock: respx.MockRouter, authorization: str
) -> None:
    async with httpx.AsyncClient() as client:
        result: Final = await resolve_copilot_identity(
            {"authorization": authorization, "x-litellm-api-key": "Bearer forged"}, client=client, cache=UserApiKeyCache()
        )
    assert isinstance(result, CopilotAuthFailure)
    assert result.status_code == 401
    assert not respx_mock.calls


@pytest.mark.parametrize(
    "path,method,selected",
    [
        ("/inference/responses", "POST", True),
        ("/inference/responses", "GET", False),
        ("/inference/responses/extra", "POST", False),
        ("/inference/models", "POST", False),
        ("/user/new", "POST", False),
    ],
)
def test_copilot_parser_is_scoped_to_configured_path_and_method(path: str, method: str, selected: bool) -> None:
    request: Final = Request({"type": "http", "path": path, "method": method, "headers": []})
    endpoint: Final = copilot_auth_endpoint(
        request, path,
        ({
            "path": "/inference/responses",
            "target": "https://api.githubcopilot.com/responses",
            "auth": True,
            "methods": ["POST"],
            "custom_auth_parser": "github_copilot",
        },),
    )
    assert (endpoint is not None) == selected
