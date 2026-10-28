from typing import Final

import httpx
import pytest
import respx
from fastapi import Request
from pydantic import ValidationError

from litellm.caching.in_memory_cache import InMemoryCache
from litellm.llms.github_copilot.github_host import github_host_endpoints
from litellm.proxy._types import PassThroughGenericEndpoint
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
@pytest.mark.parametrize("github_host", ["github.com", "octo.ghe.com"])
async def test_copilot_validation_forwards_only_identity_headers(
    respx_mock: respx.MockRouter, github_host: str
) -> None:
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
        "x-github-host": "untrusted.example",
        "x-copilot-inference-gateway": "true",
    }
    upstream: Final = respx_mock.get(github_host_endpoints(github_host).copilot_user_url).respond(
        json={"login": "  verified-user  ", "ignored": True}
    )
    async with httpx.AsyncClient() as client:
        result: Final = await resolve_copilot_identity(
            headers, client=client, cache=UserApiKeyCache(), github_host=github_host
        )
    assert result == CopilotIdentity(login="verified-user")
    sent: Final = upstream.calls.last.request
    for name in (
        "authorization", "copilot-integration-id", "editor-version", "editor-plugin-version",
        "copilot-api-version", "user-agent",
    ):
        assert sent.headers[name] == headers[name]
    assert sent.headers.keys().isdisjoint(
        {"copilot-session-token", "x-litellm-api-key", "x-github-login", "x-github-host", "x-copilot-inference-gateway"}
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
async def test_copilot_identity_cache_is_isolated_by_normalized_host(respx_mock: respx.MockRouter) -> None:
    headers: Final = {"authorization": "Bearer shared-oauth"}
    cache: Final = UserApiKeyCache()
    github: Final = respx_mock.get(github_host_endpoints("github.com").copilot_user_url).respond(
        json={"login": "github-user"}
    )
    enterprise: Final = respx_mock.get(github_host_endpoints("octo.ghe.com").copilot_user_url).respond(401)
    other_enterprise: Final = respx_mock.get(github_host_endpoints("other.ghe.com").copilot_user_url).respond(401)
    async with httpx.AsyncClient() as client:
        public_identity: Final = await resolve_copilot_identity(headers, client=client, cache=cache)
        rejected: Final = await resolve_copilot_identity(
            headers, client=client, cache=cache, github_host="octo.ghe.com"
        )
        assert isinstance(rejected, CopilotAuthFailure)
        assert rejected.status_code == 401
        enterprise.respond(json={"login": "enterprise-user"})
        enterprise_identity: Final = await resolve_copilot_identity(
            headers, client=client, cache=cache, github_host="OCTO.ghe.com"
        )
        cached_enterprise: Final = await resolve_copilot_identity(
            headers, client=client, cache=cache, github_host="octo.ghe.com"
        )
        cached_public: Final = await resolve_copilot_identity(headers, client=client, cache=cache)
        rejected_other: Final = await resolve_copilot_identity(
            headers, client=client, cache=cache, github_host="other.ghe.com"
        )
    assert public_identity == cached_public == CopilotIdentity(login="github-user")
    assert enterprise_identity == cached_enterprise == CopilotIdentity(login="enterprise-user")
    assert isinstance(rejected_other, CopilotAuthFailure)
    assert rejected_other.status_code == 401
    assert github.call_count == other_enterprise.call_count == 1
    assert enterprise.call_count == 2
    assert enterprise.calls.last.request.headers["authorization"] == headers["authorization"]


@pytest.mark.parametrize(
    "host",
    [
        "https://octo.ghe.com/api",
        "octo.ghe.com:443",
        "api.octo.ghe.com",
        "ghes.example.com",
        "octo.ghe.com.evil.example",
        "127.0.0.1",
        None,
    ],
)
def test_copilot_configuration_rejects_unsupported_validation_hosts(host: object) -> None:
    with pytest.raises(ValidationError, match="github_copilot_host"):
        PassThroughGenericEndpoint.model_validate({
            "path": "/inference/responses",
            "target": "https://api.githubcopilot.com/responses",
            "auth": True,
            "custom_auth_parser": "github_copilot",
            "github_copilot_host": host,
        })


@pytest.mark.asyncio
@pytest.mark.parametrize("github_host", ["github.com", "octo.ghe.com"])
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
    respx_mock: respx.MockRouter, status: int, body: bytes, expected: int, github_host: str
) -> None:
    validation_url: Final = github_host_endpoints(github_host).copilot_user_url
    upstream: Final = respx_mock.get(validation_url).respond(
        status, content=body, headers={"location": "https://untrusted.example/collect"}
    )
    cache: Final = UserApiKeyCache()
    async with httpx.AsyncClient(follow_redirects=True) as client:
        first: Final = await resolve_copilot_identity(
            {"authorization": "Bearer private-oauth"}, client=client, cache=cache, github_host=github_host
        )
        second: Final = await resolve_copilot_identity(
            {"authorization": "Bearer private-oauth"}, client=client, cache=cache, github_host=github_host
        )
    assert isinstance(first, CopilotAuthFailure)
    assert first.status_code == expected
    assert first == second
    assert upstream.call_count == 2
    assert "private" not in repr(first)
    assert all(call.request.url == validation_url for call in respx_mock.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("github_host", ["github.com", "octo.ghe.com"])
@pytest.mark.parametrize("error", [httpx.ReadTimeout, httpx.ConnectError])
async def test_copilot_transport_failures_are_retryable(
    respx_mock: respx.MockRouter, error: type[httpx.RequestError], github_host: str
) -> None:
    upstream: Final = respx_mock.get(github_host_endpoints(github_host).copilot_user_url).mock(
        side_effect=error("private transport details")
    )
    cache: Final = UserApiKeyCache()
    async with httpx.AsyncClient() as client:
        first: Final = await resolve_copilot_identity(
            {"authorization": "Bearer private-oauth"}, client=client, cache=cache, github_host=github_host
        )
        second: Final = await resolve_copilot_identity(
            {"authorization": "Bearer private-oauth"}, client=client, cache=cache, github_host=github_host
        )
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
