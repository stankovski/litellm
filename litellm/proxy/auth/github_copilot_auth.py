from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Final, Literal, NoReturn

import httpx
from fastapi import HTTPException, Request
from pydantic import BaseModel, ConfigDict, StringConstraints, TypeAdapter, ValidationError
from typing_extensions import assert_never

from litellm.constants import GITHUB_COPILOT_DEFAULT_HOST
from litellm.llms.github_copilot.github_host import github_host_endpoints
from litellm.proxy._types import LitellmUserRoles, PassThroughGenericEndpoint, UserAPIKeyAuth
from litellm.proxy.auth.resolvers.grants import (
    GrantResolver,
    LookupDegraded,
    NotAMember,
    ResolvedGrants,
    TeamGone,
    UserGone,
    UserLookup,
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.spend_tracking.carried_budget_state import carry_team_and_user_budget_state
from litellm.types.proxy.auth.user_api_key_auth import hash_token

_VALIDATION_HEADERS: Final = frozenset(
    {"copilot-integration-id", "editor-version", "editor-plugin-version", "copilot-api-version", "user-agent"}
)
_METHODS: Final = TypeAdapter[tuple[str, ...] | None](tuple[str, ...] | None)
_ENDPOINTS: Final = TypeAdapter[tuple[Mapping[str, object], ...] | None](tuple[Mapping[str, object], ...] | None)
_POLICY_FIELDS: Final[Mapping[str, bool]] = MappingProxyType({"models": True, "model_max_budget": True})


class CopilotIdentity(BaseModel):
    model_config = ConfigDict(frozen=True)

    login: Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]


class _UserModelPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    models: tuple[str, ...]
    model_max_budget: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class CopilotAuthFailure:
    status_code: Literal[401, 403, 500, 502, 503]
    message: str


def _matches_endpoint(request: Request, route: str, endpoint: Mapping[str, object]) -> bool:
    if endpoint.get("custom_auth_parser") != "github_copilot":
        return False
    path: Final = endpoint.get("path")
    if not isinstance(path, str):
        return False
    if route != path and not (endpoint.get("include_subpath") is True and route.startswith(path.rstrip("/") + "/")):
        return False
    methods: Final = _METHODS.validate_python(endpoint.get("methods"))
    return not methods or request.method.upper() in methods


def copilot_auth_endpoint(request: Request, route: str, endpoints: object) -> PassThroughGenericEndpoint | None:
    parsed_endpoints: Final = _ENDPOINTS.validate_python(endpoints)
    endpoint: Final = next(
        (endpoint for endpoint in parsed_endpoints or () if _matches_endpoint(request, route, endpoint)), None
    )
    return PassThroughGenericEndpoint.model_validate(endpoint) if endpoint is not None else None


def copilot_credential_id(authorization: str) -> str:
    scheme, _, credential = authorization.partition(" ")
    candidate: Final = credential.strip() if scheme.lower() == "bearer" else authorization
    return f"copilot-sha256:{hash_token(candidate)}"


def _bearer(headers: Mapping[str, str]) -> str | CopilotAuthFailure:
    scheme, separator, token = headers.get("authorization", "").partition(" ")
    credential: Final = token.strip()
    if scheme.lower() != "bearer" or not separator or not credential or any(c.isspace() for c in credential):
        return CopilotAuthFailure(401, "A GitHub OAuth bearer token is required")
    return credential


async def resolve_copilot_identity(
    headers: Mapping[str, str],
    *,
    client: httpx.AsyncClient,
    cache: UserApiKeyCache,
    github_host: str = GITHUB_COPILOT_DEFAULT_HOST,
) -> CopilotIdentity | CopilotAuthFailure:
    token: Final = _bearer(headers)
    if isinstance(token, CopilotAuthFailure):
        return token
    endpoints: Final = github_host_endpoints(github_host)
    cache_key: Final = f"{endpoints.host}:{copilot_credential_id(f'Bearer {token}')}"
    cached: Final = await cache.async_get_cache(key=cache_key, model_type=CopilotIdentity, local_only=True)
    if cached is not None:
        return cached
    forwarded: Final = {name: value for name, value in headers.items() if name.lower() in _VALIDATION_HEADERS}
    try:
        response: Final = await client.get(
            endpoints.copilot_user_url,
            headers={**forwarded, "Authorization": f"Bearer {token}"},
            timeout=5.0,
            follow_redirects=False,
        )
    except httpx.RequestError:
        return CopilotAuthFailure(503, "GitHub Copilot authentication is temporarily unavailable")
    if response.status_code in (401, 403):
        return CopilotAuthFailure(401, "GitHub Copilot rejected the OAuth credential")
    if response.status_code == 429 or response.status_code >= 500:
        return CopilotAuthFailure(503, "GitHub Copilot authentication is temporarily unavailable")
    if response.status_code != 200:
        return CopilotAuthFailure(502, "GitHub Copilot returned an invalid authentication response")
    try:
        identity: Final = CopilotIdentity.model_validate_json(response.content)
    except ValidationError:
        return CopilotAuthFailure(502, "GitHub Copilot returned an invalid authentication response")
    await cache.async_set_cache(key=cache_key, value=identity, ttl=300, local_only=True)
    return identity


async def authenticate_copilot_request(
    request: Request,
    *,
    endpoint: PassThroughGenericEndpoint,
    client: httpx.AsyncClient,
    cache: UserApiKeyCache,
    grants: GrantResolver,
) -> UserAPIKeyAuth | CopilotAuthFailure:
    if not endpoint.auth or "auth" not in endpoint.model_fields_set or endpoint.include_subpath:
        return CopilotAuthFailure(500, "Copilot authentication requires an authenticated, exact pass-through route")
    if len(request.headers.getlist("authorization")) != 1:
        return CopilotAuthFailure(401, "Exactly one GitHub OAuth bearer token is required")
    identity: Final = await resolve_copilot_identity(
        request.headers, client=client, cache=cache, github_host=endpoint.github_copilot_host
    )
    if isinstance(identity, CopilotAuthFailure):
        return identity
    user_id: Final = (
        identity.login
        if endpoint.github_copilot_host == GITHUB_COPILOT_DEFAULT_HOST
        else f"{endpoint.github_copilot_host}:{identity.login}"
    )
    resolved: Final = await grants.resolve(UserLookup(user_id=user_id), team_id=None)
    match resolved:
        case UserGone():
            return CopilotAuthFailure(403, "The GitHub user is not provisioned in LiteLLM")
        case LookupDegraded() | TeamGone() | NotAMember():
            return CopilotAuthFailure(503, "The LiteLLM user policy is temporarily unavailable")
        case ResolvedGrants(user_object=user):
            if user is None:
                return CopilotAuthFailure(403, "The GitHub user is not provisioned in LiteLLM")
            try:
                policy: Final = _UserModelPolicy.model_validate(user.model_dump(include=_POLICY_FIELDS))
            except ValidationError:
                return CopilotAuthFailure(503, "The LiteLLM user policy is temporarily unavailable")
            auth: Final = UserAPIKeyAuth(
                api_key=copilot_credential_id(request.headers["authorization"]),
                user_id=user_id,
                user_role=LitellmUserRoles.INTERNAL_USER,
                models=list(policy.models),
                allowed_routes=[endpoint.path],
                metadata={"allowed_passthrough_routes": [endpoint.path]},
                user_tpm_limit=user.tpm_limit,
                user_rpm_limit=user.rpm_limit,
                user_max_budget=user.max_budget,
                user_spend=user.spend,
                user_model_max_budget=policy.model_max_budget,
            )
            carry_team_and_user_budget_state(auth, team_object=None, user_object=user)
            return auth
        case _:
            assert_never(resolved)


def raise_copilot_auth_failure(failure: CopilotAuthFailure) -> NoReturn:
    raise HTTPException(status_code=failure.status_code, detail=failure.message)
