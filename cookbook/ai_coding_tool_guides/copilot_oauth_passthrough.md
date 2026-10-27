# Use Copilot OAuth through LiteLLM

The official Copilot CLI or SDK keeps its GitHub OAuth login and sends native inference requests through LiteLLM. LiteLLM validates the bearer with `GET https://api.github.com/copilot_internal/user` and loads the provisioned LiteLLM user whose `user_id` matches the returned GitHub `login`. No separate LiteLLM key or server-side GitHub login is needed for these inference requests

This requires a client build containing managed inference routing from [github/copilot-agent-runtime#24482](https://github.com/github/copilot-agent-runtime/pull/24482). Do not assume an older released CLI supports it. Authentication, model discovery, and session-token requests stay on the client's normal Copilot endpoints

## Configure the proxy

Merge these exact pass-through endpoints into the proxy configuration and restart it. The prefix is configurable, but it must match the client's inference URL. Keep `auth: true`, `forward_headers: true`, and the three explicit POST routes. Wildcard routes are not supported by this authentication parser

```yaml
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: os.environ/DATABASE_URL
  pass_through_endpoints:
    - path: /copilot/chat/completions
      target: https://api.githubcopilot.com/chat/completions
      methods: [POST]
      auth: true
      custom_auth_parser: github_copilot
      forward_headers: true
    - path: /copilot/v1/messages
      target: https://api.githubcopilot.com/v1/messages
      methods: [POST]
      auth: true
      custom_auth_parser: github_copilot
      forward_headers: true
    - path: /copilot/responses
      target: https://api.githubcopilot.com/responses
      methods: [POST]
      auth: true
      custom_auth_parser: github_copilot
      forward_headers: true
```

Use HTTPS between the client and a deployed proxy. These routes forward the client's bearer and applicable `Copilot-Session-Token` to the configured upstream. Do not replace Authorization with a shared credential or configure an untrusted target

Provision each caller through the existing user-management API or dashboard. Replace `GITHUB_LOGIN` with the caller's GitHub login, not their email or numeric GitHub ID. For example, against a local proxy

```bash
curl --fail-with-body http://localhost:4000/user/new \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"GITHUB_LOGIN","user_role":"internal_user","auto_create_key":false}'
```

Set model permissions, budgets, and rate limits on that user through the usual management surfaces. Model permissions use the native Copilot model IDs, without LiteLLM model aliases. Provisioning is required even if GitHub accepts the token. The OAuth credential never inherits a provisioned user's administrative role

## Configure the client

Set the following managed runtime setting using the configuration mechanism supported by your compatible client build

```json
{
  "copilotInferenceUrl": "https://gateway.example/copilot"
}
```

SDK sessions must also opt into managed settings with `enableManagedSettings: true`. Keep the client's normal GitHub OAuth authentication. Do not use the older `COPILOT_API_URL` override, which also changes control-plane routing

HTTP and SSE are supported. Native WebSocket forwarding is not included. The compatible runtime falls back from a failed WebSocket Responses upgrade to HTTP on the same gateway. Do not set `COPILOT_CLI_DISABLE_WEBSOCKET_HTTP_FALLBACK=true`. To select HTTP explicitly, launch the client with `COPILOT_CLI_DISABLE_WEBSOCKET_RESPONSES=true`

## Authentication and policy behavior

Successful token-to-login validation is cached locally for five minutes under a SHA-256 token identifier. Failures are not cached, and expired validation never falls back to a stale identity. GitHub-side revocation can therefore take up to five minutes to be recognized. LiteLLM resolves provisioning and policy separately on each request through its existing user cache and database lookup behavior

The proxy returns 401 for a missing, malformed, or rejected bearer and 403 for a validated login without a provisioned user. Invalid GitHub responses return 502. Validation outages, timeouts, rate limiting, or unavailable user policy return 503. These failures never fall back to a virtual key, shared provider account, or anonymous database-outage identity

Normal LiteLLM authentication is unchanged outside the opted-in endpoints. Master keys still protect management APIs. The diagnostic `X-Copilot-Inference-Gateway` header and caller-supplied identity headers do not authenticate a user

Check a completed request in LiteLLM's Logs page for the validated user ID. Authentication and logging use a hashed credential identifier, while credential-header copies redact both the OAuth bearer and Copilot session token. Native inference bodies, SSE streams, and upstream error responses retain the existing pass-through behavior

This integration does not add Copilot-specific billing. Existing user policy checks apply, but generic pass-through does not guarantee accurate Copilot dollar-cost accounting from native billing metadata. It also does not add a model catalog, an OAuth acquisition flow, or enterprise host discovery
