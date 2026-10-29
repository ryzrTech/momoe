# LiteLLM integration for the local MoMoE / Ornith server

The MoMoE server exposes an OpenAI-compatible API at
`http://127.0.0.1:8080/v1`. It currently requires **no API key** (`api_key: false`), so any value works as the
placeholder `api_key`. This file wires an existing LiteLLM gateway at that endpoint.

## 1. `config.yaml` for LiteLLM (proxy mode)

```yaml
model_list:
  - model_name: ornith-1.5-35b-a3b          # the name your apps request
    litellm_params:
      model: openai/ornith-1.5-35b-a3b      # "openai/" routes to the OpenAI-compatible client
      base_url: http://127.0.0.1:8080/v1    # MoMoE server (localhost only - do not expose)
      api_key: "momoe-local"                # placeholder; server api_key is currently disabled

general_settings:
  # keep it bound to this host unless you add authenticated LAN access (see INSTALL_REPORT.md)
```

Start the gateway: `litellm --config config.yaml --port 4000`
then point clients at `http://127.0.0.1:4000/v1` with `model: ornith-1.5-35b-a3b`.

## 2. Working test request

This is exactly what LiteLLM forwards to the MoMoE server. Run it directly against the server to confirm the
endpoint is reachable (no key needed):

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ornith-1.5-35b-a3b",
    "messages": [{"role": "user", "content": "In one sentence, what is a mixture of experts?"}],
    "max_tokens": 64,
    "reasoning_effort": "none"
  }'
```

With an API key configured on the server, add: `-H "Authorization: Bearer <key>"`.

## 3. Tool-calling request (structured function calling)

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ornith-1.5-35b-a3b",
    "messages": [{"role": "user", "content": "What is the weather in Tokyo right now?"}],
    "tools": [{"type": "function", "function": {
      "name": "get_weather",
      "description": "Get current weather for a city",
      "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
    "max_tokens": 150
  }'
```

Expected: a `tool_calls` array with `function.name == "get_weather"` and JSON `arguments`.

## 4. Notes

- The model answers through `reasoning_content` (it is a thinking model); the final text is in `message.content`.
- `reasoning_effort` accepts `none | low | medium | high` (default high).
- Do not expose `:8080` publicly. If the LiteLLM gateway needs to reach it from another host, use authenticated
  LAN access (`momoe start ornith --host 0.0.0.0 --api-key <secret> --save`) or an authenticated reverse proxy —
  never an unauthenticated administrative/inference API.
