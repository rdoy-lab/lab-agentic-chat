# lab-agentic-chat

Web-based LLM chat app built with LangGraph, instrumented with Arize Phoenix OTEL. Includes a Wikipedia fact-checking tool.

## Setup

```bash
pip install -e .
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes | OpenRouter API key |
| `OPENROUTER_MODEL` | Yes | Model ID (e.g. `google/gemini-2.5-flash-preview`) |
| `PHOENIX_COLLECTOR_ENDPOINT` | No | Phoenix collector URL (default: `localhost:4317`) |
| `PHOENIX_API_KEY` | No | Phoenix API key |

## Run

```bash
export OPENROUTER_API_KEY=sk-or-...
export OPENROUTER_MODEL=google/gemini-2.5-flash-preview
python src/server.py
```

Open http://localhost:8000.

## Docker

```bash
docker build -t lab-agentic-chat .
docker run -p 8000:8000 \
  -e OPENROUTER_API_KEY=sk-or-... \
  -e OPENROUTER_MODEL=google/gemini-2.5-flash-preview \
  lab-agentic-chat
```

## Publish

Push to `main` triggers a GitHub Actions build that publishes to `ghcr.io` with version `0.{commit-count}`.
