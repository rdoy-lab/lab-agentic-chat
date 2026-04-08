"""
Web-based LLM chat app built with LangGraph + FastAPI, instrumented with Arize Phoenix OTEL.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python src/server.py
"""

import json
import os
from typing import Annotated

from fastapi import FastAPI
from fastapi.responses import FileResponse
from datetime import date

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from sse_starlette.sse import EventSourceResponse
from typing_extensions import TypedDict

from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

# ---------------------------------------------------------------------------
# 1. Phoenix OTEL setup
# ---------------------------------------------------------------------------
_endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")
if _endpoint and not _endpoint.rstrip("/").endswith("/v1/traces"):
    _endpoint = _endpoint.rstrip("/") + "/v1/traces"

tracer_provider = register(
    project_name="lab-agentic-chat",
    endpoint=_endpoint,
    api_key=os.environ.get("PHOENIX_API_KEY"),
    protocol="http/protobuf",
    batch=True,
    verbose=True,
)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

# ---------------------------------------------------------------------------
# 2. Tools
# ---------------------------------------------------------------------------
wikipedia_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=3,
        doc_content_chars_max=2000,
    )
)


@tool
def get_today_date() -> str:
    """Returns today's date in YYYY-MM-DD format."""
    return date.today().isoformat()


tools = [wikipedia_tool, get_today_date]

# ---------------------------------------------------------------------------
# 3. LangGraph state & graph definition
# ---------------------------------------------------------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOpenAI(
    model=os.environ["OPENROUTER_MODEL"],
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    streaming=True,
).bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph = graph_builder.compile()

# ---------------------------------------------------------------------------
# 4. FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI()


@app.get("/")
async def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


@app.get("/chat")
def chat(message: str):
    def _sse(type_, **kwargs):
        return {"data": json.dumps({"type": type_, **kwargs})}

    def event_stream():
        try:
            for event in graph.stream(
                {"messages": [{"role": "user", "content": message}]}
            ):
                for node, value in event.items():
                    if not isinstance(value, dict) or "messages" not in value:
                        continue
                    msgs = value["messages"]
                    msg = msgs[-1]
                    content = msg.content if hasattr(msg, "content") else ""
                    if node == "tools":
                        for tc in content if isinstance(content, list) else [content]:
                            yield _sse("tool_result", content=str(tc)[:500])
                    elif hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            yield _sse("tool_call", name=tc["name"], args=tc["args"])
                    elif content:
                        text = json.dumps(content) if isinstance(content, list) else str(content)
                        yield _sse("assistant", content=text)
        except Exception as exc:
            yield _sse("error", content=str(exc))
        yield _sse("done")

    return EventSourceResponse(event_stream())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
