import json
import logging
import os
from typing import Annotated, Literal, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from rag.memory import SessionMemory
from rag.tools import ALL_TOOLS

logger = logging.getLogger(__name__)

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o")

_SYSTEM_PROMPT = """You are an expert legal document analyst assistant for EaseSign.

You have access to these tools — use them proactively based on user intent:
- rag_search_tool            → answer factual questions about document content
- risk_analysis_tool         → when user asks about risks or "is this safe to sign?"
- key_clause_extraction_tool → when user asks about key clauses or document structure
- red_flag_scanner_tool      → when user asks about warnings or unusual terms
- document_compare_tool      → when user wants to compare two documents

Rules:
1. For content questions, always use rag_search_tool first.
2. Cite specific clauses or sections in your answers.
3. You may call multiple tools in one turn if the user's question requires it.
4. If document_id is provided, always pass it to the tool.
5. Be concise, precise, and professional about legal terminology.

Session context: session={session_id} | document={document_id}
"""

# Build a name→tool lookup once at module load
_TOOL_MAP: dict[str, BaseTool] = {t.name: t for t in ALL_TOOLS}


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    session_id: str
    document_id: str | None
    user_id: str


# ── Custom tool executor (no ToolNode dependency) ─────────────────────────────

async def _execute_tool_calls(ai_message: AIMessage) -> list[ToolMessage]:
    """Execute every tool_call in an AIMessage and return ToolMessages."""
    results = []
    for tc in ai_message.tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        tool_id = tc["id"]

        tool_fn = _TOOL_MAP.get(tool_name)
        if tool_fn is None:
            content = json.dumps({"error": f"Tool '{tool_name}' not found."})
        else:
            try:
                content = await tool_fn.ainvoke(tool_args)
                if not isinstance(content, str):
                    content = json.dumps(content)
            except Exception as exc:
                logger.exception(f"Tool '{tool_name}' raised an exception")
                content = json.dumps({"error": str(exc)})

        results.append(ToolMessage(content=content, tool_call_id=tool_id, name=tool_name))

    return results


# ── Graph nodes ───────────────────────────────────────────────────────────────

def _build_graph():
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0, streaming=True)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    def agent_node(state: AgentState):
        system_content = _SYSTEM_PROMPT.format(
            session_id=state["session_id"],
            document_id=state.get("document_id") or "none",
        )
        messages = [SystemMessage(content=system_content)] + list(state["messages"])
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    async def tool_node(state: AgentState):
        last = state["messages"][-1]
        tool_messages = await _execute_tool_calls(last)
        return {"messages": tool_messages}

    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    return graph.compile()


_graph = _build_graph()


# ── Public API ────────────────────────────────────────────────────────────────

async def chat(
    user_message: str,
    session_id: str,
    user_id: str,
    document_id: str | None = None,
) -> dict:
    memory = SessionMemory(session_id)
    history = await memory.load()

    state: AgentState = {
        "messages": history + [HumanMessage(content=user_message)],
        "session_id": session_id,
        "document_id": document_id,
        "user_id": user_id,
    }

    result = await _graph.ainvoke(state)
    final_messages = list(result["messages"])

    await memory.save(final_messages)

    ai_response = next(
        (
            m for m in reversed(final_messages)
            if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None)
        ),
        None,
    )
    return {
        "response": ai_response.content if ai_response else "",
        "session_id": session_id,
        "document_id": document_id,
    }


async def stream_chat(
    user_message: str,
    session_id: str,
    user_id: str,
    document_id: str | None = None,
):
    """Async generator — yields text tokens for SSE streaming."""
    memory = SessionMemory(session_id)
    history = await memory.load()

    state: AgentState = {
        "messages": history + [HumanMessage(content=user_message)],
        "session_id": session_id,
        "document_id": document_id,
        "user_id": user_id,
    }

    final_messages: list[BaseMessage] = []

    async for event in _graph.astream_events(state, version="v2"):
        kind = event.get("event")
        if kind == "on_chat_model_stream":
            chunk = event["data"].get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                yield chunk.content
        elif kind == "on_chain_end" and event.get("name") == "LangGraph":
            output = event["data"].get("output", {})
            if "messages" in output:
                final_messages = list(output["messages"])

    if final_messages:
        await memory.save(final_messages)
    else:
        await memory.save(list(state["messages"]))
