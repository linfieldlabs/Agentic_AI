import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ["GROQ_API_KEY"]

# llama-4-scout supports tool calling correctly on Groq
model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct", groq_api_key=api_key
)


# --- State schema ---


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# --- Tools ---


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


tools = [search]
model_with_tools = model.bind_tools(tools)
tools_by_name = {t.name: t for t in tools}


# --- Nodes ---


def call_model(state: AgentState):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def call_tools(state: AgentState):
    last_message = state["messages"][-1]
    results = []
    for call in last_message.tool_calls:
        result = tools_by_name[call["name"]].invoke(call["args"])
        results.append(ToolMessage(content=result, tool_call_id=call["id"]))
    return {"messages": results}


def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


# --- Build graph ---

graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", call_tools)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")

app = graph.compile()

result = app.invoke(
    {"messages": [{"role": "user", "content": "Search for LangGraph v1 features"}]}
)
print(result["messages"][-1].content)
