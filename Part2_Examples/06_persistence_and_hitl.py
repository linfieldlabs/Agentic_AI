import os
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

MODEL = "groq:meta-llama/llama-4-scout-17b-16e-instruct"


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


# --- Example 1: Persistence with MemorySaver ---

print("=== Example 1: Persistence with MemorySaver ===")

memory = MemorySaver()

agent = create_agent(
    model=MODEL,
    tools=[search],
    checkpointer=memory,
)

config = {"configurable": {"thread_id": "session-1"}}

# Turn 1
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Search for LangChain v1"}]},
    config=config,
)
print("Turn 1:", result["messages"][-1].content)

# Turn 2 — same thread_id continues the conversation
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What did you find?"}]},
    config=config,
)
print("Turn 2:", result["messages"][-1].content)


# --- Example 2: Human-in-the-loop with interrupt ---

print("\n=== Example 2: Human-in-the-loop (interrupt) ===")

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import ToolMessage
from typing import TypedDict, Annotated

api_key = os.environ["GROQ_API_KEY"]
model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct", groq_api_key=api_key
)
model_with_tools = model.bind_tools([search])
tools_by_name = {"search": search}


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


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


checkpointer = MemorySaver()

graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", call_tools)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")

# interrupt_before="tools" pauses execution before any tool call for human approval
app = graph.compile(checkpointer=checkpointer, interrupt_before=["tools"])

config_hitl = {"configurable": {"thread_id": "hitl-1"}}

# Step 1: run until interrupted
result = app.invoke(
    {"messages": [{"role": "user", "content": "Search for LangGraph persistence"}]},
    config=config_hitl,
)
print("Interrupted — pending tool calls:")
for msg in result["messages"]:
    if getattr(msg, "tool_calls", None):
        for call in msg.tool_calls:
            print(f"  {call['name']}({call['args']})")

# Step 2: human approves — resume by passing None
print("Resuming after human approval...")
result = app.invoke(None, config=config_hitl)
print("Final:", result["messages"][-1].content)
