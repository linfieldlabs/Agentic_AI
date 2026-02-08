"""
LangGraph Example 2: Workflow with Conditional Edges
=====================================================
This example demonstrates a more complex workflow with conditional routing
based on the agent's decision.

Key Concepts:
- Conditional edges: Dynamic routing based on state
- Multiple nodes: Different processing steps
- Control flow: Explicit decision-making
"""

from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage


class WorkflowState(TypedDict):
    """State for the workflow."""
    messages: Annotated[list, add_messages]
    iteration: int
    needs_research: bool
    research_done: bool


def analyze_query(state: WorkflowState) -> dict:
    """Analyze if the query needs research."""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )
    
    last_message = state["messages"][-1].content
    
    # Ask LLM if research is needed
    analysis = llm.invoke([
        {"role": "system", "content": "Determine if this query needs external research. Reply only 'YES' or 'NO'."},
        {"role": "user", "content": last_message}
    ])
    
    needs_research = "yes" in analysis.content.lower()
    
    return {
        "needs_research": needs_research,
        "iteration": state["iteration"] + 1
    }


def research_node(state: WorkflowState) -> dict:
    """Simulate research step."""
    print("  → Performing research...")
    
    # Simulated research results
    research_result = AIMessage(
        content="[Research completed] Found relevant information about the topic."
    )
    
    return {
        "messages": [research_result],
        "research_done": True,
        "iteration": state["iteration"] + 1
    }


def generate_response(state: WorkflowState) -> dict:
    """Generate final response."""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7
    )
    
    context = "with research" if state.get("research_done", False) else "without research"
    
    response = llm.invoke([
        {"role": "system", "content": f"Generate a helpful response ({context})."},
        *state["messages"]
    ])
    
    return {
        "messages": [response],
        "iteration": state["iteration"] + 1
    }


def should_research(state: WorkflowState) -> Literal["research", "respond"]:
    """Conditional routing function."""
    if state.get("needs_research", False) and not state.get("research_done", False):
        return "research"
    return "respond"


def main():
    # Create the workflow
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("analyze", analyze_query)
    workflow.add_node("research", research_node)
    workflow.add_node("respond", generate_response)
    
    # Set entry point
    workflow.set_entry_point("analyze")
    
    # Add conditional edge from analyze
    workflow.add_conditional_edges(
        "analyze",
        should_research,
        {
            "research": "research",
            "respond": "respond"
        }
    )
    
    # Add edge from research to respond
    workflow.add_edge("research", "respond")
    
    # Add edge from respond to end
    workflow.add_edge("respond", END)
    
    # Compile
    app = workflow.compile()
    
    # Test with two different queries
    print("=" * 60)
    print("LANGGRAPH WORKFLOW EXAMPLE")
    print("=" * 60)
    
    # Query 1: Simple question (no research needed)
    print("\n--- Query 1: Simple Question ---")
    result1 = app.invoke({
        "messages": [HumanMessage(content="What is 2+2?")],
        "iteration": 0,
        "needs_research": False,
        "research_done": False
    })
    print(f"Response: {result1['messages'][-1].content}")
    print(f"Research performed: {result1.get('research_done', False)}")
    
    # Query 2: Complex question (needs research)
    print("\n--- Query 2: Complex Question ---")
    result2 = app.invoke({
        "messages": [HumanMessage(content="What are the latest developments in quantum computing?")],
        "iteration": 0,
        "needs_research": False,
        "research_done": False
    })
    print(f"Response: {result2['messages'][-1].content}")
    print(f"Research performed: {result2.get('research_done', False)}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
