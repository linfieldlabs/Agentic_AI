"""Example: LangGraph StateGraph for Complex Workflows.

This example demonstrates LangGraph v1 StateGraph for building
complex agent workflows with branches, loops, and state management.
"""
import os
from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# Load environment variables
load_dotenv()

def get_groq_api_key():
    """Retrieve the Groq API key from environment variables."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables. "
                         "Please ensure it is set in your .env file.")
    return api_key


# Define state schema using TypedDict
class AgentState(TypedDict):
    """State schema for the agent workflow.
    
    Annotated[List[BaseMessage], add] tells LangGraph to append new messages
    to the list rather than overwriting it.
    """
    messages: Annotated[List[BaseMessage], add]
    step_count: int
    user_query: str
    result: str


def process_query(state: AgentState) -> Dict[str, Any]:
    """Node to process the user query and increment count."""
    print(f"[Node: process_query] Processing query...")
    # Just need to return the fields we want to update
    return {
        "step_count": state["step_count"] + 1,
        "messages": [HumanMessage(content=state["user_query"])]
    }


def generate_response(state: AgentState) -> Dict[str, Any]:
    """Node to generate AI response using the model."""
    print("[Node: generate_response] Generating response...")
    
    api_key = get_groq_api_key()
    model = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=api_key,
        temperature=0.7
    )
    
    # Generate response from all messages in state
    response = model.invoke(state["messages"])
    
    print(f"[Node: generate_response] Response generated: {response.content[:100]}...")
    
    return {
        "messages": [response],
        "result": response.content,
        "step_count": state["step_count"] + 1
    }


def should_continue(state: AgentState) -> str:
    """Conditional edge logic."""
    # Loop back if we haven't reached a "threshold" (demonstration of branching)
    if state["step_count"] < 3:
        print(f"[Condition: should_continue] Continuing (step_count={state['step_count']})")
        return "continue"
    else:
        print(f"[Condition: should_continue] Ending (step_count={state['step_count']})")
        return "end"


def finalize(state: AgentState) -> Dict[str, Any]:
    """Node to finalize the workflow."""
    print("[Node: finalize] Finalizing workflow...")
    return {"step_count": state["step_count"] + 1}


def main():
    """Run the LangGraph example."""
    print("=" * 60)
    print("LangGraph Example: StateGraph Workflow")
    print("=" * 60)
    
    # Create the StateGraph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("process_query", process_query)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("finalize", finalize)
    
    # Set entry point
    workflow.set_entry_point("process_query")
    
    # Add logical flow
    workflow.add_edge("process_query", "generate_response")
    
    # Add conditional edge from generate_response
    workflow.add_conditional_edges(
        "generate_response",
        should_continue,
        {
            "continue": "generate_response",  # Loop back
            "end": "finalize"
        }
    )
    
    # Add final edge
    workflow.add_edge("finalize", END)
    
    # Compile the graph
    app = workflow.compile()
    
    # Run the workflow
    initial_input = {
        "messages": [],
        "step_count": 0,
        "user_query": "Briefly explain what LangGraph is and how it differs from LangChain",
        "result": ""
    }
    
    print(f"\nInitial query: {initial_input['user_query']}")
    print("\nExecuting workflow...\n")
    
    try:
        # Execute the graph
        final_state = app.invoke(initial_input)
        
        print("\n" + "=" * 60)
        print("Workflow Execution Complete")
        print("=" * 60)
        print(f"Total steps: {final_state['step_count']}")
        print(f"Total messages in state: {len(final_state['messages'])}")
        print(f"\nFinal result summary:")
        print(final_state['result'][:300] + "...")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError executing workflow: {e}")


if __name__ == "__main__":
    main()
