"""
LangGraph Example 3: Streaming and Inspection
==============================================
This example demonstrates LangGraph's streaming capabilities and state inspection,
which are crucial for debugging and monitoring.

Key Concepts:
- Streaming: Real-time state updates
- State inspection: Visibility into each step
- Debugging: Understanding the flow
"""

from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import json


class StreamState(TypedDict):
    """State for streaming example."""
    messages: Annotated[list, add_messages]
    step: str
    data: dict


def step_1_collect(state: StreamState) -> dict:
    """First step: Collect information."""
    print("  [Step 1] Collecting information...")
    
    return {
        "step": "collect",
        "data": {"collected": True, "items": ["item1", "item2", "item3"]}
    }


def step_2_process(state: StreamState) -> dict:
    """Second step: Process the data."""
    print("  [Step 2] Processing data...")
    
    items = state["data"].get("items", [])
    processed = [f"processed_{item}" for item in items]
    
    return {
        "step": "process",
        "data": {**state["data"], "processed": processed}
    }


def step_3_generate(state: StreamState) -> dict:
    """Third step: Generate response."""
    print("  [Step 3] Generating response...")
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7
    )
    
    processed_items = state["data"].get("processed", [])
    
    response = llm.invoke([
        {"role": "system", "content": "Summarize the processed items."},
        {"role": "user", "content": f"Items: {', '.join(processed_items)}"}
    ])
    
    return {
        "messages": [response],
        "step": "generate",
        "data": {**state["data"], "final": True}
    }


def main():
    # Create workflow
    workflow = StateGraph(StreamState)
    
    # Add nodes
    workflow.add_node("collect", step_1_collect)
    workflow.add_node("process", step_2_process)
    workflow.add_node("generate", step_3_generate)
    
    # Set up the flow
    workflow.set_entry_point("collect")
    workflow.add_edge("collect", "process")
    workflow.add_edge("process", "generate")
    workflow.add_edge("generate", END)
    
    # Compile
    app = workflow.compile()
    
    # Initial state
    initial_state = {
        "messages": [HumanMessage(content="Process my data")],
        "step": "start",
        "data": {}
    }
    
    print("=" * 60)
    print("LANGGRAPH STREAMING EXAMPLE")
    print("=" * 60)
    print("\nStreaming state updates:\n")
    
    # Stream the execution
    for i, event in enumerate(app.stream(initial_state), 1):
        print(f"\n--- Event {i} ---")
        
        # Each event is a dict with node name as key
        for node_name, node_state in event.items():
            print(f"Node: {node_name}")
            print(f"Current step: {node_state.get('step', 'N/A')}")
            
            # Show data state
            if "data" in node_state:
                print(f"Data state: {json.dumps(node_state['data'], indent=2)}")
            
            # Show messages if present
            if "messages" in node_state and node_state["messages"]:
                last_msg = node_state["messages"][-1]
                if hasattr(last_msg, 'content'):
                    print(f"Latest message: {last_msg.content[:100]}...")
    

    print("=" * 60)


if __name__ == "__main__":
    main()
