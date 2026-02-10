"""
LangGraph Example 1: State Management
======================================
This example demonstrates LangGraph's explicit state management using TypedDict
and the add_messages reducer.

Key Concepts:
- TypedDict: Defines the state schema
- Annotated: Adds metadata to fields (e.g., reducers)
- add_messages: Built-in reducer for message lists
- Explicit state flow
"""

from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage


# Define the state schema
class AgentState(TypedDict):
    """State that flows through the graph.
    
    Attributes:
        messages: List of conversation messages (with add_messages reducer)
        iteration: Counter for tracking iterations
        user_name: Example of additional state
    """
    messages: Annotated[list, add_messages]
    iteration: int
    user_name: str


def agent_node(state: AgentState) -> dict:
    """Process the current state and generate a response.
    
    Args:
        state: Current agent state
        
    Returns:
        Dictionary with updated state fields
    """
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7
    )
    
    # Get the last user message
    last_message = state["messages"][-1]
    
    # Create a system message with context
    system_context = f"You are a helpful assistant talking to {state['user_name']}."
    
    # Invoke the LLM
    response = llm.invoke([
        {"role": "system", "content": system_context},
        *state["messages"]
    ])
    
    # Return updated state
    return {
        "messages": [response],
        "iteration": state["iteration"] + 1
    }


def main():
    # Create the state graph
    workflow = StateGraph(AgentState)
    
    # Add the agent node
    workflow.add_node("agent", agent_node)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add edge to end
    workflow.add_edge("agent", END)
    
    # Compile the graph
    app = workflow.compile()
    
    # Initial state
    initial_state = {
        "messages": [HumanMessage(content="Hello! What's the weather like today?")],
        "iteration": 0,
        "user_name": "Alice"
    }
    
    # Run the graph
    print("=" * 60)
    print("LANGGRAPH STATE MANAGEMENT EXAMPLE")
    print("=" * 60)
    print(f"\nUser: {initial_state['user_name']}")
    print(f"Initial iteration: {initial_state['iteration']}")
    print(f"\nUser message: {initial_state['messages'][0].content}")
    
    result = app.invoke(initial_state)
    
    print(f"\nAI response: {result['messages'][-1].content}")
    print(f"\nFinal iteration: {result['iteration']}")
    print(f"Total messages in state: {len(result['messages'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
