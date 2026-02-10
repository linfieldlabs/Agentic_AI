"""
LangGraph Example 4: Composition of Basics
===========================================
This example demonstrates how to compose multiple LangGraph concepts together:
- Multiple nodes with different responsibilities
- State management across complex workflows
- Conditional routing based on state
- Combining all basic concepts into a cohesive system

Key Concepts:
- Node composition: Building complex graphs from simple nodes
- State accumulation: Tracking data across multiple steps
- Conditional logic: Dynamic routing based on state
- Error handling: Graceful failure management
"""

from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage


class ComposedState(TypedDict):
    """Comprehensive state for composed workflow."""
    messages: Annotated[list, add_messages]
    user_input: str
    analysis: dict
    processing_steps: list
    final_output: str
    error: str


def validate_input(state: ComposedState) -> dict:
    """Node 1: Validate and prepare input."""
    print("  [Node 1] Validating input...")
    
    user_input = state["user_input"]
    
    if len(user_input.strip()) < 5:
        return {
            "error": "Input too short",
            "processing_steps": state["processing_steps"] + ["validation_failed"]
        }
    
    return {
        "processing_steps": state["processing_steps"] + ["validation_passed"]
    }


def analyze_content(state: ComposedState) -> dict:
    """Node 2: Analyze the content."""
    print("  [Node 2] Analyzing content...")
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    analysis_result = llm.invoke([
        {"role": "system", "content": "Analyze this text and identify: topic, sentiment, complexity (simple/moderate/complex)"},
        {"role": "user", "content": state["user_input"]}
    ])
    
    # Simulate parsing the analysis
    analysis = {
        "topic": "technology",
        "sentiment": "positive",
        "complexity": "moderate"
    }
    
    return {
        "analysis": analysis,
        "processing_steps": state["processing_steps"] + ["analysis_complete"]
    }


def process_simple(state: ComposedState) -> dict:
    """Node 3a: Process simple content."""
    print("  [Node 3a] Processing as simple content...")
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
    
    response = llm.invoke([
        {"role": "system", "content": "Provide a brief, simple response."},
        {"role": "user", "content": state["user_input"]}
    ])
    
    return {
        "final_output": response.content,
        "processing_steps": state["processing_steps"] + ["simple_processing"]
    }


def process_complex(state: ComposedState) -> dict:
    """Node 3b: Process complex content."""
    print("  [Node 3b] Processing as complex content...")
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
    
    response = llm.invoke([
        {"role": "system", "content": "Provide a detailed, comprehensive response."},
        {"role": "user", "content": state["user_input"]}
    ])
    
    return {
        "final_output": response.content,
        "processing_steps": state["processing_steps"] + ["complex_processing"]
    }


def handle_error(state: ComposedState) -> dict:
    """Node 4: Handle errors gracefully."""
    print("  [Node 4] Handling error...")
    
    return {
        "final_output": f"Error: {state['error']}. Please provide valid input.",
        "processing_steps": state["processing_steps"] + ["error_handled"]
    }


def route_after_validation(state: ComposedState) -> Literal["analyze", "error"]:
    """Route based on validation result."""
    if state.get("error"):
        return "error"
    return "analyze"


def route_after_analysis(state: ComposedState) -> Literal["simple", "complex"]:
    """Route based on complexity analysis."""
    complexity = state.get("analysis", {}).get("complexity", "simple")
    
    if complexity == "complex":
        return "complex"
    return "simple"


def main():
    # Build the composed graph
    workflow = StateGraph(ComposedState)
    
    # Add all nodes
    workflow.add_node("validate", validate_input)
    workflow.add_node("analyze", analyze_content)
    workflow.add_node("simple", process_simple)
    workflow.add_node("complex", process_complex)
    workflow.add_node("error", handle_error)
    
    # Set entry point
    workflow.set_entry_point("validate")
    
    # Add conditional routing
    workflow.add_conditional_edges(
        "validate",
        route_after_validation,
        {
            "analyze": "analyze",
            "error": "error"
        }
    )
    
    workflow.add_conditional_edges(
        "analyze",
        route_after_analysis,
        {
            "simple": "simple",
            "complex": "complex"
        }
    )
    
    # Add edges to END
    workflow.add_edge("simple", END)
    workflow.add_edge("complex", END)
    workflow.add_edge("error", END)
    
    # Compile the graph
    app = workflow.compile()
    
    # Test the composed workflow
    print("=" * 70)
    print("LANGGRAPH COMPOSITION EXAMPLE")
    print("=" * 70)
    
    # Test 1: Valid complex input
    print("\n--- Test 1: Complex Content ---")
    result1 = app.invoke({
        "messages": [],
        "user_input": "Explain the implications of quantum computing on modern cryptography and data security",
        "analysis": {},
        "processing_steps": [],
        "final_output": "",
        "error": ""
    })
    
    print(f"\nProcessing Steps: {' → '.join(result1['processing_steps'])}")
    print(f"Final Output: {result1['final_output'][:100]}...")
    
    # Test 2: Invalid input
    print("\n\n--- Test 2: Invalid Input ---")
    result2 = app.invoke({
        "messages": [],
        "user_input": "Hi",
        "analysis": {},
        "processing_steps": [],
        "final_output": "",
        "error": ""
    })
    
    print(f"\nProcessing Steps: {' → '.join(result2['processing_steps'])}")
    print(f"Final Output: {result2['final_output']}")
    
    print("\n" + "=" * 70)
   

if __name__ == "__main__":
    main()
