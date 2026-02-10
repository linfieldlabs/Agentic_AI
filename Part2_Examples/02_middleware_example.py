"""Example: Custom Middleware-like functionality in LangChain v1.

This example demonstrates how to create custom middleware-like functionality 
using RunnableLambdas to extend model functionality with lifecycle hooks.
"""
import os
from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda, RunnableConfig

# Load environment variables
load_dotenv()

def get_groq_api_key():
    """Retrieve the Groq API key from environment variables."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables. "
                         "Please ensure it is set in your .env file.")
    return api_key


# Define a simple tool for demonstration
@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location.
    
    Args:
        location: The city or location name
        
    Returns:
        str: Weather information
    """
    return f"The weather in {location} is sunny, 72°F"


class HistorySummarizer:
    """Middleware-like component to manage conversation history.
    
    This demonstrates how to manipulate messages before they're sent to the model.
    """
    
    def __init__(self, max_messages: int = 4):
        self.max_messages = max_messages
    
    def __call__(self, input_data: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
        """Process messages before model invocation."""
        messages = input_data.get("messages", [])
        
        # If we have too many messages, prune the older ones
        if len(messages) > self.max_messages:
            print(f"[Middleware] History has {len(messages)} messages, "
                  f"keeping only the last {self.max_messages}")
            input_data["messages"] = messages[-self.max_messages:]
        
        return input_data


def log_response(response: Any) -> Any:
    """Post-processing hook to log model responses."""
    print(f"[Middleware] Model response received: {type(response).__name__}")
    return response


def main():
    """Run the middleware example."""
    # Get API key
    api_key = get_groq_api_key()
    
    # Initialize Groq model
    model = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=api_key,
        temperature=0.7
    )
    
    # Define our "middleware" pipeline
    summarizer = HistorySummarizer(max_messages=2)
    
    # Chain components: Input -> Summarizer -> (lambda x: x["messages"]) -> Model -> Logger
    chain = RunnableLambda(summarizer) | (lambda x: x["messages"]) | model | RunnableLambda(log_response)
    
    print("=" * 60)
    print("Middleware Example: History Management via Runnables")
    print("=" * 60)
    
    # Simulate a history with many messages
    long_history = [
        HumanMessage(content="Hi there!"),
        get_weather.invoke({"location": "New York"}),
        HumanMessage(content="What's the weather in San Francisco?"),
    ]
    
    print("\nInvoking chain with 3 messages (max_messages=2)...")
    response = chain.invoke({"messages": long_history})
    
    print("\nAgent Response:")
    print(response.content)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
