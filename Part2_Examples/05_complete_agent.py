"""Example: Simple Tool Calling with Groq.

This example demonstrates tool calling without complex agent framework.
"""
import os
import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Load environment variables
load_dotenv()

def get_groq_api_key():
    """Retrieve the Groq API key from environment variables."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    return api_key

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Define tools
@tool
def calculate(expression: str) -> str:
    """Perform a mathematical calculation safely.
    
    Args:
        expression: Mathematical expression (e.g., '15 * 23 + 42', '2**8')
        
    Returns:
        str: Calculation result
    """
    try:
        import re
        # Allow numbers, operators, spaces, parentheses, and **
        if not re.match(r"^[0-9+\-*/\s().**]+$", expression):
            return "Error: Invalid characters in expression"
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


def run_agent_loop(model_with_tools, user_query, max_iterations=5):
    """Simple agent loop that handles tool calls."""
    messages = [HumanMessage(content=user_query)]
    
    for iteration in range(max_iterations):
        # Get model response
        response = model_with_tools.invoke(messages)
        messages.append(response)
        
        # Check if there are tool calls
        if not response.tool_calls:
            # No tool calls, we have final answer
            return response.content
        
        # Execute tool calls
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            print(f"  → Calling tool: {tool_name} with args: {tool_args}")
            
            # Execute the tool
            if tool_name == "calculate":
                result = calculate.invoke(tool_args)
                print(f"  → Tool result: {result}")
                
                # Add tool result to messages
                messages.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=tool_call["id"]
                    )
                )
    
    return "Max iterations reached"


def main():
    """Run the complete agent example."""
    try:
        # Get API key
        api_key = get_groq_api_key()
        
        # Initialize model
        model = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=api_key,
            temperature=0
        )
        
        # Bind tools to model
        model_with_tools = model.bind_tools([calculate])
        
        print("=" * 60)
        print("Complete Agent Example: LangChain Agent with Tools")
        print("=" * 60)
        
        # Example queries
        queries = [
            "Calculate 15 * 23 + 42",
            "What is 2 to the power of 8?",
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\nQuery {i}: {query}")
            
            try:
                # Run agent loop
                result = run_agent_loop(model_with_tools, query)
                print(f"\nResponse: {result}")
                print("-" * 30)
            except Exception as e:
                logger.error(f"Error processing query {i}: {e}")
                print(f"\nError: {e}")
                print("-" * 30)
        
        print("\n" + "=" * 60)
        print("Example completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()