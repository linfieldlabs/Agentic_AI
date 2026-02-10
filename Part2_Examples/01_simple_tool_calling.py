"""Example: Simple Tool Calling with bind_tools in LangChain v1.

This example demonstrates how to use the .bind_tools() method to enable
transparent integration of custom functions with language models.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

# Load environment variables
load_dotenv()


def get_groq_api_key():
    """Retrieve the Groq API key from environment variables."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found in environment variables. "
            "Please ensure it is set in your .env file."
        )
    return api_key


# Define a simple calculator tool
@tool
def calculate(expression: str) -> str:
    """Perform a mathematical calculation safely.

    Args:
        expression: A mathematical expression to evaluate

    Returns:
        str: The result of the calculation
    """
    try:
        # Safe evaluation with restricted builtins
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error calculating: {str(e)}"


# Define a weather tool for demonstration
@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location.

    Args:
        location: The city or location name

    Returns:
        str: Weather information
    """
    # Simulated weather data
    weather_data = {
        "new york": "Sunny, 72°F",
        "london": "Cloudy, 15°C",
        "tokyo": "Rainy, 20°C",
        "paris": "Partly cloudy, 18°C",
    }
    location_lower = location.lower()
    weather = weather_data.get(
        location_lower, f"Weather data not available for {location}"
    )
    return f"The weather in {location} is {weather}"


# Define a search tool
@tool
def search_web(query: str) -> str:
    """Search the web for information.

    Args:
        query: The search query

    Returns:
        str: Search results
    """
    # Simulated search results
    results = f"Found 3 relevant articles about {query}."
    return f"Search results for '{query}': {results}"


def run_agent_loop(model_with_tools, user_query, max_iterations=5):
    """Run a simple agent loop with tool calling.

    Args:
        model_with_tools: The language model with tools bound
        user_query: The user's query
        max_iterations: Maximum number of iterations to prevent infinite loops

    Returns:
        str: The final response from the agent
    """
    messages = [HumanMessage(content=user_query)]

    print(f"\n{'=' * 60}")
    print(f"User Query: {user_query}")
    print(f"{'=' * 60}\n")

    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}:")

        # Invoke the model
        response = model_with_tools.invoke(messages)
        messages.append(response)

        # Check if the model wants to use tools
        if not response.tool_calls:
            print(f"  -> Final Response: {response.content}\n")
            return response.content

        # Execute tool calls and add results to messages
        print(f"  -> Tool calls requested: {len(response.tool_calls)}")
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"    * Calling {tool_name}({tool_args})")

            # Execute the appropriate tool
            if tool_name == "calculate":
                result = calculate.invoke(tool_args)
            elif tool_name == "get_weather":
                result = get_weather.invoke(tool_args)
            elif tool_name == "search_web":
                result = search_web.invoke(tool_args)
            else:
                result = f"Unknown tool: {tool_name}"

            print(f"    * Result: {result}")

            # Add tool result to messages
            tool_msg = ToolMessage(content=result, tool_call_id=tool_call["id"])
            messages.append(tool_msg)
        print()

    return "Max iterations reached without final answer"


def main():
    """Run the simple tool calling examples."""
    # Get API key
    api_key = get_groq_api_key()

    # Initialize Groq model
    # Note: parallel_tool_calls=False prevents API errors with multiple tools
    model = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=api_key,
        temperature=0,
        model_kwargs={"parallel_tool_calls": False},
    )

    print("\n" + "=" * 60)
    print("Simple Tool Calling with bind_tools - Examples")
    print("=" * 60)

    # Example 1: Single tool (calculator)
    print("\n--- Example 1: Calculator Tool ---")
    model_with_calculator = model.bind_tools([calculate])
    run_agent_loop(model_with_calculator, "Calculate 15 * 23 + 42")

    # Example 2: Single tool (weather)
    print("\n--- Example 2: Weather Tool ---")
    model_with_weather = model.bind_tools([get_weather])
    run_agent_loop(model_with_weather, "What's the weather like in Tokyo?")

    # Example 3: Multiple tools
    print("\n--- Example 3: Multiple Tools ---")
    all_tools = [calculate, get_weather, search_web]
    model_with_all_tools = model.bind_tools(all_tools)
    run_agent_loop(
        model_with_all_tools, "What's the weather in Paris and calculate 100 / 4"
    )

    # Example 4: Sequential tool usage
    print("\n--- Example 4: Sequential Tool Usage ---")
    run_agent_loop(model_with_all_tools, "Calculate the result of 256 * 128")

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
