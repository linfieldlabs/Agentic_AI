import os
from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

# llama-4-scout supports tool calling correctly on Groq
MODEL = "groq:meta-llama/llama-4-scout-17b-16e-instruct"


# --- Single tool ---


@tool
def calculate(expression: str) -> str:
    """Perform a mathematical calculation."""
    import sympy

    return str(sympy.sympify(expression))


agent = create_agent(
    model=MODEL,
    tools=[calculate],
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Calculate 15 * 23 + 42"}]}
)
print(result["messages"][-1].content)


# --- Multiple tools ---


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny, 22°C."


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Top result for '{query}': LangChain v1 released."


agent_multi = create_agent(
    model=MODEL,
    tools=[calculate, get_weather, search],
)

result = agent_multi.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What's the weather in Tokyo and search for LangChain v1?",
            }
        ]
    }
)
print(result["messages"][-1].content)
