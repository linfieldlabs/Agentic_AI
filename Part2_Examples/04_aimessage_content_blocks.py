import os
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ["GROQ_API_KEY"]

# llama-4-scout supports tool calling correctly on Groq
model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct", groq_api_key=api_key
)


# --- Example 1: Basic content access ---

print("=== Example 1: Basic Content Access ===")

response = model.invoke("Explain LangChain briefly.")

# .content for raw string
print(response.content[:200])

# .text property (v1) — replaces deprecated .text() method
print(response.text[:200])


# --- Example 2: content_blocks ---

print("\n=== Example 2: content_blocks ===")

for block in response.content_blocks:
    if block["type"] == "reasoning":
        print(f"Reasoning: {block.get('reasoning')}")
    elif block["type"] == "text":
        print(f"Text: {block.get('text', '')[:200]}")


# --- Example 3: Inspecting tool calls via agent ---

print("\n=== Example 3: Tool Calls ===")


@tool
def calculate(expression: str) -> str:
    """Perform a mathematical calculation."""
    import sympy

    return str(sympy.sympify(expression))


agent = create_agent(
    model="groq:meta-llama/llama-4-scout-17b-16e-instruct",
    tools=[calculate],
)

result = agent.invoke({"messages": [{"role": "user", "content": "What is 10 * 5?"}]})

for msg in result["messages"]:
    if getattr(msg, "tool_calls", None):
        for call in msg.tool_calls:
            print(f"Tool called: {call['name']} with args {call['args']}")

print(result["messages"][-1].content)
