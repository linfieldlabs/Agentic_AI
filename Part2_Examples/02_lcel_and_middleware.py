import os
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ["GROQ_API_KEY"]
model = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key)


# --- Example 1: LCEL pipeline (non-agent) ---

print("=== Example 1: LCEL Pipeline ===")


def log_response(response):
    print(f"Response type: {type(response).__name__}")
    return response


chain = RunnableLambda(lambda x: x["messages"]) | model | RunnableLambda(log_response)

response = chain.invoke({"messages": ["Explain LangChain in one sentence."]})
print(response.content)


# --- Example 2: Agent without tools (middleware concept demo) ---

print("\n=== Example 2: Agent with create_agent ===")

# Note: SummarizationMiddleware is shown here as the v1 pattern for
# managing long conversation history inside agents.
# Groq tool_use_failed limitation prevents demo with tools + middleware together.
# The middleware config below shows the correct v1 API usage:
#
# from langchain.agents.middleware import SummarizationMiddleware
# agent = create_agent(
#     model="groq:llama-3.3-70b-versatile",
#     tools=[...],
#     middleware=[
#         SummarizationMiddleware(
#             model="groq:llama-3.3-70b-versatile",
#             trigger=("tokens", 4000),
#             keep=("messages", 20),
#         )
#     ],
# )

agent = create_agent(
    model="groq:llama-3.3-70b-versatile",
    tools=[],
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is LangChain v1?"}]}
)
print(result["messages"][-1].content)
