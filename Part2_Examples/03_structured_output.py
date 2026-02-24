import os
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ["GROQ_API_KEY"]


class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str


# --- Example 1: Direct model call with .with_structured_output() ---
# Uses llama-4-scout which supports json_schema on Groq

print("=== Example 1: Direct Model Call ===")

model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct", groq_api_key=api_key
)
structured_model = model.with_structured_output(ContactInfo)

contact = structured_model.invoke(
    "Extract contact: John Doe, john@example.com, (555) 123-4567"
)
print(contact)


# --- Example 2: Agent with ProviderStrategy ---
# Uses llama-4-scout which supports json_schema on Groq

print("\n=== Example 2: Agent with ProviderStrategy ===")

agent = create_agent(
    model="groq:meta-llama/llama-4-scout-17b-16e-instruct",
    tools=[],
    response_format=ProviderStrategy(ContactInfo),
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Return contact info for Jane Smith, jane@example.com, (555) 987-6543",
            }
        ]
    }
)
print(result["structured_response"])
