"""Example: Structured Output with Pydantic in LangChain v1.

This example demonstrates how to use the .with_structured_output() method
to get typed Pydantic objects directly from the model.
"""
import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

def get_groq_api_key():
    """Retrieve the Groq API key from environment variables."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables. "
                         "Please ensure it is set in your .env file.")
    return api_key


# Define a Pydantic schema for structured output
class ActionDetails(BaseModel):
    """Details about an action."""
    search_query: str = Field(description="The query to search for")
    recipient: str = Field(description="Email recipient if applicable")


class AgentAction(BaseModel):
    """Schema for agent actions with structured output."""
    action_type: str = Field(description="The category of action to perform")
    details: ActionDetails = Field(description="Structured details for the action")
    confidence: float = Field(
        description="Confidence score between 0 and 1",
        ge=0.0,
        le=1.0
    )


def main():
    """Run the structured output example."""
    # Get API key
    api_key = get_groq_api_key()
    
    # Initialize Groq model
    model = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=api_key,
        temperature=0.1  # Low temperature for reliability
    )
    
    # Bind the schema to the model for structured output
    structured_llm = model.with_structured_output(AgentAction)
    
    print("=" * 60)
    print("Structured Output Example: .with_structured_output()")
    print("=" * 60)
    
    # Query that requires structured interpretation
    query = "Search for latest AI news and send a summary to admin@example.com"
    
    print(f"\nQuery: {query}")
    print("\nInvoking model for structured output...")
    
    try:
        # The model returns an instance of AgentAction directly
        result: AgentAction = structured_llm.invoke(query)
        
        print("\nParsed Result (Pydantic object):")
        print(f"Action Type: {result.action_type}")
        print(f"Details: {result.details}")
        print(f"Confidence: {result.confidence}")
        
        # Access nested fields with type safety
        print(f"\nTarget Recipient: {result.details.recipient}")
        
    except Exception as e:
        print(f"\nError obtaining structured output: {e}")
    
    print("\n" + "=" * 60)
    print("Benefits of .with_structured_output():")
    print("- Automatic parsing into Pydantic models")
    print("- Built-in validation")
    print("- IDE completion and type safety")
    print("=" * 60)


if __name__ == "__main__":
    main()
