"""Example: Handling Complex Content in LangChain v1.

This example demonstrates how to work with message content in LangChain,
including text and potential tool calls, which are the standard way 
multi-part content is handled.
"""
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage

# Load environment variables
load_dotenv()

def get_groq_api_key():
    """Retrieve the Groq API key from environment variables."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables. "
                         "Please ensure it is set in your .env file.")
    return api_key


def process_response(response: AIMessage):
    """Process and log response content.
    
    Args:
        response: Model response (AIMessage)
    """
    print("\n" + "=" * 60)
    print("Response Content Analysis")
    print("=" * 60)
    
    # In LangChain, content is usually a string, but can be a list of dicts 
    # for multimodal or complex models.
    content = response.content
    
    if isinstance(content, str):
        print(f"Content Type: String")
        print(f"Text Content: {content[:200]}...")
    elif isinstance(content, list):
        print(f"Content Type: List (Multi-part)")
        for i, part in enumerate(content, 1):
            part_type = part.get("type", "unknown")
            print(f"  Part {i}: Type = {part_type}")
            if part_type == "text":
                print(f"    Text: {part.get('text', '')[:100]}...")
    
    # Check for tool calls (another form of "structured content")
    if response.tool_calls:
        print(f"\nTool Calls Detected: {len(response.tool_calls)}")
        for i, call in enumerate(response.tool_calls, 1):
            print(f"  Call {i}: {call['name']}({call['args']})")


def main():
    """Run the content analysis example."""
    # Get API key
    api_key = get_groq_api_key()
    
    # Initialize Groq model
    # Most Groq models currently return string content
    model = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=api_key,
        temperature=0.7
    )
    
    print("=" * 60)
    print("Content Analysis Example: Processing AIMessage")
    print("=" * 60)
    
    query = "List 3 benefits of LangChain and provide a brief explanation for each."
    
    print(f"\nQuery: {query}")
    print("\nInvoking model...")
    
    # Invoke the model
    response = model.invoke(query)
    
    # Process the response
    process_response(response)
    
    print("\n" + "=" * 60)
    print("Key Concepts:")
    print("- AIMessage.content: Primary response data")
    print("- AIMessage.tool_calls: Structured data for tool execution")
    print("- Multi-part content: Supports text, images, and more in some models")
    print("=" * 60)


if __name__ == "__main__":
    main()
