"""
LangChain Example 1: Basic Chain
=================================
This example demonstrates the modern LCEL (LangChain Expression Language) approach
to creating a simple chain using the pipe operator.

Key Concepts:
- ChatPromptTemplate: Defines the prompt structure
- ChatGroq: The LLM model (using Groq API)
- Chain composition using | operator
"""

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


def main():
    # Initialize the LLM with Groq
    # Note: Requires GROQ_API_KEY environment variable
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7
    )
    
    # Create a prompt template
    prompt = ChatPromptTemplate.from_template(
        "Describe the key features of {product} in 3 concise bullet points."
    )
    
    # Compose the chain using the pipe operator (LCEL)
    chain = prompt | llm
    
    # Invoke the chain
    result = chain.invoke({"product": "iPhone"})
    
    # Display the result
    print("=" * 60)
    print("BASIC CHAIN EXAMPLE")
    print("=" * 60)
    print(f"\nProduct: iPhone")
    print(f"\nResponse:\n{result.content}")
    print("=" * 60)


if __name__ == "__main__":
    main()
