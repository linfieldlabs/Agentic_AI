"""
LangChain Example 2: Multi-Step Pipeline
=========================================
This example shows how to build a multi-step pipeline using RunnablePassthrough
to chain multiple operations together.

Key Concepts:
- RunnablePassthrough: Passes data through while adding new fields
- assign(): Adds new fields to the state
- Multi-step composition
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


def extract_entities(data: dict) -> str:
    """Simulates entity extraction from text."""
    text = data.get("text", "")
    # In a real scenario, this would use NER or LLM-based extraction
    entities = ["Apple", "iPhone", "technology"]
    return ", ".join(entities)


def analyze_sentiment(data: dict) -> str:
    """Simulates sentiment analysis."""
    text = data.get("text", "")
    # In a real scenario, this would use a sentiment model
    return "Positive"


def main():
    # Initialize the LLM with Groq
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7
    )
    
    # Create a prompt that uses the extracted entities and sentiment
    prompt = ChatPromptTemplate.from_template(
        """Given the following analysis:
        
Text: {text}
Entities: {entities}
Sentiment: {sentiment}

Generate a brief summary highlighting the key points."""
    )
    
    # Build the multi-step pipeline
    pipeline = (
        RunnablePassthrough.assign(entities=extract_entities)
        | RunnablePassthrough.assign(sentiment=analyze_sentiment)
        | prompt
        | llm
    )
    
    # Test input
    input_data = {
        "text": "The new iPhone is amazing! Apple has outdone themselves with this technology."
    }
    
    # Run the pipeline
    result = pipeline.invoke(input_data)
    
    # Display results
    print("=" * 60)
    print("MULTI-STEP PIPELINE EXAMPLE")
    print("=" * 60)
    print(f"\nInput Text: {input_data['text']}")
    print(f"\nFinal Summary:\n{result.content}")
    print("=" * 60)


if __name__ == "__main__":
    main()
