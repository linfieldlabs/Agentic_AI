"""
LangChain Example 4: Composition of Basics
===========================================
This example demonstrates how to compose multiple LangChain concepts together:
- Multiple chains working together
- Combining prompts, parsers, and LLMs
- Building complex workflows from simple components

Key Concepts:
- Chain composition: Building complex workflows from simple parts
- Output parsers: Structured output from LLMs
- RunnableLambda: Custom processing steps
- Parallel execution: Running multiple chains simultaneously
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser


def main():
    # Initialize the LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7
    )
    
    # Component 1: Topic analyzer
    topic_prompt = ChatPromptTemplate.from_template(
        "Identify the main topic in one word: {text}"
    )
    topic_chain = topic_prompt | llm | StrOutputParser()
    
    # Component 2: Sentiment analyzer
    sentiment_prompt = ChatPromptTemplate.from_template(
        "What is the sentiment (positive/negative/neutral): {text}"
    )
    sentiment_chain = sentiment_prompt | llm | StrOutputParser()
    
    # Component 3: Summary generator
    summary_prompt = ChatPromptTemplate.from_template(
        "Summarize in one sentence: {text}"
    )
    summary_chain = summary_prompt | llm | StrOutputParser()
    
    # Compose everything together using RunnableParallel
    # This runs all three chains in parallel
    parallel_analysis = RunnableParallel(
        topic=topic_chain,
        sentiment=sentiment_chain,
        summary=summary_chain
    )
    
    # Final composition: Analyze, then generate report
    report_prompt = ChatPromptTemplate.from_template(
        """Based on this analysis:
        
Topic: {topic}
Sentiment: {sentiment}
Summary: {summary}

Generate a brief analytical report."""
    )
    
    # Complete pipeline
    full_pipeline = (
        {"text": RunnablePassthrough()}  # Pass input text to all parallel chains
        | parallel_analysis
        | report_prompt
        | llm
        | StrOutputParser()
    )
    
    # Test the composition
    test_text = "I absolutely love the new features in this product! The team has done an amazing job improving the user experience."
    
    print("=" * 70)
    print("LANGCHAIN COMPOSITION EXAMPLE")
    print("=" * 70)
    print(f"\nInput Text:\n{test_text}")
    print("\n" + "-" * 70)
    print("\nProcessing through composed pipeline...")
    print("-" * 70)
    
    result = full_pipeline.invoke(test_text)
    
    print(f"\nFinal Report:\n{result}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
