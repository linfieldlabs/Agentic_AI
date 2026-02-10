"""
LangChain Example 3: Simple Tool-Using Agent
=============================================
This example demonstrates a simplified agent pattern that uses tools
to answer questions.

Key Concepts:
- @tool decorator: Creates a tool from a function
- Manual tool selection: Agent decides which tool to use
- Sequential reasoning: Step-by-step problem solving
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate


@tool
def search_product(name: str) -> str:
    """Search for a product and return its price.
    
    Args:
        name: The name of the product to search for
        
    Returns:
        Product information including price
    """
    # Simulated product database
    products = {
        "keyboard": "$99",
        "mouse": "$49",
        "monitor": "$299",
        "laptop": "$1299"
    }
    
    name_lower = name.lower()
    for product, price in products.items():
        if product in name_lower:
            return f"{product.capitalize()}: {price}"
    
    return f"Product '{name}' not found in our database."


@tool
def calculate_discount(price: str, discount_percent: int) -> str:
    """Calculate the discounted price.
    
    Args:
        price: Original price (e.g., "$99")
        discount_percent: Discount percentage (e.g., 20 for 20%)
        
    Returns:
        Discounted price
    """
    try:
        # Extract numeric value from price string
        numeric_price = float(price.replace("$", ""))
        discounted = numeric_price * (1 - discount_percent / 100)
        return f"${discounted:.2f}"
    except:
        return "Error calculating discount"


def simple_agent(query: str, tools: list, llm):
    """
    A simple agent that uses LLM to decide which tools to use.
    This is a simplified version that works without advanced tool-calling APIs.
    """
    
    # Create tool descriptions
    tool_descriptions = "\n".join([
        f"- {tool.name}: {tool.description}" 
        for tool in tools
    ])
    
    # First, ask LLM to analyze the query and decide which tools to use
    analysis_prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant with access to these tools:

{tool_descriptions}

User query: {query}

Think step by step:
1. What information do I need?
2. Which tool(s) should I use?
3. In what order?

Provide your analysis and the tool(s) to use."""
    )
    
    analysis_chain = analysis_prompt | llm
    analysis = analysis_chain.invoke({
        "tool_descriptions": tool_descriptions,
        "query": query
    })
    
    print(f"\n[Agent Thinking]: {analysis.content[:200]}...")
    
    # Execute tools based on the query
    # For simplicity, we'll use keyword matching
    results = []
    
    # Check if we need to search for a product
    if any(word in query.lower() for word in ["cost", "price", "keyboard", "mouse", "monitor", "laptop"]):
        for tool in tools:
            if tool.name == "search_product":
                # Extract product name from query
                for product in ["keyboard", "mouse", "monitor", "laptop"]:
                    if product in query.lower():
                        result = tool.invoke({"name": product})
                        results.append(f"Product search result: {result}")
                        print(f"[Tool Used]: search_product('{product}') -> {result}")
    
    # Check if we need to calculate discount
    if "discount" in query.lower():
        # Try to extract discount percentage
        import re
        discount_match = re.search(r'(\d+)%', query)
        if discount_match and results:
            discount_percent = int(discount_match.group(1))
            # Get price from previous result
            price_match = re.search(r'\$(\d+)', results[0])
            if price_match:
                price = f"${price_match.group(1)}"
                for tool in tools:
                    if tool.name == "calculate_discount":
                        result = tool.invoke({
                            "price": price,
                            "discount_percent": discount_percent
                        })
                        results.append(f"Discount calculation: {result}")
                        print(f"[Tool Used]: calculate_discount('{price}', {discount_percent}) -> {result}")
    
    # Generate final answer using the tool results
    final_prompt = ChatPromptTemplate.from_template(
        """Based on the following information, answer the user's question:

User query: {query}

Tool results:
{tool_results}

Provide a clear, concise answer."""
    )
    
    final_chain = final_prompt | llm
    final_answer = final_chain.invoke({
        "query": query,
        "tool_results": "\n".join(results) if results else "No tools were needed."
    })
    
    return final_answer.content


def main():
    # Initialize the LLM with Groq
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7
    )
    
    # Define tools
    tools = [search_product, calculate_discount]
    
    # Test the agent
    print("=" * 60)
    print("SIMPLE TOOL-USING AGENT EXAMPLE")
    print("=" * 60)
    
    # Example 1: Simple product search
    print("\n--- Example 1: Product Search ---")
    query1 = "What does a keyboard cost?"
    print(f"Query: {query1}")
    result1 = simple_agent(query1, tools, llm)
    print(f"\n[Final Answer]: {result1}")
    
    # Example 2: Multi-step reasoning with tools
    print("\n\n--- Example 2: Price with Discount ---")
    query2 = "What would a monitor cost with a 15% discount?"
    print(f"Query: {query2}")
    result2 = simple_agent(query2, tools, llm)
    print(f"\n[Final Answer]: {result2}")
    
    print("\n" + "=" * 60)
   

if __name__ == "__main__":
    main()
