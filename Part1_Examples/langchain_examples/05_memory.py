"""
LangChain Example 5: Memory and Conversation History
=====================================================
This example demonstrates how to maintain conversation context using
explicit state management in LangChain.

Key Concepts:
- ConversationBufferMemory: Stores conversation history
- Message history: Maintaining context across turns
- Explicit state: Modern approach to memory
- Conversation chains: Building chatbots with context
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory


# Store for conversation histories
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Get or create a chat history for a given session.
    
    Args:
        session_id: Unique identifier for the conversation session
        
    Returns:
        Chat message history for the session
    """
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def example_1_basic_memory():
    """Example 1: Basic conversation with memory using modern approach."""
    
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Conversation Memory")
    print("=" * 70)
    
    # Initialize LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7
    )
    
    # Create a prompt that includes message history
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Keep track of the conversation context."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # Create the chain
    chain = prompt | llm
    
    # Wrap with message history
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )
    
    # Conversation session
    session_id = "user_123"
    
    # Turn 1
    print("\n[Turn 1]")
    print("User: My name is Alice")
    response1 = chain_with_history.invoke(
        {"input": "My name is Alice"},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"Assistant: {response1.content}")
    
    # Turn 2 - Test if it remembers
    print("\n[Turn 2]")
    print("User: What's my name?")
    response2 = chain_with_history.invoke(
        {"input": "What's my name?"},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"Assistant: {response2.content}")
    
    # Turn 3 - Continue conversation
    print("\n[Turn 3]")
    print("User: What did I tell you in the first message?")
    response3 = chain_with_history.invoke(
        {"input": "What did I tell you in the first message?"},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"Assistant: {response3.content}")
    
    print("\n" + "=" * 70)


def example_2_manual_memory():
    """Example 2: Manual memory management with explicit message list."""
    
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Manual Memory Management")
    print("=" * 70)
    
    # Initialize LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7
    )
    
    # Manually maintain conversation history
    conversation_history = [
        {"role": "system", "content": "You are a helpful math tutor."}
    ]
    
    def chat(user_message: str) -> str:
        """Send a message and get a response while maintaining history."""
        # Add user message to history
        conversation_history.append({"role": "user", "content": user_message})
        
        # Get response from LLM
        response = llm.invoke(conversation_history)
        
        # Add assistant response to history
        conversation_history.append({"role": "assistant", "content": response.content})
        
        return response.content
    
    # Conversation
    print("\n[Turn 1]")
    print("User: What is 5 + 3?")
    response1 = chat("What is 5 + 3?")
    print(f"Assistant: {response1}")
    
    print("\n[Turn 2]")
    print("User: Now multiply that by 2")
    response2 = chat("Now multiply that by 2")
    print(f"Assistant: {response2}")
    
    print("\n[Turn 3]")
    print("User: What was my first question?")
    response3 = chat("What was my first question?")
    print(f"Assistant: {response3}")
    
    # Show conversation history
    print("\n[Conversation History]")
    for i, msg in enumerate(conversation_history[1:], 1):  # Skip system message
        role = "User" if msg["role"] == "user" else "Assistant"
        print(f"{i}. {role}: {msg['content'][:60]}...")
    
    print("\n" + "=" * 70)


def example_3_session_management():
    """Example 3: Multiple conversation sessions."""
    
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Multiple Conversation Sessions")
    print("=" * 70)
    
    # Initialize LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    chain = prompt | llm
    
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )
    
    # Session 1: Alice
    print("\n[Session 1: Alice]")
    response_alice = chain_with_history.invoke(
        {"input": "My favorite color is blue"},
        config={"configurable": {"session_id": "alice_session"}}
    )
    print(f"Alice: My favorite color is blue")
    print(f"Assistant: {response_alice.content}")
    
    # Session 2: Bob
    print("\n[Session 2: Bob]")
    response_bob = chain_with_history.invoke(
        {"input": "My favorite color is red"},
        config={"configurable": {"session_id": "bob_session"}}
    )
    print(f"Bob: My favorite color is red")
    print(f"Assistant: {response_bob.content}")
    
    # Back to Session 1: Alice
    print("\n[Back to Session 1: Alice]")
    response_alice2 = chain_with_history.invoke(
        {"input": "What's my favorite color?"},
        config={"configurable": {"session_id": "alice_session"}}
    )
    print(f"Alice: What's my favorite color?")
    print(f"Assistant: {response_alice2.content}")
    
    # Back to Session 2: Bob
    print("\n[Back to Session 2: Bob]")
    response_bob2 = chain_with_history.invoke(
        {"input": "What's my favorite color?"},
        config={"configurable": {"session_id": "bob_session"}}
    )
    print(f"Bob: What's my favorite color?")
    print(f"Assistant: {response_bob2.content}")
    
    print("\n" + "=" * 70)


def main():
    print("=" * 70)
    print("LANGCHAIN MEMORY EXAMPLES")
    print("=" * 70)
    print("\nDemonstrating different approaches to conversation memory:")
    print("1. RunnableWithMessageHistory (Modern approach)")
    print("2. Manual memory management")
    print("3. Multiple conversation sessions")
    
    # Run examples
    example_1_basic_memory()
    example_2_manual_memory()
    example_3_session_management()
    
    print("\n" + "=" * 70)
   


if __name__ == "__main__":
    main()
