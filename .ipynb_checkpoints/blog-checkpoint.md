<div align="center">
<h1>🔗 Graph Over Chain:</h1>

<h2>Scaling AI Agents Beyond LangChain Limits</h2>

<div align="center">
  <img
    src="./lngChVslngGr.png"
    alt="LangChain vs LangGraph Comparison"
    style="max-width: 900px; width: 100%;"
  />
</div>

</div>

<br>

## 🚀 Quick Start Guide: Jumpstart Your Agentic AI Journey

Welcome to your first step into the world of **Agentic AI** where AI systems don't just respond, but **think, plan, and act**.

This Quick Start Guide is the entry point to our **Agentic AI Design Patterns** blog series. The goal is simple:

**👉 Get you building agentic systems fast without getting stuck in theory.**

**No long lectures.**  
**No buzzword overload.**  
**Just practical ideas, clean patterns, and code you can actually run.**

<br>

---

**📂 Repository Structure:**
```
langchain_examples/     # 5 LangChain examples
langgraph_examples/     # 4 LangGraph examples
```

---

## 🧭 Series Structure

1. **Quick Start** – Environment + first agent
2. **Agentic Patterns** – Why/when patterns exist
3. **Implementation** – LangChain & LangGraph
4. **Comparisons** – Choosing the right tool

---

## 🎉 Let's Start Coding

From here on, every section will move you closer to real, working agents.

So, open your editor, grab some coffee ☕, and get ready to experiment.

**Enjoy coding from here**  
**Your Agentic AI journey starts now...**

<br>

---

# Part 1 — Understanding LangChain and LangGraph

## Introduction

As AI systems grow from single prompts to multi-step, tool-using agents, orchestration matters: state, control flow, retries, and observability. **LangChain** and **LangGraph** address this problem space with different trade-offs.

<div align="center">
  <img
    src="./lngChVslngGr2.png"
    alt="LangChain vs LangGraph Comparison"
    style="max-width: 900px; width: 100%;"
  />
</div>


---

# 🔗 What Is LangChain?

LangChain is a framework for building LLM-powered applications. Created by **Harrison Chase (2022)**, it popularized composable abstractions (prompts, tools, runnables) and a large integration ecosystem.

**Key philosophy:** productive defaults and composability. You trade some low-level control for speed.



---

## Fundamental Concepts of LangChain

### 1️⃣ Chains (Modern View)

Historically, `LLMChain` and `SequentialChain` were common. **Today, the recommended approach is LCEL / `Runnable` composition**.

```python
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

llm = ChatGroq(model="llama-3.3-70b-versatile")
prompt = ChatPromptTemplate.from_template(
    "Describe the key features of {product} in 3 concise bullet points."
)

chain = prompt | llm
result = chain.invoke({"product": "iPhone"})
print(result.content)
```

---

### 2️⃣ Multi‑Step Pipelines

Use `Runnable` composition instead of `SequentialChain`.

```python
from langchain_core.runnables import RunnablePassthrough

pipeline = (
    RunnablePassthrough.assign(entities=extract_entities)
    | RunnablePassthrough.assign(sentiment=analyze_sentiment)
    | generate_response
)
```



---

### 3️⃣ Agents (Current API)

Agents dynamically select tools. **Prefer `create_react_agent` or graph-based agents** over older function-agent helpers.

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_groq import ChatGroq
from langchain_core.tools import tool

@tool
def search_product(name: str) -> str:
    return f"{name}: $99"

llm = ChatGroq(model="llama-3.3-70b-versatile")
agent = create_react_agent(llm, tools=[search_product])
executor = AgentExecutor(agent=agent, tools=[search_product])

print(executor.invoke({"input": "What does a keyboard cost?"})["output"])
```



---

### 4️⃣ Memory

Memory exists, but **LangChain increasingly favors explicit state or external stores**.

**Modern Approach: RunnableWithMessageHistory**

```python
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# Store for conversation histories
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Create chain with memory
llm = ChatGroq(model="llama-3.3-70b-versatile")
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

# Use with session ID
response = chain_with_history.invoke(
    {"input": "My name is Alice"},
    config={"configurable": {"session_id": "user_123"}}
)
```

**Alternative: Manual Memory Management**

```python
# Manually maintain conversation history
conversation_history = [
    {"role": "system", "content": "You are a helpful assistant."}
]

def chat(user_message: str):
    conversation_history.append({"role": "user", "content": user_message})
    response = llm.invoke(conversation_history)
    conversation_history.append({"role": "assistant", "content": response.content})
    return response.content
```

<div align="center">
  <img
    src="./langchain1.png"
    alt="LangChain vs LangGraph Comparison"
    style="max-width: 900px; width: 80%; height:80%"
  />
</div>



---

## LangChain — Strengths & Limitations

### ✅ Strengths
- Fast prototyping
- Large integration ecosystem
- Strong docs and community

### ⚠️ Limitations (Nuanced)
- Control exists but is **indirect**
- Intermediate steps are visible via callbacks/LangSmith, but not first‑class state
- Complex custom orchestration becomes awkward

---

# 🕸️ What Is LangGraph?

LangGraph is a framework for **explicit, stateful, graph-based orchestration** of LLM workflows. Built by the LangChain team, it targets complex, long‑running, or multi‑agent systems.



---

## Fundamental Concepts of LangGraph

### 1️⃣ State

State is a typed object that flows through nodes.

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    iteration: int
```



---

### 2️⃣ Nodes

Nodes transform state.

```python
def agent_node(state: AgentState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response], "iteration": state["iteration"] + 1}
```

---

### 3️⃣ Edges & Control Flow

```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)
app = workflow.compile()
```

---

### 4️⃣ Streaming & Inspection

```python
for event in app.stream({"messages": [], "iteration": 0}):
    print(event)
```
<div align="center">
  <img
    src="./langraph2.png"
    alt="LangChain vs LangGraph Comparison"
    style="max-width: 900px; width: 80%; height:80%"
  />
</div>



---

## LangGraph — Strengths & Limitations

### ✅ Strengths
- Explicit state
- Deterministic control flow
- Easier debugging for complex systems

### ⚠️ Limitations
- More code for simple tasks
- Steeper learning curve
- Smaller ecosystem than LangChain

---

## Architecture Comparison 

| Aspect | LangChain | LangGraph |
|------|-----------|-----------|
| Abstraction | High | Medium–Low |
| Control | Indirect | Explicit |
| State | Implicit or external | Explicit |
| Debugging | Via callbacks/LangSmith | Via state inspection |
| Best For | Fast builds | Complex workflows |

---



## Environment Setup

### Installation

```bash
pip install -r requirements.txt
```

This installs:
- `langchain` - Core framework
- `langgraph` - Graph-based orchestration
- `langchain-groq` - Groq API integration
- `langchain-core` - Core abstractions
- `python-dotenv` - Environment variable management

### API Key Setup

Create a `.env` file:
```
GROQ_API_KEY=your-groq-api-key-here
```

Or set environment variable:
```bash
# Windows PowerShell
$env:GROQ_API_KEY = "your-key"

# Linux/Mac
export GROQ_API_KEY="your-key"
```

### Running Examples

```bash
# LangChain examples
cd langchain_examples
python 01_basic_chain.py
python 02_multi_step_pipeline.py
python 03_agent_with_tools.py
python 04_composition.py
python 05_memory.py

# LangGraph examples
cd langgraph_examples
python 01_state_management.py
python 02_workflow.py
python 03_streaming.py
python 04_composition.py
```

---

## Final Takeaway

- **LangChain** excels at speed and integrations.
- **LangGraph** excels at explicit control and complex orchestration.
- They are **complementary**, not competitors.

Use LangChain to move fast. Use LangGraph when the workflow itself becomes your product.

---

# 📅 Next Article

## **Part 2: Agentic Design Patterns – Implementation Guide**

⏳ *Coming soon...*

<br>

---

<br>

**Happy Coding! 🎉**

*Built with ❤️ for the AI Agent Community*
