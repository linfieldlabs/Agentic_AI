# 🔗 Agentic AI with LangChain & LangGraph

<div align="center">

### **Graph Over Chain: Scaling AI Agents Beyond LangChain Limits**

</div>

---

🚀 **Start Here**  
📖 Companion Blog Posts —

1. **[Graph Over Chain: Scaling AI Agents Beyond LangChain Limits](https://github.com/linfieldlabs/Agentic_AI.git)**
2. **[Graph Over Chain — Part 2: LangChain v1 & LangGraph v1 for Engineering-Grade Agentic AI](https://github.com/linfieldlabs/Agentic_AI.git)**  
   Learn the concepts, comparisons, and architecture patterns used in this repo.

---

## 📖 About This Repository

This repository contains **practical code examples** demonstrating LangChain and LangGraph frameworks for building AI agents. It includes **15 runnable examples** organized into two parts:

- **Part 1:** Foundational concepts (9 examples)
- **Part 2:** Production-grade patterns (6 examples)

All examples use the **Groq API** (free tier available) for fast LLM inference.

---

## 📂 Repository Structure

```
QuickStart/
│
├── Part1_Examples/                    # Foundational Concepts (9 examples)
│   │
│   ├── langchain_examples/            # 5 LangChain Examples
│   │   ├── 01_basic_chain.py          # LCEL basics: prompt | llm
│   │   ├── 02_multi_step_pipeline.py  # Composing operations
│   │   ├── 03_agent_with_tools.py     # Dynamic tool selection
│   │   ├── 04_composition.py          # Reusable components
│   │   └── 05_memory.py               # Conversation history
│   │
│   └── langgraph_examples/            # 4 LangGraph Examples
│       ├── 01_state_management.py     # Typed state objects
│       ├── 02_workflow.py             # Graph orchestration
│       ├── 03_streaming.py            # Real-time state inspection
│       └── 04_composition.py          # Multi-agent systems
│
├── Part2_Examples/                    # Production Patterns (6 examples)
│   ├── 01_simple_agent.py             # Simple agent with tool calling
│   ├── 02_lcel_and_middleware.py      # LCEL and agent middleware concepts
│   ├── 03_structured_output.py        # Type-safe responses (Pydantic)
│   ├── 04_aimessage_content_blocks.py # Message content blocks & reasoning
│   ├── 05_langgraph_stategraph.py     # StateGraph basics (nodes/edges)
│   └── 06_persistence_and_hitl.py     # Persistence & Human-in-the-loop
│
├── requirements.txt                   # Python dependencies
├── .env                               # API keys (create this)
└── README.md                          # This file
```

---

## 🚀 Getting Started

### **Prerequisites**

- Python 3.8 or higher
- Groq API key ([Get free key here](https://console.groq.com/))

### **Step 1: Clone the Repository**

```bash
git clone https://github.com/linfieldlabs/Agentic_AI.git
cd Agentic_AI/QuickStart
```

### **Step 2: Install Dependencies**

```bash
pip install -r requirements.txt
```

**Installed packages:**

- `langchain` - Core LangChain framework
- `langgraph` - Graph-based orchestration
- `langchain-groq` - Groq API integration
- `langchain-core` - Core abstractions
- `python-dotenv` - Environment variables

### **Step 3: Set Up API Key**

Create a `.env` file in the root directory:

```bash
GROQ_API_KEY=your-groq-api-key-here
```

**Or set as environment variable:**

```bash
# Windows PowerShell
$env:GROQ_API_KEY = "your-key-here"

# Linux/Mac
export GROQ_API_KEY="your-key-here"
```

### **Step 4: Run Examples**

**Part 1 - LangChain Examples:**

```bash
cd Part1_Examples/langchain_examples
python 01_basic_chain.py
python 02_multi_step_pipeline.py
python 03_agent_with_tools.py
python 04_composition.py
python 05_memory.py
```

**Part 1 - LangGraph Examples:**

```bash
cd Part1_Examples/langgraph_examples
python 01_state_management.py
python 02_workflow.py
python 03_streaming.py
python 04_composition.py
```

**Part 2 - Advanced Examples:**

```bash
cd Part2_Examples
python 01_simple_agent.py
python 02_lcel_and_middleware.py
python 03_structured_output.py
python 04_aimessage_content_blocks.py
python 05_langgraph_stategraph.py
python 06_persistence_and_hitl.py
```

---

## 📚 What's Included

### **Part 1: Foundational Concepts**

#### **LangChain Examples (5 files)**

| File                        | Description                                          |
| --------------------------- | ---------------------------------------------------- |
| `01_basic_chain.py`         | Modern LCEL syntax with prompt templates             |
| `02_multi_step_pipeline.py` | Chaining operations with RunnablePassthrough         |
| `03_agent_with_tools.py`    | Agent with dynamic tool selection                    |
| `04_composition.py`         | Building reusable chain components                   |
| `05_memory.py`              | Conversation history with RunnableWithMessageHistory |

#### **LangGraph Examples (4 files)**

| File                     | Description                                    |
| ------------------------ | ---------------------------------------------- |
| `01_state_management.py` | Typed state objects with TypedDict             |
| `02_workflow.py`         | Graph-based orchestration with nodes and edges |
| `03_streaming.py`        | Real-time state inspection and streaming       |
| `04_composition.py`      | Multi-agent coordination with subgraphs        |

### **Part 2: Production-Grade Patterns**

| File                             | Description                                     |
| -------------------------------- | ----------------------------------------------- |
| `01_simple_agent.py`             | Simple agent with dynamic tool calling          |
| `02_lcel_and_middleware.py`      | LCEL pipelines and agent middleware concepts    |
| `03_structured_output.py`        | Type-safe responses (Pydantic/ProviderStrategy) |
| `04_aimessage_content_blocks.py` | Handling message content blocks and reasoning   |
| `05_langgraph_stategraph.py`     | Building agents using LangGraph StateGraph      |
| `06_persistence_and_hitl.py`     | Advanced persistence and Human-in-the-Loop      |

---

## 🔧 Troubleshooting

### Common Issues

**"ModuleNotFoundError: No module named 'langchain'"**

```bash
pip install -r requirements.txt
```

**"GROQ_API_KEY not found"**

```bash
# Create .env file with your API key
echo "GROQ_API_KEY=your-key-here" > .env
```

## 📄 License

This project is licensed under the MIT License.

---

**[⭐ Star this repo](https://github.com/linfieldlabs/Agentic_AI)** if you found it helpful!

</div>
