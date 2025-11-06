# 🚀 **SwarmLens — Generative AI Agent Pro**

<div align="center">

![SwarmLens](https://img.shields.io/badge/🤖-SwarmLens_Generative_AI_Agent-blueviolet)
![LangGraph](https://img.shields.io/badge/🧩-LangGraph_Workflow-ff6b6b)
![RAG](https://img.shields.io/badge/🔍-RAG_Pipeline-4ecdc4)
![Streamlit](https://img.shields.io/badge/🎨-Beautiful_UI-45b7d1)

**An Intelligent RAG-Powered AI Agent built using LangGraph-style Architecture**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 🌟 Overview

**SwarmLens — Generative AI Agent Pro** is a **Retrieval-Augmented Generation (RAG)** system that uses a **LangGraph-style 4-node workflow**:
`plan → retrieve → answer → reflect`.
It intelligently fetches relevant information from a local knowledge base, generates human-like responses using an LLM, and evaluates the relevance of its answers.

> 🧠 Designed for the SwarmLens Internship Assignment — this project demonstrates mastery of RAG pipelines, LangGraph architecture, and multi-model orchestration.

---

## ⚙️ Core Features

### 🤖 Multi-AI Model Support

* **Gemini 1.5 Flash** — Deep reasoning & creative insights
* **Groq Llama 3.1** — Lightning-fast factual Q&A
* **Smart Router** — Auto-selects the best model for the query type

### 🔍 Advanced RAG Pipeline

* **ChromaDB Vector Store** for semantic search
* **Hugging Face Embeddings** via Sentence Transformers
* **Context-Aware Retrieval** for precise knowledge fetching

### 🧩 LangGraph-Inspired Workflow

| Node            | Description                                           |
| --------------- | ----------------------------------------------------- |
| 🧠 **Plan**     | Understands the user query and decides retrieval need |
| 📚 **Retrieve** | Fetches relevant chunks from vector DB                |
| 💬 **Answer**   | Generates contextual response using LLM               |
| 🔎 **Reflect**  | Evaluates quality and completeness of the answer      |

### 🎨 Beautiful Streamlit UI

* Gradient header with card-based layout
* Animated progress indicators
* Expandable result sections
* Fully responsive (Desktop + Mobile)

---

## 🖼️ Demo Preview

> 📸 *Add your screenshots here later (UI, dashboard, analytics, etc.)*

```
[Insert Demo Image 1 — Q&A Interface]
[Insert Demo Image 2 — Retrieval Context View]
[Insert Demo Image 3 — Reflection & Confidence Metrics]
```

---

## 🧠 Tech Stack

| Component  | Technology                                  |
| ---------- | ------------------------------------------- |
| Framework  | LangGraph-style (custom implementation)     |
| RAG        | LangChain / ChromaDB                        |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`)  |
| LLMs       | Gemini, Groq, (fallback: OpenAI)            |
| UI         | Streamlit                                   |
| Storage    | Local Vector DB (Chroma Persistent Storage) |
| Evaluation | Reflection Node + Optional LLM-as-a-Judge   |

---

## 🚀 Quick Start

### 🧩 Prerequisites

* Python **3.8+**
* Internet for model API calls
* API keys for Gemini / Groq *(Configured in code)*

### 🧰 Setup & Installation

```bash
# Clone the project
git clone https://github.com/your-repo/swarmlens-genai-agent.git
cd swarmlens-genai-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### ▶️ Run the App

```bash
streamlit run SwarmLens_GenerativeAI_Agent.py
```

Then open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 🧾 Project Structure

```
swarmlens-genai-agent/
├── SwarmLens_GenerativeAI_Agent.py   # Main LangGraph-style workflow
├── requirements.txt                   # Dependencies
├── README.md                          # Documentation
├── corpus/                            # Knowledge base (TXT/PDF files)
└── chroma_db/                         # Vector store (auto-created)
```

---

## 🧭 How It Works

1. **User Input:** You ask a question
2. **Plan Node:** Agent decides retrieval strategy
3. **Retrieve Node:** Fetches top relevant chunks using embeddings
4. **Answer Node:** Generates human-like answer via LLM
5. **Reflect Node:** Evaluates response accuracy and relevance

---

## 📊 Evaluation Metrics

* **Confidence Score** — Based on text relevance overlap
* **Reflection Verdict** — “Relevant” or “Needs Improvement”
* **LLM-as-a-Judge (Optional)** — Future upgrade
* **Performance Metrics** — Response time, context precision

---

## 🔐 Configuration

**Pre-configured API Keys (Environment variables):**

```
GEMINI_API_KEY=
GROQ_API_KEY=
OPENAI_API_KEY=
```

You can set these in your terminal:

```bash
export GEMINI_API_KEY="your_key"
export GROQ_API_KEY="your_key"
```

---

## 🧩 Bonus Integrations (Optional)

* **LangSmith** — Trace every agent node and pipeline execution
* **TruLens** — Evaluate RAG quality with transparency
* **Gradio Interface** — Alternate lightweight UI

---

## 🧠 Sample Questions

Try these:

1. *“What are the benefits of renewable energy?”*
2. *“Explain the three pillars of sustainability.”*
3. *“Compare solar and wind energy technologies.”*
4. *“How does carbon capture technology work?”*

---

## 💡 Challenges & Learnings

During implementation:

* Ensured **consistent embeddings and Chroma indexing**
* Balanced **retrieval granularity** vs **performance**
* Built a **reflection node** for relevance scoring without extra API costs
* Designed the **UI with Streamlit** to visualize each agent step

> 🔍 *Future work:* Integrate BLEU / ROUGE / BERTScore-based evaluation and add support for Claude or GPT-4.

---

## 📈 Future Enhancements

* [ ] Multi-language support
* [ ] Custom document uploads
* [ ] Enhanced evaluation metrics (BERTScore, RAGAs)
* [ ] Authenticated user sessions
* [ ] Full LangSmith analytics dashboard

---

## 🤝 Acknowledgments

* **SwarmLens Team** — for the opportunity to build this
* **LangChain / LangGraph** — for the workflow inspiration
* **ChromaDB** — for efficient vector search
* **Streamlit** — for creating a smooth UX
* **Open Source Community** — for shared resources

---

<div align="center">

**Built with ❤️ and RAG Power by Tanmay Kshirsagar**
**Final Year | Data Science & AI Developer**

[![GitHub](https://img.shields.io/badge/GitHub-Portfolio-black?logo=github)](https://github.com/your-github)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](https://linkedin.com/in/your-linkedin)

*If you found this project interesting, give it a ⭐ on GitHub!*

</div>

---

## 🧩 Quick Commands

```bash
# Install requirements
pip install -r requirements.txt

# Run locally
streamlit run SwarmLens_GenerativeAI_Agent.py
```

**Happy Coding & Keep Innovating! ⚡**

---
