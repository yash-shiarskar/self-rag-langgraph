# Self-RAG with LangGraph

A production-style implementation of **Self-RAG (Self-Reflective Retrieval-Augmented Generation)** using:

- LangGraph
- LangChain
- OpenAI GPT Models
- FAISS Vector Database
- Tavily Web Search
- Pydantic Structured Outputs

This system decides when retrieval is needed, evaluates retrieved documents, rewrites queries when necessary, and performs web search fallback to improve answer quality.

---

## Architecture

```text
User Query
     │
     ▼
Retrieval Decision
     │
 ┌───┴──────────┐
 │              │
 ▼              ▼
Direct LLM    Retrieve Docs
Response           │
                   ▼
          Relevance Grading
                   │
          ┌────────┴────────┐
          │                 │
          ▼                 ▼
 Generate Answer     Rewrite Query
 From Context             │
                          ▼
                    Web Search
                          │
                          ▼
                    Final Answer
```

---

## Features

### Intelligent Retrieval Decision

The system first determines whether external knowledge is required.

- Simple questions → Direct LLM response
- Knowledge-intensive questions → Retrieval pipeline

---

### Document Retrieval

Relevant documents are retrieved from a FAISS vector store using OpenAI embeddings.

Supported document types:

- PDF files
- Company Policies
- Product Documentation
- Business Profiles
- Knowledge Bases

---

### Relevance Grading

Retrieved documents are evaluated before being used.

Benefits:

- Removes noisy retrievals
- Reduces hallucinations
- Improves answer accuracy

---

### Context-Aware Generation

Answers are generated strictly from retrieved context.

The model avoids using outside knowledge when context is available.

---

### Query Rewriting

When retrieved documents are not relevant:

- Original query is rewritten
- Search intent is optimized
- Better retrieval opportunities are created

---

### Web Search Fallback

If internal knowledge is insufficient:

- Query is converted into search keywords
- Tavily Search is invoked
- Fresh web information is retrieved

---

## Technology Stack

| Component | Technology |
|------------|------------|
| Orchestration | LangGraph |
| LLM | OpenAI GPT-4o Mini |
| Embeddings | OpenAI text-embedding-3-large |
| Vector Store | FAISS |
| Search | Tavily Search |
| Validation | Pydantic |
| Retrieval Framework | LangChain |

---

## Project Structure

```text
self-rag/
│
├── documents/
│   ├── Company_Policies.pdf
│   ├── Company_Profile.pdf
│   └── Product_and_Pricing.pdf
│
├── self_rag_web.ipynb
│
├── README.md
│
└── requirements.txt
```

---

## Workflow

### Step 1: Retrieval Decision

Determine whether retrieval is required.

```python
should_retrieve = True or False
```

---

### Step 2: Retrieve Documents

Retrieve top matching documents from FAISS.

```python
retriever.invoke(question)
```

---

### Step 3: Relevance Grading

Evaluate whether retrieved content helps answer the question.

```python
is_relevant = True or False
```

---

### Step 4: Generate from Context

Generate grounded answers using only retrieved context.

---

### Step 5: Query Rewriting

Rewrite the query when no relevant documents are found.

Example:

```text
Original:
Who won the Aus vs Zim T20 match?

Rewritten:
Australia Zimbabwe T20 match result 2026 top scorer
```

---

### Step 6: Web Search

Perform Tavily search using rewritten query.

---

### Step 7: Final Response

Return the best available answer.

---

## Installation

```bash
git clone https://github.com/yourusername/self-rag.git

cd self-rag

pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file.

```env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

---

## Example Query

```text
Who won the Aus vs Zim World T20 match 2026 and who was the top scorer?
```

Pipeline:

```text
Question
 → Retrieval Decision
 → Document Search
 → Relevance Check
 → Query Rewrite
 → Web Search
 → Final Answer
```

---

## Research Inspiration

This project is inspired by:

### Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection

Authors:

- Akari Asai
- Zequn Xu
- Khyathi Raghavi Chandu
- Jason Wei
- Yizhong Wang
- Others

Paper:

**Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection**

arXiv:2310.11511

Research Link:

https://arxiv.org/abs/2310.11511

---

## Future Improvements

- Hallucination Grader
- Answer Quality Grader
- Multi-Hop Retrieval
- Hybrid Search (BM25 + Vector Search)
- Citation Generation
- Agentic RAG Extensions
- Human-in-the-Loop Evaluation

---

## License

MIT License

---

## Author

Yash Santosh Shiraskar

Electronics & Telecommunication Engineering

Building AI Agents, RAG Systems, LangGraph Workflows, and Agentic AI Applications.
