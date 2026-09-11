# AI Agent for CV Matching and Job Recommendation Using Large Language Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-red.svg)](https://qdrant.tech/)
[![vLLM](https://img.shields.io/badge/Serving-vLLM-blueviolet.svg)](https://vllm.ai/)

## 1. Project Overview
This repository contains the end-to-end architecture and implementation of an autonomous **AI Agent for CV Matching and Job Recommendation** powered by parameter-efficient fine-tuned open-source Large Language Models (LLMs), dense vector retrieval (RAG), cross-encoder reranking, and explainable AI (XAI).

The system operates strictly **autonomously and locally** without reliance on third-party commercial LLM APIs, ensuring candidate data privacy, zero recurring API expenses, and deep domain adaptability.

---

## 2. Core Architecture & Inference Pipeline
```
[Candidate CV (PDF/DOCX)]
          │
          ▼
   [1. CV Parser] ──────────► [Structured JSON: Skills, Exp, Edu]
          │                                  │
          ▼                                  ▼
[2. Dense Embedding]                [Metadata Filtering]
          │                                  │
          └────────────────┬─────────────────┘
                           ▼
          [3. Vector DB Similarity Search (Qdrant)]
                           │
                           ▼
                 [Initial Top-K Jobs]
                           │
                           ▼
             [4. Cross-Encoder Reranker]
      (Multi-factor: Experience, Salary, Location)
                           │
                           ▼
              [Refined Top-N Job Matches]
                           │
                           ▼
        [5. HR AI Agent (LangGraph Reasoning)]
      (Analyzes match alignment vs. job criteria)
                           │
                           ▼
         [6. Explainable AI (XAI) Generation]
      (Calculates matching %, identifies missing skills,
       generates personalized recommendations)
                           │
                           ▼
     [7. Interactive Web UI Presentation to Candidate]
```

---

## 3. Technology Stack
* **Base Models:** `Qwen2.5-7B-Instruct` (Fine-tuned), `Qwen3-8B`
* **Embedding Model:** `BGE-M3` / `Qwen Embedding`
* **Reranker:** `BGE-Reranker-Large` (Cross-Encoder)
* **Agent Framework:** `LangGraph`, `LangChain`
* **LLM Serving Engine:** `vLLM` (PagedAttention, 4-bit/AWQ/GPTQ, continuous batching)
* **Backend API:** `FastAPI`, `Uvicorn`, `Pydantic v2`, `Redis`
* **Vector Database:** `Qdrant`
* **Fine-Tuning:** `PyTorch`, `Hugging Face Transformers`, `PEFT`, `TRL` (LoRA / QLoRA)
* **DevOps & Monitoring:** `Docker`, `Docker Compose`, `Kubernetes`, `MLflow`, `Prometheus`, `Grafana`

---

## 4. Repository Structure
```
SS9 - FinalSeason/
├── configs/               # System configurations (YAML)
├── data/                  # Data directories (raw, processed, datasets, evaluation)
├── docker/                # Dockerfile and Docker Compose manifests
├── docs/                  # Architecture specs and research drafts
├── eval/                  # IR and NLP evaluation metrics and benchmarks
├── knowledge-base/        # Project knowledge guides and system instructions
├── models/                # Checkpoints and LoRA adapter weights
├── notebooks/             # EDA and experimentation notebooks
├── scripts/               # Automation, scraper, and utility scripts
├── src/                   # Main application source code
│   ├── agent/             # LangGraph agent orchestration
│   ├── api/               # FastAPI backend endpoints
│   ├── core/              # Config, logging, and error handling
│   ├── embeddings/        # Embedding models service
│   ├── parser/            # CV & JD entity extraction
│   ├── reranker/          # Cross-encoder reranking
│   ├── scraper/           # Job portal crawlers
│   ├── vector_db/         # Qdrant client and collections
│   └── xai/               # Explainable AI & skill gap analysis
├── tests/                 # Unit and integration tests
└── training/              # Fine-tuning scripts and training configs
```

---

## 5. Team & Stakeholders
* **Supervisors:** Mr. Nguyễn Xuân Huy, Mr. Nguyễn Quốc Trung
* **Engineering Team:**
  * **Lê Trí Dũng** (Leader) - Student Code: SE196261
  * **Thái Thành Nhân** (Member) - Student Code: SE196293
  * **Nguyễn Danh Bằng** (Member) - Student Code: SE194871
