# HealthAdvisor – RAG-SMT

A retrieval-augmented generation (RAG) framework for grounded healthcare question answering with structured guardrails and reliability evaluation.

## Overview

HealthAdvisor is a research-driven system designed to reduce hallucinations in medical AI responses by combining:

- Embedding-based retrieval  
- Controlled large language model (LLM) generation  
- Structured validation mechanisms  
- Communication-aware modeling  

The goal is to build healthcare AI systems that are not only accurate, but accountable and interpretable in high-stakes contexts.


## Motivation

Large language models often generate fluent but unsupported medical claims. In healthcare settings, this is unsafe.

HealthAdvisor explores how retrieval-based grounding and structured guardrails can:

- Improve factual consistency  
- Reduce hallucinations  
- Increase response reliability  
- Provide measurable confidence signals  

This project shifts from creative generation toward constraint-aware system design.

## System Architecture

The framework consists of:

1️⃣ **Document Preprocessing**
- Cleaning and chunking health documents
- Metadata structuring

2️⃣ **Embedding Generation**
- Transformer-based embeddings (Hugging Face)
- Vector representation of knowledge sources

3️⃣ **Vector Indexing**
- FAISS similarity search
- Context retrieval prior to generation

4️⃣ **Controlled Generation**
- Retrieval-augmented prompt construction
- Guardrail-enhanced prompting strategies

5️⃣ **Evaluation & Confidence Scoring**
- Semantic similarity metrics
- Response completeness checks
- Confidence calibration signals
- Hallucination detection indicators


## Sense-Making Methodology (SMT) Integration

This repository also incorporates communication-aware modeling using Sense-Making Methodology (SMT).

Structured communication factors such as:
- Context
- Uncertainty
- Information gaps
- User literacy

are modeled to evaluate how AI responses interact with human interpretation dynamics.

This layer connects technical reliability with human-centered understanding.

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- FAISS
- NumPy / Pandas
- JSON schema validation tools


## Research Goals

- Reduce hallucinations in healthcare LLM systems
- Develop constraint-based generation frameworks
- Introduce measurable reliability metrics
- Bridge technical NLP systems with communication science theory

## Repository Structure

```text
HealthAdvisor--RAG-SMT/
│
├── data_processing/
├── embedding_pipeline/
├── vector_indexing/
├── generation/
├── evaluation/
├── communication_modeling/
└── README.md
