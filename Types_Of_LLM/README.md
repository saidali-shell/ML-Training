# LLM-Types

> A collection of lightweight, CPU-friendly proof-of-concept implementations demonstrating different categories of Large Language Model (LLM) architectures and supporting AI components.

The project showcases how to build semantic retrieval, image captioning, text generation, embeddings, and latent-space experiments using entirely open-source models—without requiring GPUs or API keys.

---

## Features

* CPU-only execution
* Open-source models only
* No API keys required
* Lightweight and easy to understand
* Independent demo scripts
* Windows-friendly setup

---

# Table of Contents

* [Overview](#overview)
* [Project Modules](#project-modules)
* [Installation](#installation)
* [Usage](#usage)
* [Project Structure](#project-structure)
* [Extending the Pipeline](#extending-the-pipeline)
* [Requirements](#requirements)
* [License](#license)

---

# Overview

This repository demonstrates several common AI and LLM building blocks:

* **Semantic Concept Retrieval**
* **Sentence Embeddings**
* **Image Captioning**
* **Causal Language Models**
* **Latent Space Mapping**
* **Mixture-of-Experts (MoE)**

Rather than being a single application, each script is intentionally small and self-contained so the underlying concepts are easy to understand and modify.

---

# Project Modules

| Module     | Purpose                   | Description                                                                                                                      |
| ---------- | ------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **lcm.py** | Language Concept Model    | Encodes domain concepts, performs semantic similarity search, and expands retrieved concepts through a NetworkX knowledge graph. |
| **slm.py** | Sentence Language Model   | Demonstrates sentence embeddings using a lightweight Sentence Transformer.                                                       |
| **gpt.py** | Generative Language Model | Minimal wrapper around a causal language model (TinyLlama, Qwen, etc.) using Hugging Face Transformers.                          |
| **vlm.py** | Vision Language Model     | Performs image captioning using the BLIP model entirely on CPU.                                                                  |
| **lam.py** | Latent-space Mapping      | Demonstrates projecting embedding vectors into a lower-dimensional latent representation.                                        |
| **moe.py** | Mixture-of-Experts        | Placeholder implementation for future expert-routing experiments.                                                                |

---

# Installation

## 1. Clone the repository

```powershell
git clone <repository-url>
cd LLM-Types
```

Or navigate directly to the existing folder:

```powershell
cd C:\Users\MuhammedSaidali\Desktop\LLM-Types
```

---

## 2. Create a virtual environment

```powershell
python -m venv venv
```

Activate it:

### PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

### Command Prompt

```cmd
venv\Scripts\activate.bat
```

---

## 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

---

## First Run

During the initial execution, the project downloads the **all-MiniLM-L6-v2** Sentence Transformer model (~100 MB).

This occurs only once. Subsequent runs use the locally cached model.

---

# Usage

## Concept Retrieval (`lcm.py`)

Builds a semantic index of concepts, retrieves the most relevant ones for a user query, and expands them using a small knowledge graph.

```powershell
python lcm.py
```

Example output:

```text
Concepts:
['PCI compliance', 'RAG']

Related nodes:
['tokenization', 'encryption']
```

---

## Vision Language Model (`vlm.py`)

Generate captions for images using the BLIP image-captioning model.

```powershell
python vlm.py
```

Notes:

* Creates a dummy white image if `invoice.png` is missing.
* Replace `invoice.png` with your own image for custom captions.

---

## Text Generation (`gpt.py`)

Generate text with a causal language model.

```powershell
python gpt.py "Explain the benefits of vector databases."
```

---

## Sentence Embeddings (`slm.py`)

Generate sentence embeddings.

```powershell
python slm.py "Your sentence here."
```

---

## Latent Mapping (`lam.py`)

Example script demonstrating latent-space projection.

```powershell
python lam.py
```

---

## Mixture of Experts (`moe.py`)

Placeholder for future routing and expert-selection experiments.

```powershell
python moe.py
```

---

# Project Structure

```text
LLM-Types/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── lcm.py      # Language Concept Model
├── slm.py      # Sentence Language Model
├── gpt.py      # Generative Language Model
├── vlm.py      # Vision Language Model
├── lam.py      # Latent-space Mapping
├── moe.py      # Mixture-of-Experts
│
└── venv/       # Local virtual environment (generated)
```

---

# Extending the Pipeline

The semantic retrieval workflow in **lcm.py** can serve as the foundation for a Retrieval-Augmented Generation (RAG) system.

Possible extensions include:

## 1. Expand the Concept Library

Add more domain-specific concepts to improve semantic retrieval quality.

```python
concepts = [
    ...
]
```

---

## 2. Build a Richer Knowledge Graph

Increase the number of nodes and edges or load them from an external source such as JSON, CSV, or Neo4j.

---

## 3. Connect to an LLM

Use the retrieved concepts and neighboring graph nodes as contextual information when prompting an LLM.

```python
prompt = f"""
Answer the user's question using the following context.

Concepts:
{', '.join(selected)}

Related Terms:
{', '.join(neighbors)}

User Query:
{user}
"""
```

The generated prompt can then be passed directly to `gpt.py` or another language model.

---

## 4. Cache Embeddings

Avoid recomputing embeddings for large concept collections.

```python
np.save("concepts.npy", embeddings)
```

Later:

```python
embeddings = np.load("concepts.npy")
```

---

# Requirements

Main dependencies include:

* transformers
* sentence-transformers
* torch
* networkx
* numpy
* Pillow

All package versions are pinned in `requirements.txt` for reproducibility.

---

# License

This project is provided **as-is** for educational and research purposes.

You are free to:

* Use
* Modify
* Fork
* Extend
* Integrate

into your own projects.

No warranty or guarantees are provided.

---

## Contributing

Contributions, suggestions, and improvements are welcome.

If you encounter bugs, dependency issues, or model download problems, feel free to open an issue or submit a pull request.

---

# Happy Hacking!

Experiment, modify, and build upon these examples to explore different LLM architectures and AI pipelines.
