# Unified Codebase Context Platform

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python Version](https://img.shields.io/badge/python-3.8+-blue)
![License](https://img.shields.io/badge/license-MIT-orange)

The Unified Codebase Context Platform is a sophisticated system designed to create a deep, contextual understanding of a software project. It integrates code intelligence, project memory, and data ingestion to provide a comprehensive knowledge base for AI assistants, developers, and external tools.

This platform is built on the principles of three core concepts:
*   **RepoHyper:** For building a Code Property Graph (CPG) and performing semantic analysis on the codebase.
*   **ConPort:** For managing structured project knowledge like architectural decisions, progress logs, and system patterns.
*   **CocoIndex:** For creating flexible and incremental data ingestion pipelines for code and documentation.

---

## Core Features

*   **Code Property Graph (CPG):** Constructs a graph of code entities (functions, classes) and their relationships (calls, inheritance).
*   **Project Memory:** Persistently stores and manages structured project knowledge (decisions, progress, patterns) in a local database.
*   **Unified Search:** Perform semantic search across code, documentation, and project knowledge from a single query point.
*   **RAG Context Generation:** Generates rich, relevant context for Retrieval-Augmented Generation (RAG) prompts to be used with Large Language Models (LLMs).
*   **Code Completion Context:** Provides deep contextual information (surrounding code, related entities, project decisions) to enhance code completion suggestions.
*   **REST API:** A robust FastAPI-based API allows for easy integration with external tools, IDE extensions, and AI agents.
*   **Incremental Updates:** Designed to efficiently process changes in the codebase and documentation without full re-indexing.

---

## Architecture Overview

The platform is composed of several key modules that work together:

*   **`DataIngestionEngine` (`data_ingestion.py`):** Handles the ingestion and processing of source code and documentation. It's responsible for chunking files and preparing them for analysis and embedding.
*   **`CodeIntelligenceEngine` (`code_intelligence.py`):** Builds and manages the Code Property Graph (CPG). It performs semantic analysis, generates embeddings for code entities, and powers code-specific searches.
*   **`ProjectMemoryEngine` (`project_memory.py`):** Manages all structured project knowledge. It uses a SQLite database to store decisions, progress, patterns, and other contextual data.
*   **`QueryEngine` (`query_engine.py`):** The central hub that integrates all other engines. It orchestrates unified searches, generates RAG context, and provides comprehensive answers to complex queries.
*   **`API` (`api.py`):** Exposes the platform's capabilities through a RESTful API built with FastAPI, making it accessible to other services.
*   **`CLI` (`main.py`):** Provides a command-line interface for initializing the workspace, ingesting data, and running the API server.

---

## Getting Started

Follow these instructions to get the platform up and running on your local machine.

### Prerequisites

*   Python 3.8 or higher
*   Git

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repository-name>.git
cd <your-repository-name>
```

### 2. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
# For Unix/macOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

Install all the required Python packages using the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Configuration

The platform can be configured using environment variables. The main ones are:

*   `EMBEDDING_MODEL`: The sentence-transformer model to use for generating embeddings.
    *   **Default:** `Qodo/Qodo-Embed-1-1.5B`
*   `REPOHYPER_MODEL_PATH`: The path to the pre-trained Graph Neural Network (GNN) model for advanced code analysis.
    *   **Default:** `./models/repo_hyper_model.pt`

You can set them in your shell or using a `.env` file (though you would need to add `python-dotenv` to `requirements.txt` and load it in `config.py`).

### 5. Running the Platform

The `main.py` script is the main entry point for managing the platform.

**a. Place Your Code:**

Before you begin, place the source code you want to analyze into the `code_repo/` directory and any documentation into the `docs/` directory. (You may need to create these directories).

**b. Initialize the Workspace:**

This command sets up the SQLite database and necessary directories.

```bash
python -m unified_codebase_context.main --init
```

**c. Ingest Data and Build the Graph:**

This command processes the files in your `code_repo/` and `docs/` directories, and then builds the Code Property Graph.

```bash
python -m unified_codebase_context.main --ingest --build-graph
```

**d. Start the API Server:**

This command starts the FastAPI server, making the platform's features available via the API.

```bash
python -m unified_codebase_context.main --serve
```

The API will be running at `http://0.0.0.0:8000`.

---

## API Usage

Once the server is running, you can interact with the API. FastAPI provides automatic interactive documentation.

*   **Interactive Docs (Swagger UI):** http://127.0.0.1:8000/docs
*   **Alternative Docs (ReDoc):** http://127.0.0.1:8000/redoc

### Example: Unified Search

Here is an example of how to use `curl` to perform a unified search for the term "authentication".

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/search' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "authentication",
  "search_code": true,
  "search_docs": true,
  "search_knowledge": true,
  "top_k": 5
}'
```

---

## Contributing

Contributions are welcome! If you have ideas for new features, improvements, or have found a bug, please open an issue or submit a pull request.

1.  Fork the repository.
2.  Create a new feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

---

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.