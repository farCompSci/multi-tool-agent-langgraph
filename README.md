# Multi-Tool Agent with LangGraph

This repository showcases an intelligent, autonomous AI agent developed using LangGraph. It is designed to interpret complex natural language tasks, decompose them into actionable subgoals, and execute them by orchestrating various specialized tools (e.g., coding, web search, mathematical calculations, document summarization, and general chat).

The agent aims to provide a robust and flexible framework for handling multi-step user queries, demonstrating advanced reasoning and tool-use capabilities.

## Features

*   **Complex Task Interpretation:** Understands and processes intricate natural language requests.
*   **Dynamic Task Decomposition:** Breaks down complex queries into smaller, manageable subtasks.
*   **Intelligent Tool Selection:** Automatically chooses the most appropriate tool for each subtask (e.g., `code` for programming, `search` for information retrieval, `math` for calculations, `summarize` for documents, `general_chat` for conversational responses).
*   **Sequential Task Execution:** Executes subtasks in a logical order, often chaining results from one task to the next.
*   **Stateful Workflow Management:** Leverages LangGraph for robust state management and control flow.
*   **Local LLM Integration:** Configured to work with local Ollama models (e.g., `llama3.2`, `qwen2.5-coder:3b`) and OpenAI models (e.g., `gpt-4o-mini`) for flexible deployment.
*   **Code Security Evaluation:** Includes a basic security evaluation step for generated code before execution.
*   **Memory Management:** Stores and recalls relevant information to maintain context across tasks.

## Project Structure

The core logic is organized as follows:

*   `main_graph.py`: Defines the LangGraph state graph, including nodes and edges for the agent's workflow.
*   `supervisor_operations.py`: Contains the core logic for task classification, decomposition, and queue processing.
*   `graph_nodes/`: Directory containing individual tool-specific nodes (e.g., `coder_node`, `search_node`, `math_nodes`, `document_summarizer_node`).
*   `helpers/`: Utility functions for model configuration, code security, summarization, and supervisor-related operations.
*   `app.py`: The Streamlit application interface for interacting with the agent.
*   `benchmarks/`: Contains complex queries and logs used for evaluating the agent's performance.

## Setup and Installation

To get this project up and running locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:farCompSci/multi-tool-agent-langgraph.git
    cd multi-tool-agent-langgraph
    ```

2.  **Install `uv` (if you don't have it):**
    `uv` is a fast Python package installer and resolver.
    ```bash
    pip install uv
    ```

3.  **Install dependencies using `uv`:**
    This project uses `pyproject.toml` for dependency management.
    ```bash
    uv sync
    ```
    This will create a virtual environment (`.venv`) and install all required packages.

4.  **Activate the virtual environment:**
    ```bash
    source .venv/bin/activate
    ```

5.  **Set up Ollama (for local LLMs):**
    *   Download and install [Ollama](https://ollama.com/download).
    *   Pull the required models:
        ```bash
        ollama pull llama3.2
        ollama pull qwen2.5-coder:3b
        ```

6.  **Set up OpenAI API Key (for OpenAI models):**
    *   If you plan to use OpenAI models (e.g., `gpt-4o-mini` for `classify_task_node`), set your OpenAI API key as an environment variable:
        ```bash
        export OPENAI_API_KEY="your_openai_api_key_here"
        ```
    *   You can also add this to a `.env` file in the root directory, and the `python-dotenv` package will load it.
    *   You also need to setup Wolfram Alpha, and Tavily Search API keys and set them in `env`

7.  **Initialize ChromaDB (for memory/retrieval):**
    The agent uses ChromaDB for long-term memory. Ensure the `chroma_db/` directory is created and accessible. The first run might initialize it.

## Usage

To run the Streamlit application and interact with the agent:

1.  **Activate your virtual environment** (if not already active):
    ```bash
    source .venv/bin/activate
    ```

2.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

    This will open the application in your web browser, where you can input complex queries and observe the agent's responses.

## Evaluation

The `benchmarks/` directory contains `complex_queries.py` and corresponding log files (`queryX_logs.txt`) used for evaluating the agent's performance on multi-step tasks. The `performance_analysis.md` file can be used to document the evaluation metrics and results.

## Contributing

Feel free to fork the repository, open issues, or submit pull requests.
