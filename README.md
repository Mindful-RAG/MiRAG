# MiRAG

## Running the App

## Installation

The project uses [uv](https://github.com/astral-sh/uv)

# MiRAG: Mindful RAG Workflow - Command Line Arguments

This document describes the command-line arguments for the MiRAG (Mindful RAG) workflow script. MiRAG is a retrieval-augmented generation system that uses a mindful approach to improve the quality of retrieved information.

## Basic Usage

```bash
uv run mirag [arguments]
```

## Available Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--llm` | string | `"gpt-4o-mini"` | LLM model to use for generation tasks. Examples: `"gpt-4o-mini"`, `"gpt-4o"`, `"gemini-1.5-flash"` |
| `--embed_model` | string | `"BAAI/bge-large-en-v1.5"` | Embedding model to use for vector embeddings. Examples: `"BAAI/bge-large-en-v1.5"`, `"sentence-transformers/all-mpnet-base-v2"` |
| `--output-file` | string | `"mirag_output.jsonl"` | Path to the output file where results will be saved |
| `--retry-attempts` | integer | `2` | Number of retry attempts for queries that fail during processing |
| `--continue-from-file` | flag | - | Continue processing from a previous output file, focusing on error entries |
| `--process-errors-only` | flag | - | Alias for `--continue-from-file`, process only the error entries in the output file |
| `--vector-db-path` | string | `"./chroma_db"` | Directory path where the vector database will be stored |
| `--collection-name` | string | `"mirag_collection"` | Name of the collection in the vector database |
| `--use-existing-db` | flag | - | Use an existing vector database collection if available |

## Example Commands

### Basic Processing

Run with default settings:
```bash
uv run mirag
```

### Model Selection

Use a different LLM or embedding model:
```bash
uv run mirag --llm gpt-4o --embed_model "sentence-transformers/all-mpnet-base-v2"
```

### Vector Database Management

Create and use a specific vector database:
```bash
uv run mirag --vector-db-path ./my_project_vectors --collection-name my_project
```

Reuse an existing vector database:
```bash
uv run mirag --use-existing-db --vector-db-path ./my_project_vectors --collection-name my_project
```

### Error Management

Process a file with previous errors:
```bash
uv run mirag --continue-from-file --output-file previous_run.jsonl
```

Run with more retry attempts for failed queries:
```bash
uv run mirag --retry-attempts 5
```

### Combined Workflow

Complete workflow with vector database and error handling:
```bash
uv run mirag --llm gpt-4o --vector-db-path ./vectors --collection-name project_vectors --retry-attempts 3
```

Continue from a previous run using an existing vector database:
```bash
uv run mirag --use-existing-db --vector-db-path ./vectors --collection-name project_vectors --continue-from-file
```

## Output Files

The script produces two main output files:
- `[output-file]`: Contains the detailed results for each query, including answers and error information
- `summary_[output-file]`: Contains summary statistics of the run, including accuracy metrics and completion percentage

## Notes

- The `TAVILY_API_KEY` environment variable is required for external search functionality
- The `OPENAI_API_KEY` environment variable is required for OpenAI completions
- The script requires GPU acceleration for efficient embedding generation
- When using `--continue-from-file`, only failed entries are reprocessed, preserving successful ones
- Vector database collections can significantly speed up subsequent runs by avoiding recomputing embeddings
- ChromaDB is used as the vector database for storing and retrieving embeddings
- A successful run will update both the main output file and the summary file with current statistics
