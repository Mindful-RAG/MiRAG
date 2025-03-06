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
| `--output-file` | string | `"mirag_output.json"` | Path to the output file where results will be saved |
| `--retry-attempts` | integer | `2` | Number of retry attempts for queries that fail during processing |
| `--continue-from-file` | flag | - | Continue processing from a previous output file, focusing on error entries |
| `--process-errors-only` | flag | - | Alias for `--continue-from-file`, process only the error entries in the output file |
| `--persist-index` | flag | - | Persist the vector index to disk for reuse in future runs |
| `--persist-path` | string | `"./persisted_index"` | Directory path where the index will be saved or loaded from |
| `--load-index` | flag | - | Load a previously persisted index instead of creating a new one |

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

### Index Management

Create and save index for future use:
```bash
uv run mirag --persist-index --persist-path ./my_project_index
```

Load a previously saved index:
```bash
uv run mirag --load-index --persist-path ./my_project_index
```

### Error Management

Process a file with previous errors:
```bash
uv run mirag --continue-from-file --output-file previous_run.json
```

Run with more retry attempts for failed queries:
```bash
uv run mirag --retry-attempts 5
```

### Combined Workflow

Complete workflow with index persistence and error handling:
```bash
uv run mirag --llm gpt-4o --persist-index --retry-attempts 3
```

Continue from a previous run using a saved index:
```bash
uv run mirag --load-index --persist-path ./my_project_index --continue-from-file
```

## Output Files

The script produces two main output files:
- `[output-file]`: Contains the detailed results for each query, including answers and error information
- `summary_[output-file]`: Contains summary statistics of the run, including accuracy metrics and completion percentage

## Notes

- The `TAVILY_API_KEY` environment variable is required for external search functionality
- The `OPENAI_API_KEY` environment variable is required for openai completions
- The script requires GPU acceleration for efficient embedding generation
- When using `--continue-from-file`, only failed entries are reprocessed, preserving successful ones
- Persistent indexes can significantly speed up subsequent runs by avoiding recomputing embeddings
- A successful run will update both the main output file and the summary file with current statistics
