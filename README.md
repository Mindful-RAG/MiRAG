# MiRAG

## Running the App

## Installation

The project uses [uv](https://github.com/astral-sh/uv)

# MiRAG: Mindful RAG Workflow - Command Line Arguments

This document describes the command-line arguments for the MiRAG (Mindful RAG) workflow. MiRAG is a retrieval-augmented generation system that uses a mindful approach to improve the quality of retrieved information.

## Basic Usage

```bash
uv run mirag [command] [arguments]
```

## Available Commands

MiRAG supports two main commands:

- `cli` - Run the MiRAG command-line interface (default if no command is specified)
- `api` - Run the MiRAG API server

## CLI Command Arguments

```bash
uv run mirag cli [arguments]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--llm` | string | `"gpt-4o-mini"` | LLM model to use for generation tasks |
| `--embed_model` | string | `"BAAI/bge-large-en-v1.5"` | Embedding model to use for vector embeddings |
| `--output-file` | string | `"mirag_output.jsonl"` | Output file for the workflow results |
| `--retry-attempts` | integer | `2` | Number of retry attempts for failed queries |
| `--continue-from-file` | flag | - | Continue processing from a previous output file |
| `--process-errors-only` | flag | - | Process only the error entries in the output file |
| `--persist-index` | flag | - | Persist the index to disk for future use |
| `--persist-path` | string | `"./persisted_index"` | Path to persist the index |
| `--load-index` | flag | - | Load index from disk instead of creating a new one |
| `--debug` | flag | - | Enable debug logs |

## API Command Arguments

```bash
uv run mirag api [arguments]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--host` | string | `"0.0.0.0"` | Host to run the API server on |
| `--port` | integer | `8000` | Port to run the API server on |
| `--reload` | flag | - | Enable auto-reload for development |

## Environment Variables

MiRAG relies on several environment variables that should be set in a `.env` file or in your environment:

- `OPENAI_API_KEY` - Required when using OpenAI models
- `GOOGLE_API_KEY` - Required when using Gemini models
- `SEARXNG_URL` - URL for the SearXNG instance (defaults to "http://localhost:8080")
- `USE_CUDA` - Set to "true" to enable GPU acceleration for embeddings
- `PERSIST_PATH` - Path to persist the index (can be overridden by --persist-path)

## Example Commands

### CLI Examples

Run with default settings:
```bash
uv run mirag cli
```

Use a different LLM model:
```bash
uv run mirag cli --llm gpt-4o
```

Use a Gemini model:
```bash
uv run mirag cli --llm gemini-1.5-flash
```

Create and persist an index:
```bash
uv run mirag cli --persist-index --persist-path ./my_index
```

Load a previously persisted index:
```bash
uv run mirag cli --load-index --persist-path ./my_index
```

Continue processing from a previous run, focusing on errors:
```bash
uv run mirag cli --continue-from-file --output-file previous_run.jsonl
```

### API Examples

Run the API server with default settings:
```bash
uv run mirag api
```

Run the API server on a specific port with auto-reload:
```bash
uv run mirag api --port 9000 --reload
```

## API Endpoints

When running the API server, the following endpoints are available:

- `POST /query` - Submit a query to the MiRAG workflow
  - Request body:
    ```json
    {
      "query": "Your question here",
      "llm_model": "gpt-4o-mini",
      "embed_model": "BAAI/bge-large-en-v1.5"
    }
    ```
  - Response includes short answer, long answer, status, and markdown-formatted results

- `GET /health` - Check the health status of the API

## Output Files

The CLI mode produces two main output files:
- `[output-file]`: Contains the detailed results for each query
- `summary_[output-file]`: Contains summary statistics of the run

## Notes

- Before using the API, you should run the CLI mode at least once with `--persist-index` to create an index
- The API dynamically switches LLM models based on requests, but this requires a brief initialization period
- The SearXNG instance must be running for external search functionality
- GPU acceleration is recommended for efficient embedding generation
