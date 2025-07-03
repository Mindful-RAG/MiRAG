# MiRAG

MiRAG (Mindful RAG) is a retrieval-augmented generation system that uses a mindful approach to improve the quality of retrieved information. The project provides both a command-line interface and a REST API server for processing queries.

## Installation

The project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install dependencies
uv sync

# Install development dependencies
uv sync --group dev
```

## Basic Usage

MiRAG supports two main modes of operation:

### CLI Mode (Default)
```bash
uv run mirag [arguments]
```

### API Server Mode
```bash
uv run api [arguments]
```

## CLI Command Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--llm` | string | `"gpt-4o-mini"` | LLM model to use for generation tasks |
| `--embed_model` | string | `"BAAI/bge-large-en-v1.5"` | Embedding model to use for vector embeddings |
| `--split` | string | `"subset_100"` | Dataset split to use |
| `--data_name` | string | `"nq"` | Benchmark name (e.g., "nq") |
| `--output-file` | string | `"mirag_output.jsonl"` | Output file for the workflow results |
| `--retry-attempts` | integer | `2` | Number of retry attempts for failed queries |
| `--continue-from-file` | flag | - | Continue processing from a previous output file |
| `--process-errors-only` | flag | - | Process only the error entries in the output file |
| `--persist-index` | flag | - | Persist the index to disk for future use |
| `--persist-path` | string | `"./persisted_index"` | Path to persist the index |
| `--load-index` | flag | - | Load index from disk instead of creating a new one |
| `--collection-name` | string | `"nq_corpus"` | Name of the collection to use for indexing |
| `--lfqa` | flag | - | Enable LFQA metric evaluation |
| `--lfqa-size` | integer | `500` | Size of LFQA dataset to use |
| `--debug` | flag | - | Enable debug logs |

## API Server Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--host` | string | `"0.0.0.0"` | Host to run the API server on |
| `--port` | integer | `8000` | Port to run the API server on |
| `--reload` | flag | - | Enable auto-reload for development |

## Environment Variables

MiRAG requires several environment variables to be set in a `.env` file:

### Required Variables
- `OPENAI_API_KEY` - Required when using OpenAI models
- `GOOGLE_API_KEY` - Required when using Gemini models
- `DEEPSEEK_API_KEY` - Required when using DeepSeek models

### Optional Variables
- `SEARXNG_URL` - URL for the SearXNG instance (defaults to "http://localhost:8003")
- `PERSIST_PATH` - Path to persist the index (defaults to "./persisted_index")
- `ENVIRONMENT` - Environment setting ("dev", "production", etc.)

### API-Specific Variables
- `DB_HOST` - Database host for DynamoDB local (defaults to "http://localhost:8040")
- `AWS_REGION` - AWS region (defaults to "ap-southeast-1")
- `AWS_ACCESS_KEY_ID` - AWS access key ID
- `AWS_SECRET_ACCESS_KEY` - AWS secret access key
- `TABLE_NAME` - DynamoDB table name for users
- `CHAT_STORE_TABLE_NAME` - DynamoDB table name for chat sessions
- `BUCKET_NAME` - S3 bucket name for file storage
- `SECRET_KEY` - Secret key for session management
- `JWT_SECRET_KEY` - Secret key for JWT tokens
- `FRONTEND_URL` - Frontend URL for CORS (defaults to "http://localhost:3000")
- `ALLOWED_ORIGINS` - Comma-separated list of allowed origins for CORS

### Firebase Configuration (for API authentication)
- `GOOGLE_APPLICATION_CREDENTIALS` - Path to Firebase service account key
- `FIREBASE_PROJECT_ID` - Firebase project ID
- `FIREBASE_CLIENT_EMAIL` - Firebase client email
- `FIREBASE_PRIVATE_KEY` - Firebase private key

### Azure OpenAI Configuration (optional)
- `AZURE_OPENAI_KEY1` - Azure OpenAI API key
- `AZURE_OPENAI_KEY2` - Azure OpenAI API key (backup)
- `AZURE_REGION` - Azure region
- `AZURE_OPENAI_ENDPOINT` - Azure OpenAI endpoint

## Supported Models

### LLM Models
- **OpenAI**: `gpt-4o-mini`, `gpt-4o`, `gpt-3.5-turbo`, etc.

### Embedding Models
- `BAAI/bge-large-en-v1.5` (default)
- `BAAI/bge-small-en-v1.5`
- `sentence-transformers/all-MiniLM-L6-v2`
- Any HuggingFace embedding model

## CLI Examples

### Basic Usage
```bash
# Run with default settings
uv run mirag

# Use a different LLM model
uv run mirag --llm gpt-4o

# Use a Gemini model
uv run mirag --llm gemini-1.5-flash

# Use DeepSeek model
uv run mirag --llm deepseek-chat

# Use Ollama model (requires local Ollama)
uv run mirag --llm llama3.2
```

### Index Management
```bash
# Create and persist an index
uv run mirag --persist-index --persist-path ./my_index

# Load a previously persisted index
uv run mirag --load-index --persist-path ./my_index

# Use a custom collection name
uv run mirag --collection-name my_custom_corpus
```

### Dataset and Evaluation
```bash
# Use a specific dataset split
uv run mirag --split subset_500 --data_name nq

# Enable LFQA evaluation
uv run mirag --lfqa --lfqa-size 1000

# Continue from a previous run
uv run mirag --continue-from-file --output-file previous_run.jsonl

# Process only failed queries
uv run mirag --process-errors-only --output-file previous_run.jsonl
```

### Debugging
```bash
# Enable debug logging
uv run mirag --debug
```

## API Server Examples

### Starting the Server
```bash
# Run with default settings
uv run api

# Run on a specific port with auto-reload
uv run api --port 9000 --reload

# Run on a specific host
uv run api --host 127.0.0.1 --port 8080
```

## API Endpoints

### Health Check
- `GET /health` - Check the health status of the API

### Authentication
- `POST /auth/login` - Login with Firebase token
- `POST /auth/logout` - Logout user
- `GET /auth/me` - Get current user info

### Chat and Query Processing
- `POST /chat/mirag` - Submit a query to the MindfulRAG workflow
- `POST /chat/longrag` - Submit a query to the LongRAG workflow
- `POST /chat/upload` - Upload a PDF file for indexing
- `DELETE /chat/upload/{corpus_id}` - Delete a custom corpus

### Request/Response Examples

#### MiRAG Query
```bash
curl -X POST "http://localhost:8000/chat/mirag" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the capital of France?",
    "llm_model": "gpt-4o-mini",
    "embed_model": "BAAI/bge-large-en-v1.5"
  }'
```

#### File Upload
```bash
curl -X POST "http://localhost:8000/chat/upload" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -F "file=@document.pdf"
```

## Output Files

The CLI mode produces output files containing:
- `[output-file]`: Detailed results for each query in JSONL format
- `summary_[output-file]`: Summary statistics of the run

## Prerequisites

### For CLI Mode
- Python 3.12+
- SearXNG instance running (for external search functionality)
- Required API keys based on chosen models

### For API Mode
- All CLI requirements
- AWS DynamoDB (local or cloud)
- AWS S3 (for file storage)
- Firebase project (for authentication)
- Docker (recommended for SearXNG)

## Docker Setup

A `docker-compose.yml` file is provided for running SearXNG locally:

```bash
docker-compose up -d
```

## Development

### Install development dependencies
```bash
uv sync --group dev
```

### Run linting
```bash
uv run ruff check
uv run ruff format
```

### Pre-commit hooks
```bash
uv run pre-commit install
```

## Architecture

- **CLI Mode**: Direct command-line interface for batch processing
- **API Mode**: REST API server with authentication and session management
- **Vector Store**: ChromaDB for embedding storage and retrieval
- **LLM Integration**: Support for multiple LLM providers
- **Authentication**: Firebase-based user authentication
- **Database**: DynamoDB for user and session management
- **Storage**: S3 for file storage and processing

## Notes

- For API usage, run the CLI mode at least once with `--persist-index` to create an index
- The API dynamically switches LLM models based on requests
- GPU acceleration is recommended for efficient embedding generation
- Rate limiting is enabled on API endpoints
- CORS is configured for web frontend integration
