import argparse


class CLI:
    @staticmethod
    def parse_arguments():
        """Parse command line arguments for MiRAG workflow"""
        parser = argparse.ArgumentParser(description="MiRAG: Mindful RAG Workflow")

        subparsers = parser.add_subparsers(dest="command", help="Command to run")

        # CLI command
        cli_parser = subparsers.add_parser("cli", help="Run the MiRAG CLI")
        cli_parser.add_argument("--llm", type=str, default="gpt-4o-mini", help="LLM model to use")
        cli_parser.add_argument(
            "--embed_model",
            type=str,
            default="BAAI/bge-large-en-v1.5",
            help="embeddings model to use",
        )
        cli_parser.add_argument(
            "--split",
            type=str,
            default="subset_100",
            help="Dataset split to use",
        )
        cli_parser.add_argument(
            "--output-file",
            type=str,
            default="mirag_output.jsonl",
            help="Output of the workflow",
        )
        cli_parser.add_argument(
            "--retry-attempts",
            type=int,
            default=2,
            help="Number of retry attempts for failed queries",
        )
        cli_parser.add_argument(
            "--continue-from-file",
            action="store_true",
            help="Continue processing from a previous output file",
        )
        cli_parser.add_argument(
            "--process-errors-only",
            action="store_true",
            help="Process only the error entries in the output file",
        )
        cli_parser.add_argument(
            "--persist-index",
            action="store_true",
            help="Persist the index to disk",
        )
        cli_parser.add_argument(
            "--persist-path",
            type=str,
            default="./persisted_index",
            help="Path to persist the index",
        )
        cli_parser.add_argument(
            "--load-index",
            action="store_true",
            help="Load index from disk instead of creating a new one",
        )
        cli_parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug logs",
        )

        # API command
        api_parser = subparsers.add_parser("api", help="Run the MiRAG API server")
        api_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the API server on")
        api_parser.add_argument("--port", type=int, default=8000, help="Port to run the API server on")
        api_parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

        # If no arguments, default to CLI
        args = parser.parse_args()
        if not args.command:
            args.command = "cli"

        return args
