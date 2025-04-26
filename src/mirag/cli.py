import argparse


class CLI:
    @staticmethod
    def parse_arguments():
        """Parse command line arguments for MiRAG workflow"""
        parser = argparse.ArgumentParser(description="MiRAG: Mindful RAG Workflow")

        parser.add_argument("--llm", type=str, default="gpt-4o-mini", help="LLM model to use")
        parser.add_argument(
            "--embed_model",
            type=str,
            default="BAAI/bge-large-en-v1.5",
            help="embeddings model to use",
        )
        parser.add_argument(
            "--split",
            type=str,
            default="subset_100",
            help="Dataset split to use",
        )
        parser.add_argument(
            "--output-file",
            type=str,
            default="mirag_output.jsonl",
            help="Output of the workflow",
        )
        parser.add_argument(
            "--retry-attempts",
            type=int,
            default=2,
            help="Number of retry attempts for failed queries",
        )
        parser.add_argument(
            "--continue-from-file",
            action="store_true",
            help="Continue processing from a previous output file",
        )
        parser.add_argument(
            "--process-errors-only",
            action="store_true",
            help="Process only the error entries in the output file",
        )
        parser.add_argument(
            "--persist-index",
            action="store_true",
            help="Persist the index to disk",
        )
        parser.add_argument(
            "--persist-path",
            type=str,
            default="./persisted_index",
            help="Path to persist the index",
        )
        parser.add_argument(
            "--load-index",
            action="store_true",
            help="Load index from disk instead of creating a new one",
        )
        parser.add_argument("--data_name", type=str, default="nq", help="Benchmark name")
        parser.add_argument("--lfqa", action="store_true", help="lfqa metric")
        parser.add_argument("--lfqa-size", type=int,default=500, help="lfqa metric")
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug logs",
        )

        args = parser.parse_args()

        return args
