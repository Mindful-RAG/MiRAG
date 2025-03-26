import argparse


class CLI:
    @staticmethod
    def parse_arguments():
        """Parse command line arguments for MiRAG API"""
        parser = argparse.ArgumentParser(description="MiRAG: Mindful RAG API")

        # subparsers = parser.add_subparsers(dest="command", help="Command to run")

        # API command
        # api_parser = subparsers.add_parser("api", help="Run the MiRAG API server")
        parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the API server on")
        parser.add_argument("--port", type=int, default=8000, help="Port to run the API server on")
        parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

        args = parser.parse_args()

        return args
