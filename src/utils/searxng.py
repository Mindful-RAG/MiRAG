import asyncio
from typing import Any, Dict, List, NamedTuple, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel


class Document(BaseModel):
    """Document class to represent search results."""

    title: str
    content: str
    url: str
    score: Optional[float] = None


class SearXNGClient:
    """A client for interacting with SearXNG search engine to fetch content for LLMs."""

    def __init__(
        self, instance_url: str = "http://localhost:8080", timeout: int = 10, max_retries: int = 3, retry_delay: int = 2
    ):
        """
        Initialize the SearXNG client.

        Args:
            instance_url: URL of the SearXNG instance
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.instance_url = instance_url
        self.search_url = f"{instance_url}/search"
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp client session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _test_connection(self) -> None:
        """Test the connection to the SearXNG instance."""
        try:
            session = await self._get_session()
            async with session.get(self.instance_url, timeout=self.timeout) as response:
                if response.status == 200:
                    logger.info(f"Successfully connected to SearXNG instance at {self.instance_url}")
                else:
                    logger.warning(f"SearXNG instance returned status code {response.status}")
        except aiohttp.ClientError as e:
            logger.error(f"Failed to connect to SearXNG instance: {e}")
            logger.error("Make sure SearXNG is running at the specified URL")

    async def search(
        self,
        query: str,
        max_results: int = 10,
        categories: str = "general",
        language: str = "en",
        page: int = 1,
        time_range: Optional[str] = None,
        engines: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Search SearXNG instance and return results.

        Args:
            query: The search query
            max_results: Maximum number of results to return
            categories: Search categories (general, images, videos, etc.)
            language: Language code
            page: Page number
            time_range: Time range for results (day, week, month, year)
            engines: List of specific search engines to use

        Returns:
            JSON response containing search results or None if error
        """
        # Parameters for the search
        params = {
            "q": query,
            "format": "json",
            "categories": categories,
            "language": language,
            "pageno": page,
            "safesearch": 0,  # No safe search filtering as we need comprehensive results for LLM
        }

        # Add optional parameters if provided
        if time_range:
            params["time_range"] = time_range

        if engines:
            params["engines"] = ",".join(engines)

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "application/json",
        }

        # Try the request with retries
        session = await self._get_session()
        for attempt in range(self.max_retries):
            try:
                async with session.get(
                    self.search_url, params=params, headers=headers, timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        results = await response.json()
                        # Limit the number of results
                        if "results" in results:
                            results["results"] = results["results"][:max_results]
                        return results
                    else:
                        logger.warning(f"Attempt {attempt + 1}/{self.max_retries}: Error {response.status}")

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries}: Request failed - {e}")

            # Don't sleep after the last attempt
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay)

        logger.error(f"Failed to get search results after {self.max_retries} attempts")
        return None

    async def get_content_for_llm(self, query: str, max_results: int = 5) -> List[Document]:
        """
        Get search results as a list of Document objects for LLM consumption.

        Args:
            query: The search query
            max_results: Maximum number of search results to include

        Returns:
            List of Document objects with search results
        """
        results = await self.search(query, max_results=max_results)
        documents = []

        if not results or "results" not in results or not results["results"]:
            # Return empty list if no results found
            return documents

        for result in results["results"]:
            title = result.get("title", "No title")
            content = result.get("content", "No content")
            url = result.get("url", "No URL")
            score = result.get("score")

            # Convert score to float if possible
            if score is not None:
                try:
                    score = float(score)
                except (ValueError, TypeError):
                    score = None

            documents.append(Document(title=title, content=content, url=url, score=score))

        return documents

    def format_results_as_text(self, documents: List[Document]) -> str:
        """
        Format a list of Documents into a readable text format.

        Args:
            documents: List of Document objects

        Returns:
            Formatted string with search results
        """
        if not documents:
            return "No search results found."

        formatted_results = []

        for i, doc in enumerate(documents, 1):
            formatted_result = [f"[{i}] {doc.title}", f"Source: {doc.url}", f"Content: {doc.content}"]
            formatted_results.append("\n".join(formatted_result))

        return "\n\n".join(formatted_results)

    async def fetch_knowledge(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Fetch knowledge structured for LLM input as dictionaries.

        Args:
            query: The search query
            max_results: Maximum number of results to return

        Returns:
            List of dictionaries with structured content
        """
        results = await self.search(query, max_results=max_results)

        if not results or "results" not in results:
            return []

        knowledge_items = []

        for result in results["results"]:
            score = result.get("score")
            if score is not None:
                try:
                    score = float(score)
                except (ValueError, TypeError):
                    score = None

            knowledge_items.append(
                {
                    "title": result.get("title", "No title"),
                    "content": result.get("content", "No content"),
                    "url": result.get("url", "No URL"),
                    "score": score,
                }
            )

        return knowledge_items


async def search_searxng(
    query: str, instance_url: str = "http://localhost:8080", max_results: int = 10
) -> Optional[Dict[str, Any]]:
    """
    Simple async function to search SearXNG instance.

    Args:
        query: The search query
        instance_url: URL of the SearXNG instance
        max_results: Maximum number of results to return

    Returns:
        JSON response containing search results
    """
    client = SearXNGClient(instance_url)
    try:
        return await client.search(query, max_results=max_results)
    finally:
        await client.close()


async def get_search_results_for_llm(
    query: str, max_results: int = 5, instance_url: str = "http://localhost:8080"
) -> List[Document]:
    """
    Get search results as Document objects suitable for feeding into an LLM.

    Args:
        query: The search query
        max_results: Maximum number of results to include
        instance_url: URL of the SearXNG instance

    Returns:
        List of Document objects with search results
    """
    client = SearXNGClient(instance_url)
    try:
        return await client.get_content_for_llm(query, max_results=max_results)
    finally:
        await client.close()


async def main():
    # Example: Get content formatted for an LLM
    query = 'What does "HP" stand for in War and Order?'

    # Method 1: Using the convenience function
    documents = await get_search_results_for_llm(query, max_results=5)
    for doc in documents:
        print(f"Title: {doc.title}")
        print(f"URL: {doc.url}")
        print(f"Content: {doc.content}")
        print(f"Score: {doc.score}")
        print("-" * 40)

    print("\n" + "=" * 50 + "\n")

    # Method 2: Using the client directly
    client = SearXNGClient()
    try:
        await client._test_connection()
        documents = await client.get_content_for_llm(query, max_results=5)
        print(documents)

        # Format the documents into text
        formatted_text = client.format_results_as_text(documents)
        print(formatted_text)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
