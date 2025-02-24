import os
from dotenv import load_dotenv
from icecream import ic
from llama_index.core import SimpleDirectoryReader

from crag.utils import CorrectiveRAGWorkflow

load_dotenv()


async def main():
    documents = SimpleDirectoryReader("./data").load_data()
    workflow = CorrectiveRAGWorkflow()
    index = await workflow.run(documents=documents)

    response = await workflow.run(
        query_str="How was Llama2 pretrained?",
        index=index,
        tavily_ai_apikey=os.getenv("TAVILY_API_KEY"),
    )

    with open("crag_out.txt", "w") as f:
        f.write(str(response))
    ic(response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
