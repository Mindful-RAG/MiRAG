"""
The first step

Retrieves documents such as wikipedia
"""

import json
from datasets import load_dataset
from datasets.load import DatasetDict
from icecream import ic
import tiktoken


def main():
    test_data = load_dataset("TIGER-Lab/LongRAG", "nq", split="subset_100")
    ic(test_data)
    # enc = tiktoken.get_encoding("cl100k_base")
    # with open("test_data.json", "w") as f:
    #     json.dump(test_data[0], f)
    # f.write(str(test_data[0]))
    # ic(enc)
    # ic(len(enc.encode(test_data[0]["context"])))


if __name__ == "__main__":
    main()
