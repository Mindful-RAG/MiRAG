import argparse

from icecream import ic

from utils.gemini import GeminiInference

# 1. load test data
# 2. setup models
# 3. load the data with tqdm, and process
# 4. save the results to json


def main():
    gemini = GeminiInference(model_name="gemini-2.0-flash")

    ic("Gemini response: ", gemini.content("the quick brown fox, continue this"))


if __name__ == "__main__":
    main()
