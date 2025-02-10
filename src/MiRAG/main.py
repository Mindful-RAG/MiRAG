import argparse

from icecream import ic

from utils.gemini import GeminiInference

# 1. load test data
# 2. setup models
# 3. load the data with tqdm, and process
# 4. save the results to json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model to use")
    args = parser.parse_args()

    gemini = GeminiInference(model_name="gemini-2.0-flash")

    ic("Gemini response: ", gemini.infer("hello gemini!"))
    if args.model:
        ic("Using model:", args.model)


if __name__ == "__main__":
    main()
