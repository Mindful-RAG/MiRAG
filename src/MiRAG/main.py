import numpy
import argparse

from utils.gemini import GeminiInference
from utils.model import hi


def simple_numpy_function():
    arr = numpy.array([1, 2, 3])
    return numpy.mean(arr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model to use")
    args = parser.parse_args()

    result = simple_numpy_function()

    GeminiInference(device="cpu", model_name="gemini-1.5-flash").infer("input")
    hi()
    if args.model:
        print("Using model:", args.model)
    print("The mean of the array is:", result)


if __name__ == "__main__":
    main()
