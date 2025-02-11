import os
from google import genai
from google.genai import types
import warnings
from dotenv import load_dotenv


load_dotenv()


class GeminiInference:
    def __init__(self, model_name="gemini-2.0-flash"):
        """
        Initializes the GeminiInference class.

        Args:
            model_name (str): Name of the Gemini model to use.
        """
        self.model_name = model_name
        self.client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY", ""),
        )

    def content(self, content):
        """
        Generates content using the Gemini model.

        Args:
            content (str): The content to generate.

        Returns:
            The generated content as a string.
        """
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=content,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=1000,
                response_mime_type="text/plain",
                safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                ],
            ),
        )
        return response.text

    def infer(self, input_data):
        """
        Performs inference using the Gemini model.

        Args:
            input_data: The input data for inference.

        Returns:
            The inference results as a dictionary.
        """
        # Simulated inference process.
        response = self.client.models.generate_content(
            model=self.model_name, contents=input_data
        )
        print("Running inference on input:", input_data)
        result = {"prediction": "dummy_result", "confidence": 0.99}
        return response.text


# Example usage:
if __name__ == "__main__":
    gemini = GeminiInference(model_name="gemini_v2")
    input_data = "Sample input data"
    output = gemini.infer(input_data)
    print("Inference output:", output)
