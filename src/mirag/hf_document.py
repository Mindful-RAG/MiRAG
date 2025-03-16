from typing import List

from llama_index.core import Document


def hf_dataset_to_documents(dataset, text_field, metadata_fields=None) -> List[Document]:
    """
    Transform a Hugging Face dataset into a list of LlamaIndex Document objects.

    Args:
        dataset_name (str): Name of the Hugging Face dataset
        text_field (str): Field in the dataset containing the document text
        metadata_fields (list, optional): Fields to include in document metadata

    Returns:
        list: List of LlamaIndex Document objects
    """

    documents = []

    # Convert each dataset item to a LlamaIndex Document
    for item in dataset:
        # Extract the text content
        text = item[text_field]

        # Extract metadata if specified
        metadata = {}
        if metadata_fields:
            for field in metadata_fields:
                if field in item and field != text_field:
                    metadata[field] = item[field]

        # Create a LlamaIndex Document
        doc = Document(text=text, metadata=metadata)
        documents.append(doc)

    return documents
