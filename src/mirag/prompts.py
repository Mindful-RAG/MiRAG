from llama_index.core import PromptTemplate


DEFAULT_RELEVANCY_PROMPT_TEMPLATE = PromptTemplate(
    template="""As a grader, your task is to evaluate the relevance of a document retrieved in response to a user's question.

    Retrieved Document:
    -------------------
    {context_str}

    User Question:
    --------------
    {query_str}

    Evaluation Criteria:
    - Consider whether the document contains keywords or topics related to the user's question.
    - The evaluation should not be overly stringent; the primary objective is to identify and filter out clearly irrelevant retrievals.

    Decision:
    - Assign a binary score to indicate the document's relevance.
    - Use 'yes' if the document is relevant to the question, or 'no' if it is not.

    Please provide your binary score ('yes' or 'no') below to indicate the document's relevance to the user question."""
)

DEFAULT_TRANSFORM_QUERY_TEMPLATE = PromptTemplate(
    template="""Your task is to refine a query to ensure it is highly effective for retrieving relevant search results. \n
    Analyze the given input to grasp the core semantic intent or meaning. \n
    Original Query:
    \n ------- \n
    {query_str}
    \n ------- \n
    Your goal is to rephrase or enhance this query to improve its search performance. Ensure the revised query is concise and directly aligned with the intended search objective. \n
    Respond with the optimized query only:"""
)

PREDICT_LONG_ANSWER = PromptTemplate(
    template="""\
    Go through the following context and then extract the answer of the question from the context.
    Answer the question directly. Your answer should be very concise.

    The context is a list of Wikipedia documents, ordered by title: {titles}
    Each Wikipedia document contains a 'title' field and a 'text' field.

    The context is: {context}.
    The question: {question}.\
    """
)

EXTRACT_ANSWER = PromptTemplate(
    template="""\
    As an AI assistant, you have been provided with a question and its long answer.
    Your task is to derive a very concise short answer, extracting a substring from the given long answer.
    Short answer is typically an entity without any other redundant words.
    It's important to ensure that the output short answer remains as simple as possible.

    {examples}

    Question: {question}
    Long Answer: {long_answer}
    Short Answer:\
    """
)
