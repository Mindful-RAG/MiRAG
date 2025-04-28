from llama_index.core import PromptTemplate

DEFAULT_RELEVANCY_PROMPT_TEMPLATE = PromptTemplate(
    template="""As a grader, your task is to evaluate the relevance of a document retrieved in response to a user's question.

    Retrieved Document:
    -------------------
    {context_str}

    Metadata:
    ---------
    {metadata}

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

EVALUATE_QUERY = PromptTemplate(
    template="""\
    You are a query evaluator. Your task is to classify user inputs into one of the following types:

    1. **Question** – The input is seeking specific information or an answer. It usually ends with a question mark or starts with question words like "what", "how", "why", etc.

    2. **Conversational** – The input is more casual or interactive in nature. It may express thoughts, feelings, or initiate/continue a conversation without explicitly asking a question.

    3. **Chat** – The input is a chat message or a part of a conversation. It may be a response to a previous message or a standalone message.

    If the input is neither of the above (e.g., a statement or command), return: "neither".

    Respond with only one word: "question", "conversational", "chat", or "neither".

    Examples:
    - "What is the capital of Japan?" -> question
    - "Hey, what's up?" -> conversational
    - "The server is running." -> neither
    - "Tell me about the new update." -> conversational
    - "Who won the game yesterday?" -> question
    - "What was the previous question?" -> chat
    - "Based from the chat or conversation, what's the summary?" -> chat

    Now classify the following prompt:
    {query_str}
    """
)

PREDICT_ANSWER = PromptTemplate(
    template="""\
    Go through the following context and then get the  answer of the question from the context.
    Choose the most relevant text from the context and answer from it.

    Context:
    -------
    {context}
    -------

    Question:
    -------
    {question}
    -------\
    """
)

DEFAULT_TRANSFORM_QUERY_TEMPLATE = PromptTemplate(
    template="""\
    You are a question re-writer that converts an input question to a better version that is optimized
    for web search. Look at the input and try to reason about the underlying semantic intent / meaning.
    Original Query:
    -------
    {query_str}
    -------
    Formulate an improved question.\
    """
)

PREDICT_LONG_ANSWER_NQ = PromptTemplate(
    template="""\
    Go through the following context and then extract the answer of the question from the context.
    Answer the question directly. Your answer should be concise.

    The context is a list of Wikipedia documents, ordered by title: {titles}.
    Each Wikipedia document contains a 'title' field and a 'text' field.

    Choose the most relevant text from the context and extract the answer from it.

    The context is: {context}.
    The question: {question}.\
    """
)
PREDICT_LONG_ANSWER_QA = PromptTemplate(
    template="""\
    Go through the following context and then answer the question

    The context is a list of Wikipedia documents titled: {titles}.

    There are two types of questions: comparison questions, which require a yes or no answer or a selection from two candidates,
    and general questions, which demand a concise response.
    The context is: {context}.
    Find the useful documents from the context, then answer the question: {question}.
    For general questions, you should use the exact words from the context as the answer to avoid ambiguity.
    Answer the question directly and don't output other thing.\
    """
)

ELI5_LFQA = PromptTemplate(
    template="""\
    With the question: {question}

    Can you explain the following context in a way that a 5-year-old can understand?

    -----
    {context}
    -----

    Break down complex concepts using simple words, analogies, and everyday examples.
    Avoid technical jargon.

    Go through the following context and then extract the answer of the question from the context.
    Answer the question directly. Your answer should be concise.
    """
)
# Question: What happens to helium-filled birthday balloons that are released into open air?
# Answer: they go up a few thousand feet and pop as the gas in the balloon expands beyond the stretching limit for the rubber.  No balloon will ever make it into orbit.. orbit requires thousands and thousands of miles an hour of lateral velocity
# Question: Why does the moon have so many craters and the earth so few?
# Answer: Partially right. It's mainly erosion, and many smaller things are lost due to disintegration in the atmosphere. Water, wind, ice, etc- they all work to reform the landscape, erasing evidence of impacts. But we still see plenty of craters in dry places or places with less erosional activity. Meteor crater in Arizona is a good example. Also, impacts in the ocean are much less likely to leave a crater (unless it's really big), and since oceans cover over half the surface area of the earth, that many less craters!
EXTRACT_ANSWER = PromptTemplate(
    template="""\
    As an AI assistant, you have been provided with a question and its long answer.
    Your task is to derive a very concise short answer, extracting a substring from the given long answer.
    Short answer is typically an entity without any other redundant words.
    It's important to ensure that the output short answer remains as simple as possible.

    Here are some examples:
    Question: big little lies season 2 how many episodes
    Long Answer: Production on the second season began in March 2018 and is set to premiere in 2019. All seven episodes are being written by Kelley and directed by Andrea Arnold.
    Short Answer: seven

    Question: where do you cross the arctic circle in norway
    Long Answer: The Arctic circle crosses mainland Norway at Saltfjellet, which separates Helgeland from the northern part of Nordland county.
    Short Answer: Saltfjellet

    Question: how many episodes are in series 7 game of thrones
    Long Answer: the seventh season consisted of only seven episodes.
    Short Answer: seven

    Question: who is the month of may named after
    Long Answer: The month of May (in Latin, "Maius") was named for the Greek Goddess Maia, who was identified with the Roman era goddess of fertility, Bona Dea, whose festival was held in May.
    Short Answer: the Greek Goddess Maia

    Question: who played all the carly's on general hospital
    Long Answer: Sarah Joy Brown (1996–2001), Tamara Braun (2001–05), Jennifer Bransford (2005), and Laura Wright (since 2005) played the character Carly Corinthos on "General Hospital."
    Short Answer: Jennifer Bransford

    Question: who played all the carly's on general hospital
    Long Answer: Sarah Joy Brown (1996–2001), Tamara Braun (2001–05), Jennifer Bransford (2005), and Laura Wright (since 2005) played the character Carly Corinthos on "General Hospital."
    Short Answer: Laura Wright

    Question: who played all the carly's on general hospital
    Long Answer: Sarah Joy Brown (1996–2001), Tamara Braun (2001–05), Jennifer Bransford (2005), and Laura Wright (since 2005) played the character Carly Corinthos on "General Hospital."
    Short Answer: Sarah Joy Brown

    Question: who played all the carly's on general hospital
    Long Answer: Sarah Joy Brown (1996–2001), Tamara Braun (2001–05), Jennifer Bransford (2005), and Laura Wright (since 2005) played the character Carly Corinthos on "General Hospital."
    Short Answer: Tamara Braun

    Question: the first life forms to appear on earth were
    Long Answer: the first life forms to appear on earth were putative fossilized microorganisms found in hydrothermal vent precipitates.
    Short Answer: putative fossilized microorganisms

    Question: who wrote the song for once in my life
    Long Answer: Ron Miller and Orlando Murden
    Short Answer: Ron Miller

    Question: who wrote the song for once in my life
    Long Answer: Ron Miller and Orlando Murden
    Short Answer: Orlando Murden

    Question: where did the rule of 72 come from
    Long Answer: An early reference to the rule is in the "Summa de arithmetica" (Venice, 1494. Fol. 181, n. 44) of Luca Pacioli (1445–1514). He presents the rule in a discussion regarding the estimation of the doubling time of an investment, but does not derive or explain the rule, and it is thus assumed that the rule predates Pacioli by some time.
    Short Answer: Summa de arithmetica

    Question: where did the rule of 72 come from
    Long Answer: An early reference to the rule is in the "Summa de arithmetica" (Venice, 1494. Fol. 181, n. 44) of Luca Pacioli (1445–1514). He presents the rule in a discussion regarding the estimation of the doubling time of an investment, but does not derive or explain the rule, and it is thus assumed that the rule predates Pacioli by some time.
    Short Answer: of Luca Pacioli

    Question: {question}
    Long Answer: {long_answer}
    Short Answer:\
    """
)
