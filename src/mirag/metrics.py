import unicodedata
import string
import regex


puncs = list(string.punctuation)


class SimpleTokenizer(object):
    ALPHA_NUM = r"[\p{L}\p{N}\p{M}]+"
    NON_WS = r"[^\p{Z}\p{C}]"

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            "(%s)|(%s)" % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE,
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens


def _normalize(text):
    """
    Normalize Unicode strings. Necessary for text which contains non-ASCII characters.
    """
    return unicodedata.normalize("NFD", text)


def remove_articles(text):
    return regex.sub(r"\b(a|an|the)\b", " ", text)


def white_space_fix(text):
    return " ".join(text.split())


def remove_punc(text):
    exclude = set(puncs)
    return "".join(ch for ch in text if ch not in exclude)


def lower(text):
    return text.lower()


# def normalize_answer(text: str) -> str:
#     """
#     Normalize answer text for more robust matching.

#     This function:
#     1. Converts to lowercase
#     2. Removes articles (a, an, the)
#     3. Removes punctuation
#     4. Removes excessive whitespace
#     5. Strips common prefixes like "the answer is"

#     Args:
#         text: The text to normalize

#     Returns:
#         str: Normalized text
#     """
#     import re
#     import string

#     text = _normalize(text)

#     # Convert to lowercase
#     text = text.lower()

#     # Remove common prefixes that don't affect the answer's meaning
#     prefixes = ["the answer is ", "answer: ", "answer is ", "it is ", "it's ", "this is "]
#     for prefix in prefixes:
#         if text.startswith(prefix):
#             text = text[len(prefix):]

#     # Remove articles
#     text = re.sub(r'\b(a|an|the)\b', ' ', text)

#     # Remove punctuation
#     text = text.translate(str.maketrans('', '', string.punctuation))

#     # Remove excessive whitespace and trim
#     text = re.sub(r'\s+', ' ', text).strip()

#     return text

def normalize_answer(s):
    return white_space_fix(remove_articles(remove_punc(lower(_normalize(s)))))


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def single_ans_em(pred, gold):
    # pred: prediction string
    # gold: a list of gold answer strings
    if type(gold) is not list:
        gold = [gold]
    return max(compute_exact(pred, a) for a in gold)

# def single_ans_em(predicted_answer: str, ground_truth_answers) -> float:
#     """
#     Calculate exact match score for natural question answering.

#     This metric checks if the predicted answer exactly matches any of the ground truth answers
#     after normalization (removing articles, punctuation, etc.)

#     Args:
#         predicted_answer: The model's predicted answer as a string
#         ground_truth_answers: List of acceptable ground truth answers

#     Returns:
#         float: 1.0 if there's an exact match after normalization, 0.0 otherwise
#     """
#     if type(ground_truth_answers) is not list:
#         ground_truth_answers = [ground_truth_answers]
#     if not predicted_answer or not ground_truth_answers:
#         return 0.0

#     # Normalize the predicted answer
#     norm_pred = normalize_answer(predicted_answer)

#     # Check if normalized prediction matches any normalized ground truth
#     for gt_answer in ground_truth_answers:
#         if normalize_answer(gt_answer) == norm_pred:
#             return 1.0

#     return 0.0


def has_correct_answer(retrieve_doc, answers):
    tokenizer = SimpleTokenizer()
    # retrieve_doc = _normalize(retrieve_doc)
    retrieve_doc = normalize_answer(retrieve_doc)
    retrieve_doc = tokenizer.tokenize(retrieve_doc, uncased=True)

    for single_answer in answers:
        # single_answer = _normalize(single_answer)
        single_answer = normalize_answer(single_answer)
        single_answer = tokenizer.tokenize(single_answer, uncased=True)

        for i in range(0, len(retrieve_doc) - len(single_answer) + 1):
            if single_answer == retrieve_doc[i : i + len(single_answer)]:
                return 1
    return 0
