import re
import spacy
import nltk

SENT_KEYWORDS = [
    "aorta",
    "aortic",
    "ascending",
    "descending",
    "sov",
    "root",
    "aneurysm",
    " arch",
    "valsalva",
    "sinotubular",
]
LABEL_DICT = {
    "Mid Ascending": "ASC",
    "Annulus": "ANN",
    "Sinus of Valsalva": "SOV",
    "Sinotubular junction": "STJ",
    "Ascending proximal to the brachiocephalic": "PTB",
    "Top of Arch": "ARC",
    "Proximal Descending": "PDC",
    "Mid Descending": "DSC",
}
REVERSE_LABEL_DICT = {v: k for k, v in LABEL_DICT.items()}


def get_nlp():
    try:
        # Attempt to load the model
        return spacy.load("en_core_web_sm")
    except OSError:
        # If not installed, download the model first
        spacy.cli.download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


def get_sent_tokenizer():
    return nltk.data.load("tokenizers/punkt/english.pickle")


# Function to check if a token is a date or time
def is_date_or_time_spacy(token):
    return token.ent_type_ in {"DATE", "TIME"}


def check_numbers_sentence(sentence, nlp):
    if not bool(re.search(r"\d", sentence)):
        return False
    doc = nlp(sentence)
    # Iterate over tokens and check the conditions
    for token in doc:
        if (
            re.match(r"\d", token.text)
            and not is_date_or_time_spacy(token)
            and "/" not in token.text
        ):
            return True
    return False


def check_keywords(sentence):
    return any(keyword in sentence.lower() for keyword in SENT_KEYWORDS)


# Function to check if a sentence contains numbers and keywords
def check_sentence(sentence, nlp):
    return check_keywords(sentence) and check_numbers_sentence(sentence, nlp)


# Function to check if a sentence contains numbers and keywords as well as node and first sentence exclusion
def check_sample(sample, nlp):
    sentence = sample["sentence"]
    return (
        sample["start_char"] > 0
        and "node" not in sentence
        and check_sentence(sentence, nlp)
    )


# Function to replace numeric values with a placeholder
def replace_numbers(sentence):
    if isinstance(sentence, float):
        return sentence, []
    # Regular expression to match integers and floats, including those attached to letters
    pattern = re.compile(r"(\d+\.\d+|\d+)")

    # Find all matches in the sentence
    numbers = pattern.findall(sentence)

    # Replace the matches in the sentence with [NUM] using a function
    def replace_match(match):
        return "[NUM]"

    modified_sentence = pattern.sub(replace_match, sentence)

    return modified_sentence, numbers


# return original_sentence
def restore_numbers(modified_sentence, numbers):
    # Regular expression to match the [NUM] placeholder
    pattern = re.compile(r"\[NUM\]")

    # Use an index to track the position in the numbers list
    index = 0

    # Replace each [NUM] with the corresponding number from the list
    def replace_match(match):
        nonlocal index
        replacement = numbers[index]
        index += 1
        return replacement

    # Use the sub method to replace the [NUM] placeholders with the original numbers
    original_sentence = pattern.sub(replace_match, modified_sentence)

    return original_sentence


def sort_entities(entities):
    return sorted(entities, key=lambda x: x["start"])


# Function to extract entities from a labeled sentence
def extract_entities(sentence):
    tag_pattern = re.compile(r"<([^>]+)>(.*?)</\1>")
    entities = []
    offset = 0

    for match in tag_pattern.finditer(sentence):
        tag = match.group(1)
        text = match.group(2)
        start = match.start(2) - offset
        end = match.end(2) - offset - 1
        entity = next((key for key, value in LABEL_DICT.items() if value == tag), None)
        if entity:
            entities.append(
                {"start": start - 5, "end": end - 4, "text": text, "labels": [entity]}
            )

        # Update offset to account for the length of the removed tags
        offset += len(match.group(0)) - len(text)

    cleaned_sentence = re.sub(tag_pattern, r"\2", sentence)
    entities = sort_entities(entities)
    return entities, cleaned_sentence


def split_text_nltk_v2(data_dic, sent_tokenizer, nlp, id_variable):
    narrative = data_dic["Narrative"]
    id = data_dic[id_variable]
    # Tokenize sentences and get their spans
    # check if narrative is a float
    if isinstance(narrative, float):
        return [
            {
                "id": id,
                "sentence": narrative,
                "start_char": 0,
                "end_char": 0,
                "dimensions": [],
                "labeled_sentence": narrative,
                "inclusion": False,
            }
        ]
    sentences = sent_tokenizer.tokenize(narrative)
    spans = list(sent_tokenizer.span_tokenize(narrative))
    sent_list = []
    for (sent_start, sent_end), sentence in zip(spans, sentences):
        sent_dic = {
            "id": id,
            "sentence": sentence,
            "start_char": sent_start,
            "end_char": sent_end,
            "dimensions": [],
            "labeled_sentence": sentence,
        }
        if "label" in data_dic:
            dimensions = data_dic["label"]
            for dim in dimensions:
                if dim["start"] >= sent_start and dim["end"] <= sent_end:
                    new_dim = dim.copy()
                    new_dim["start"] -= sent_start
                    new_dim["end"] -= sent_start
                    new_dim["labels"] = [REVERSE_LABEL_DICT[dim["labels"][0]]]
                    sent_dic["dimensions"].append(new_dim)
                # sort the dimensions by start
            sent_dic["dimensions"] = sorted(
                sent_dic["dimensions"], key=lambda x: x["start"]
            )
            if len(sent_dic["dimensions"]) > 0:
                sent_dic["labeled_sentence"] = sent_dic["sentence"]
                # insert the dimensions into the labeled sentence in reverse order
                for dim in reversed(sent_dic["dimensions"]):
                    sent_dic["labeled_sentence"] = (
                        sent_dic["labeled_sentence"][: dim["start"]]
                        + f"<{LABEL_DICT[dim['labels'][0]]}>"
                        + sent_dic["labeled_sentence"][dim["start"] : dim["end"]]
                        + f"</{LABEL_DICT[dim['labels'][0]]}>"
                        + sent_dic["labeled_sentence"][dim["end"] :]
                    )
        sent_dic["inclusion"] = check_sample(sent_dic, nlp)
        sent_list.append(sent_dic)
    return sent_list
