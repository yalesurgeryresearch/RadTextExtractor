from .utils.model_utils import load_json, save_jsonl
from .utils.preprocess_utils import (
    get_nlp,
    get_sent_tokenizer,
    split_text_nltk_v2,
    replace_numbers,
    extract_entities,
)
import argparse
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Load labeled records from a specified path."
)
parser.add_argument("data_path", type=str, help="Path to the labeled records JSON file")

args = parser.parse_args()

# Use the provided path
data_path = args.data_path
save_path = os.path.join(os.path.dirname(data_path), "preprocessed_samples.jsonl")
data = load_json(data_path)
print(f" Number of records: {len(data)}")

# Create a tokenizer and nlp object
sent_tokenizer = get_sent_tokenizer()
nlp = get_nlp()

# split the text into sentences and integrate labels
full_sent_list = []
for sample in data:
    sent_list = split_text_nltk_v2(sample, sent_tokenizer, nlp, id_variable="id")
    full_sent_list.extend(sent_list)


# Add [NUM] token variants for BERT
def add_num_tokens(sample):
    sample["labeled_sentence_num_tokens"], sample["orignal_num_values"] = (
        replace_numbers(sample["labeled_sentence"])
    )
    sample["dimensions_num_tokens"], sample["sentence_num_tokens"] = extract_entities(
        sample["labeled_sentence_num_tokens"]
    )
    return sample


def preprocess_data(data):
    for sample in data:
        sample = add_num_tokens(sample)
    return data


full_list_preprocessed = preprocess_data(full_sent_list)

save_jsonl(full_list_preprocessed, save_path)
