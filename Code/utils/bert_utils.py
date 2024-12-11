from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig
import os
from datasets import Dataset
from ..utils.preprocess_utils import LABEL_DICT
from ..utils.model_utils import load_jsonl

ENTITY_TO_LABEL_DICT = {
    "O": 0,
    "ANN": 1,
    "SOV": 2,
    "STJ": 3,
    "ASC": 4,
    "PTB": 5,
    "ARC": 6,
    "PDC": 7,
    "DSC": 8,
}


# Function to ensure the tokenizer is present, add the [NUM] token if not present, and save the tokenizer
def ensure_tokenizer(folder_path, model_name="", tokenizer_func=AutoTokenizer):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Load the tokenizer from the folder
        tokenizer = tokenizer_func.from_pretrained(folder_path)
        print("Loaded tokenizer from existing folder.")
    else:
        # Create the folder
        os.makedirs(folder_path)
        # Load the regular BERT tokenizer
        tokenizer = tokenizer_func.from_pretrained(model_name)
        # Add [NUM] to the tokenizer if not already present
        if "[NUM]" not in tokenizer.get_vocab():
            tokenizer.add_tokens(["[NUM]"])
        # Save the modified tokenizer to the folder
        tokenizer.save_pretrained(folder_path)
        print("Created folder, added token, and saved tokenizer.")
    return tokenizer


# Function to load an AutoModelForTokenClassification model from the specified path or identifier
def load_model(model_path_or_name):
    custom_labels = {
        0: "O",
        1: "ANN",
        2: "SOV",
        3: "STJ",
        4: "ASC",
        5: "PTB",
        6: "ARC",
        7: "PDC",
        8: "DSC",
    }
    num_labels = len(custom_labels)

    config = AutoConfig.from_pretrained(
        model_path_or_name,
        id2label={label: i for i, label in ENTITY_TO_LABEL_DICT.items()},
        label2id=ENTITY_TO_LABEL_DICT,
    )
    config.num_labels = num_labels  # Ensure the config knows about the number of labels

    # Load the model from the specified path or identifier
    model = AutoModelForTokenClassification.from_pretrained(
        model_path_or_name, config=config
    )
    for param in model.parameters():
        param.data = param.data.contiguous()
    return model


# Function to tokenize and format the data for BERT finetuning
def tokenize_and_format_data(
    file_path,
    tokenizer,
    dim="dimensions",
    sent="sentence",
    inc="inclusion",
    max_length=384,
    labeled=False,
):

    def get_entities(dimensions):
        entities = []
        for dimension in dimensions:
            start = dimension["start"]
            end = dimension["end"]
            entity_type = LABEL_DICT[dimension["labels"][0]]
            entities.append((start, end, entity_type))
        return entities

    def create_and_adjust_labels(entities, offset_mapping, attention_mask):
        labels = []
        for idx, (offset_start, offset_end) in enumerate(offset_mapping):
            label = "O"  # Default label
            for start, end, entity_type in entities:
                if offset_start == start or (
                    offset_start > start and offset_end <= end
                ):
                    label = entity_type
                    break
            if attention_mask[idx] == 1:
                labels.append(ENTITY_TO_LABEL_DICT[label])
            else:
                labels.append(-100)
        return labels

    dataset_dict = {
        "input_ids": [],
        "text": [],
        "attention_mask": [],
        "offset_mapping": [],
    }

    if labeled:
        dataset_dict["labels"] = []

    json_data = load_jsonl(file_path)

    for sample in json_data:
        if sample[inc]:
            text = sample[sent]
            encoding = tokenizer(
                text,
                return_offsets_mapping=True,
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            dataset_dict["input_ids"].append(encoding["input_ids"])
            dataset_dict["text"].append(text)
            dataset_dict["attention_mask"].append(encoding["attention_mask"])
            dataset_dict["offset_mapping"].append(encoding["offset_mapping"])

            if labeled:
                entities = get_entities(sample[dim])
                labels = create_and_adjust_labels(
                    entities, encoding["offset_mapping"], encoding["attention_mask"]
                )
                dataset_dict["labels"].append(labels)

    return Dataset.from_dict(dataset_dict)


def get_pred_sent(ner_results, sentence_text):
    # Step 1: Sort the NER results by the 'start' index
    sorted_ner_results = sorted(ner_results, key=lambda x: x["start"])

    # Step 2: Merge adjacent or overlapping entities with the same type
    merged_results = []
    for result in sorted_ner_results:
        if (
            merged_results
            and merged_results[-1]["entity"] == result["entity"]
            and merged_results[-1]["end"] >= result["start"]
        ):
            # Merge the current result with the previous one
            merged_results[-1]["end"] = max(merged_results[-1]["end"], result["end"])
        else:
            # Add a new entity result to the list
            merged_results.append(result)

    # Step 3: Insert entity tags in reverse order to avoid index issues
    for result in reversed(merged_results):
        sentence_text = (
            sentence_text[: result["start"]]
            + f"<{result['entity']}>{sentence_text[result['start']:result['end']]}</{result['entity']}>"
            + sentence_text[result["end"] :]
        )

    return sentence_text
