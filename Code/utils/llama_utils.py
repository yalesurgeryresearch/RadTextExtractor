import json
from datasets import Dataset


def remove_last_occurrence(input_string, target):
    # Find the index of the last occurrence of the target word
    index = input_string.rfind(target)

    # If the target word is found, remove it
    if index != -1:
        input_string = input_string[:index] + input_string[index + len(target) :]

    return input_string


def finetuning_dataset_from_jsonl(tokenizer, jsonl_path, inc, prompt_messages):
    with open(jsonl_path, "r") as f:
        data = []
        for line in f:
            sample = json.loads(line)
            if sample[inc]:
                curr_input = (
                    prompt_messages
                    + [{"role": "user", "content": sample["sentence"] + "\n"}]
                    + [
                        {
                            "role": "assistant",
                            "content": sample["labeled_sentence"] + "\n",
                        }
                    ]
                )
                data.append(
                    {
                        "input": tokenizer.apply_chat_template(
                            curr_input, tokenize=False
                        ),
                        "label": sample["labeled_sentence"],
                    }
                )
    return Dataset.from_list(data)
