from transformers import pipeline
from ..utils.bert_utils import ensure_tokenizer, load_model, get_pred_sent
from ..utils.model_utils import load_jsonl, save_jsonl, load_json
from ..utils.preprocess_utils import restore_numbers
from tqdm import tqdm
import argparse


# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Load inference parameters from a specified path."
)
parser.add_argument(
    "param_path", type=str, help="Path to the inference parameters JSON file"
)

args = parser.parse_args()

# Use the provided path
param_path = args.param_path
param_dict = load_json(param_path)
# Study parameters
num_tokenization = param_dict["num_tokenization"]
inc = param_dict["inclusion_variable"]
data_labeled = param_dict["data_labeled"]

# paths
model_path = param_dict["model_path"]
tokenizer_path = param_dict["tokenizer_path"]
inference_path = param_dict["inference_path"]
save_path = param_dict["save_path"]

# Define sentence varibale for the trial if [NUM] tokenization is used
sent_var = "sentence_num_tokens" if num_tokenization else "sentence"


# Create the pipeline
model = load_model(model_path)
tokenizer = ensure_tokenizer(tokenizer_path)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
inference_dataset = load_jsonl(inference_path)

result_list = []
for sample in tqdm(inference_dataset):
    if sample[inc]:
        sentence_text = sample[sent_var]
        ner_results = ner_pipeline(sentence_text)
        pred_sent = get_pred_sent(ner_results, sentence_text)

        if num_tokenization:
            orig_numbers = sample["orignal_num_values"]
            final_prediction = restore_numbers(pred_sent, orig_numbers)
        else:
            final_prediction = pred_sent

        res_dict = {"sentence": sample["sentence"], "prediction": final_prediction}

        if data_labeled:
            res_dict["labeled_sentence"] = sample["labeled_sentence"]

        result_list.append(res_dict)

# Save result list to jsonl
save_jsonl(result_list, f"{save_path}/inference_results.jsonl")
