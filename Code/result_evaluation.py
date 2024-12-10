from .utils.model_utils import calculate_metrics, load_jsonl
from .utils.preprocess_utils import extract_entities
import argparse
import os
import pandas as pd

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Load inference results from a specified path.")
parser.add_argument("result_path", type=str, help="Path to the inference result JSONL file")

args = parser.parse_args()

# Use the provided path
result_path = args.result_path

result_list = load_jsonl(result_path)

label_list = []
for result_dict in result_list:
    true_labels, _ = extract_entities(result_dict["labeled_sentence"])
    predictions, _ = extract_entities(result_dict["prediction"])
    new_dict = {
        "labels": true_labels,
        "predictions": predictions
    }
    label_list.append(new_dict)

precision, recall, f1 = calculate_metrics(label_list)
metrics_list = []
value_list = []
for key in precision.keys():
    metrics_list.append(f"{key}_precision")
    value_list.append(precision[key])
    metrics_list.append(f"{key}_recall")
    value_list.append(recall[key])
    metrics_list.append(f"{key}_f1")
    value_list.append(f1[key])

for metric, value in zip(metrics_list, value_list):
    print(f"{metric}: {value}")
# save the metrics to a file
folder_path = os.path.dirname(result_path)
metrics_path = os.path.join(folder_path, "result_metrics.csv")

save_dict = {
    "metrics": metrics_list,
    "values": value_list
}
save_df = pd.DataFrame(save_dict)
save_df.to_csv(metrics_path, index=False)

