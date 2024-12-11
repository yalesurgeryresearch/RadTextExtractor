import random
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import json
from collections import defaultdict


def set_seed(seed=42):
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_jsonl(jsonl_path):
    with open(jsonl_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data


def save_jsonl(data, jsonl_path):
    with open(jsonl_path, "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")


def load_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def save_json(data, json_path):
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)


def compute_metrics_v2(p):
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

    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = []
    true_predictions = []

    for prediction, label in zip(predictions, labels):
        # Collect true labels and predictions, excluding the -100 mask token
        true_labels.extend(
            [int(lab) for lab in label if lab != -100]
        )  # Convert labels to int
        true_predictions.extend(
            [int(p) for p, l in zip(prediction, label) if l != -100]
        )  # Convert predictions to int

    # Get precision, recall, f1 score per label
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, true_predictions, labels=np.unique(true_labels), zero_division=0
    )

    # Map results to their corresponding labels
    unique_labels = [
        custom_labels[int(label)] for label in np.unique(true_labels)
    ]  # Convert labels to Python int
    label_metrics = {
        label: {  # Ensure the label is a Python int, not NumPy type
            "precision": float(precision[i]),  # Cast to standard float
            "recall": float(recall[i]),  # Cast to standard float
            "f1": float(f1[i]),  # Cast to standard float
        }
        for i, label in enumerate(unique_labels)
    }

    # Exclude the null label (assumed to be 0) when calculating the macro average
    valid_labels = [label for label in unique_labels if label != "O"]
    # Calculate macro-average for precision, recall, and f1, casting the result to Python float
    macro_precision = float(
        np.mean([label_metrics[label]["precision"] for label in valid_labels])
    )
    macro_recall = float(
        np.mean([label_metrics[label]["recall"] for label in valid_labels])
    )
    macro_f1 = float(np.mean([label_metrics[label]["f1"] for label in valid_labels]))

    return {
        "label_metrics": label_metrics,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
    }


def calculate_tp_fp_fn(true_labels, pred_labels):

    # Initialize counters
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    # Convert the lists of dicts to sets of (text, label) tuples
    true_set = set((item["text"], item["labels"][0]) for item in true_labels)
    pred_set = set((item["text"], item["labels"][0]) for item in pred_labels)

    # Calculate true positives
    for text, label in true_set:
        if (text, label) in pred_set:
            tp[label] += 1

    # Calculate false positives
    for text, label in pred_set:
        if (text, label) not in true_set:
            fp[label] += 1

    # Calculate false negatives
    for text, label in true_set:
        if (text, label) not in pred_set:
            fn[label] += 1

    return tp, fp, fn


def calculate_metrics(narratives_data_list):
    # Initialize counters
    overall_tp = defaultdict(int)
    overall_fp = defaultdict(int)
    overall_fn = defaultdict(int)

    # Initialize overall sums for micro calculation
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Calculate TP, FP, FN for each narrative and sum them up
    for narrative in narratives_data_list:
        true_labels = narrative["labels"]
        pred_labels = narrative["predictions"]
        tp, fp, fn = calculate_tp_fp_fn(true_labels, pred_labels)

        # Sum individual class-level TP, FP, FN
        for key in tp:
            overall_tp[key] += tp[key]
            total_tp += tp[key]
        for key in fp:
            overall_fp[key] += fp[key]
            total_fp += fp[key]
        for key in fn:
            overall_fn[key] += fn[key]
            total_fn += fn[key]

    # Initialize dictionaries to store precision, recall, and F1 score for each label
    precision = {}
    recall = {}
    f1 = {}

    # Calculate precision, recall, and F1 score for each label
    for label in (
        set(overall_tp.keys()).union(overall_fp.keys()).union(overall_fn.keys())
    ):
        precision[label] = (
            overall_tp[label] / (overall_tp[label] + overall_fp[label])
            if (overall_tp[label] + overall_fp[label]) > 0
            else 0
        )
        recall[label] = (
            overall_tp[label] / (overall_tp[label] + overall_fn[label])
            if (overall_tp[label] + overall_fn[label]) > 0
            else 0
        )
        f1[label] = (
            2 * precision[label] * recall[label] / (precision[label] + recall[label])
            if (precision[label] + recall[label]) > 0
            else 0
        )

    # Calculate macro-averaged F1 score
    macro_f1 = sum(f1.values()) / len(f1) if f1 else 0
    macro_recall = sum(recall.values()) / len(recall) if recall else 0
    macro_precision = sum(precision.values()) / len(precision) if precision else 0
    precision["macro"] = macro_precision
    recall["macro"] = macro_recall
    f1["macro"] = macro_f1

    # Calculate micro precision, recall, and F1 score
    micro_precision = (
        total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    )
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0
    )

    # Add micro-averaged values to the output dictionaries
    precision["micro"] = micro_precision
    recall["micro"] = micro_recall
    f1["micro"] = micro_f1

    return precision, recall, f1
