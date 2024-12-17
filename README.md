# Aortic Measurements Extraction

This repository contains the code used in our manuscript **"From Unstructured Text to Research-Ready Data: Aortic Measurement Extraction from Radiology Reports Using Instruction-Tuned Large Language Models"**, in which we explored the use of LLMs to extract aortic measurements from Chest CT radiology reports. The project enables reproduction of results and further experimentation.


## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)

## Overview

The folder `Code` containts the python code for the following:
- Preprocessing label-studio formatted labeled radiology report narratives.
- Fine-tuning BERT-based models for aortic measurement extraction.
- Instruction-tuning Llama models for aortic measurement extraction.
- Performing inference for both Llama and BERT-based models.
- Evaluating a model's performance based on the inference output.
- Exploring optuna hyperparameter tuning results.

The folder `Additional resource` includes JOSN parameter files to recreate our analysis.
For BERT this includes:
- `universal_args.json`: Universal arguments used in all BERT-based models in our study.
- `inference_params_sample.json`: Sample parameters for performing inference.
- Model specific parameter files to perform hyper parameter tuning for the six BERT-based models in our study.
For Llama this includes:
- `universal_args.json`: Universal arguments used in all Llama models in our study.
- `inference_params_sample.json`: Sample parameters for performing inference.
- `system_prompt.json`: The system prompt used for our Llama models.
- Model specific parameter files to perform hyper parameter tuning for the three Llama models in our study.

The folder `Data samples` includes:
- `label_studio_samples.json`: Anonymized sample radiology report narratives labeled using label-studio.
- `preprocessed_samples.jsonl`: The same samples, following preprocessing.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yalesurgeryresearch/RadTextExtractor.git
   cd /path/to/CT-Aorta_NLP-LLM
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Code
To run the code, ensure that your current working directory is set to `/path/to/RadTextExtractor` (the root of the project):
```bash
cd /path/to/RadTextExtractor
```

#### Why This Is Important
The project is organized as a Python module, with code residing in subdirectories such as `Code`. Running the code with the `-m` flag (e.g., `python -m Code.result_evaluation`) requires Python to locate the `Code` module relative to the project root. If you are not in the root directory, Python cannot properly resolve the imports, and you may encounter a `ModuleNotFoundError` or similar errors.

### Preprocessing
Run the following command to perform preprocessing on radiology report narratives:
```bash
python -m Code.preprocess /path/to/radiology_report_narratives.jsonl
```

### BERT fine-tuning
Create a BERT model parameter JSON file like the examples provided, including hyperparameter search space.
Run the following command to fine-tune the model:
```bash
python -m Code.BERT.bert_optuna_inference /path/to/bert_model_parameters.json
```

### BERT inference
Create an inference parameter JSON file, designating a model, tokenizer and inference dataset. 
Run the following command to perform the inference:
```bash
python -m Code.BERT.bert_inference /path/to/bert_inference_parameters.json
```

### Llama instruction-tuning
Create a BERT model parameter JSON file like the examples provided, including hyperparameter search space.
Run the following command to instruction-tune the model:
```bash
python -m Code.Llama.llama_optuna_inference /path/to/llama_model_parameters.json
```

### Llama inference
Create an inference parameter JSON file, designating a model and inference dataset. Note that the use of few-shot examples can be designated through the inference JSON.
Run the following command to perform the inference:
```bash
python -m Code.Llama.llama_inference /path/to/llama_inference_parameters.json
```

### Evaluating results
BERT and Llama inferences generate results in the same format and so the performance evaluation is shared.
Run the following command to evaluate the model performance based on the inference output:
```bash
python -m Code.result_evaluation /path/to/inference_results.jsonl
```

### Exploring hyperparameter tuning results
`study_explorer.py` can be used to select the optimal model based on hyperparameter tuning results. Note that Llama based models are ranked based on validation loss rather than macro-F1 since `SFTTrainer` does not support custom evaluation functions. To use `study_explorer.py` add the study path and run it. 

## Citation

If you use this code in your research, please cite our paper \[placeholder\]:

```perl
@article{YourPaperCitation,
  author    = {Your Name and Co-Authors},
  title     = {Title of Your Paper},
  journal   = {Journal Name},
  volume    = {XX},
  number    = {X},
  pages     = {XX-XX},
  year      = {YYYY},
  publisher = {Publisher Name},
  doi       = {DOI link},
  url       = {URL link (if applicable)}
}
```
