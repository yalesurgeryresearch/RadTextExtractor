import optuna
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification
from ..utils.bert_utils import ensure_tokenizer, load_model, tokenize_and_format_data
from ..utils.model_utils import set_seed, compute_metrics_v2, load_json, save_json
import os
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Load trial parameters from a specified path."
)
parser.add_argument(
    "trial_param_path", type=str, help="Path to the trial parameters JSON file"
)

args = parser.parse_args()

# Use the provided path to load the trial parameters
trial_param_path = args.trial_param_path
print(f"Loading trial parameters from: {trial_param_path}")
trial_param_dict = load_json(trial_param_path)

# Trial parameters
study_name = trial_param_dict["study_name"]
model_name = trial_param_dict["base_model"]
num_tokenization = trial_param_dict["num_tokenization"]
inc = trial_param_dict["inclusion_variable"]

# Paths
base_path = trial_param_dict["base_path"]
universal_args_path = trial_param_dict["universal_args_path"]
train_path = trial_param_dict["train_path"]
validation_path = trial_param_dict["validation_path"]

# Define the paths for the tokenizer and the model
model_save_path = f"{base_path}/{study_name}/models"
num_tokenizer_path = f"{base_path}/{study_name}/num_tokenizer"

# Ensure the directory exists
os.makedirs(model_save_path, exist_ok=True)

# Model parameters
models = {
    "bertbase": "bert-base-uncased",
    "pubmedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "bertbase-NER": "dslim/bert-base-NER",
}
base_model = models[model_name]

# Load unviversal training arguments from JSON file
training_args_dict = load_json(universal_args_path)

# Set up tokenizer
# tokenizer = ensure_tokenizer(num_tokenizer_path, base_model)

# Load the study's hyperparameter search dictionary
search_space_dict = trial_param_dict["search_space_dict"]


# Optuna objective function
def objective(trial):
    # Set seed
    seed = 42
    set_seed(seed)

    # Define trial id and save path
    trial_id = trial.number
    trial_path = f"{model_save_path}/trial_{trial_id}"
    os.makedirs(trial_path, exist_ok=True)
    print(f"Trial {trial_id}")

    # Load model and tokenizer
    print("Loading model")
    model = load_model(base_model)

    print("Loading tokenizer")
    tokenizer = ensure_tokenizer(num_tokenizer_path, base_model)

    # resize token embeddings in the model to account for additional [NUM] token in the tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # Define suffix for the trial if [NUM] tokenization is used
    if num_tokenization:
        dim = "dimensions_num_tokens"
        sent = "sentence_num_tokens"

    # Prepare train and validation dataset
    train_dataset = tokenize_and_format_data(
        train_path, tokenizer, dim=dim, sent=sent, inc=inc, labeled=True
    )
    validation_dataset = tokenize_and_format_data(
        validation_path, tokenizer, dim=dim, sent=sent, inc=inc, labeled=True
    )

    # Set trial hyperparameters
    learning_rate = trial.suggest_categorical(
        "learning-Rate", search_space_dict["learning-Rate"]
    )
    batch_size = trial.suggest_categorical(
        "batch-size", search_space_dict["batch-size"]
    )
    dict_update = {
        "learning_rate": learning_rate,
        "per_device_train_batch_size": batch_size,
        "output_dir": trial_path,
        "logging_dir": trial_path + "/logs",
    }
    print("Trial hyperparameters:")
    print(dict_update)

    # Update training arguments with trial-specific hyperparameters
    training_args_dict.update(dict_update)

    # Save the training arguments for this trial
    save_json(training_args_dict, f"{trial_path}/training_args.json")

    # Convert the training arguments to a TrainingArguments object
    training_args = TrainingArguments(**training_args_dict)

    # Data collator for token classification
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_v2,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_result = trainer.evaluate()

    # Log evaluation metrics
    trial.set_user_attr("eval_result", eval_result)

    return eval_result["eval_f1"]


# Define the search space
sampler = optuna.samplers.GridSampler(search_space_dict)

# Define the path for the database file
db_path = os.path.join(base_path, f"{study_name}.db")
storage = f"sqlite:///{db_path}"

# Create an Optuna study with a specific name
study = optuna.create_study(
    study_name=study_name,
    direction="maximize",
    sampler=sampler,
    storage=storage,
    load_if_exists=True,
)

# Calculate the number of trials
n_trials = 1
for key in search_space_dict:
    n_trials *= len(search_space_dict[key])

# Optimize the objective function
study.optimize(objective, n_trials=n_trials)

# print the best hyperparameters
best_trial = study.best_trial
print(f"Best trial hyperparameters: {best_trial.params}")
