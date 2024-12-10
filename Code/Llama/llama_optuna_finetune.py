import torch
from transformers import  TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from unsloth import FastLanguageModel
from ..utils.llama_utils import  finetuning_dataset_from_jsonl
from ..utils.model_utils import set_seed, load_json, save_json
import os
import optuna
import gc
import argparse

# Define the model and response template dictionaries
model_dict = {
    "2":"meta-llama/Llama-2-7b-chat-hf",
    "3":"meta-llama/Meta-Llama-3-8B-Instruct",
    "3.1":"meta-llama/Meta-Llama-3.1-8B-Instruct"
}
response_template_with_context_dict = {
    "2":"[/INST]",
    "3": "<|start_header_id|>assistant<|end_header_id|>",
    "3.1": "<|start_header_id|>assistant<|end_header_id|>"
}

# load trial parameters
# Parse command-line arguments
parser = argparse.ArgumentParser(description="Load trial parameters from a specified path.")
parser.add_argument("trial_param_path", type=str, help="Path to the trial parameters JSON file")

args = parser.parse_args()

# Use the provided path to load the trial parameters
trial_param_path = args.trial_param_path
print(f"Loading trial parameters from: {trial_param_path}")
trial_param_dict = load_json(trial_param_path)

# Trial parameters
study_name = trial_param_dict["study_name"]
model_num = trial_param_dict["model_num"]
model_id = model_dict[model_num]
inc = trial_param_dict["inclusion_variable"]
# Paths
base_path = trial_param_dict["base_path"]
universal_args_path = trial_param_dict["universal_args_path"]
train_path = trial_param_dict["train_path"]
validation_path = trial_param_dict["validation_path"]
prompt_path = trial_param_dict["prompt_path"]
prompt_version = trial_param_dict["prompt_version"]
max_seq_length = trial_param_dict["max_seq_length"]

# Define path to save model
model_save_path = f'{base_path}/{study_name}/models'
# Ensure the savedirectory exists
os.makedirs(model_save_path, exist_ok=True)



# Set the response template with context for the DataCollatorForCompletionOnlyLM
response_template_with_context = response_template_with_context_dict[model_num]

# Load the universal training arguments
training_args_dict = load_json(universal_args_path)

# Load the study's hyperparameter search dictionary
search_space_dict = trial_param_dict["search_space_dict"]

def objective(trial):
    # Set seed
    seed = 42
    set_seed(seed)
    
    # Set trial id and save path
    trial_id = trial.number
    trial_path = f'{model_save_path}/trial_{trial_id}'
    os.makedirs(trial_path, exist_ok=True)
    
    print(f"Trial {trial_id}")
    print(f"model: {model_id}")
    
    # Select hyperparameters
    learning_rate = trial.suggest_categorical("learning_rate", search_space_dict["learning_rate"])
    rank= trial.suggest_categorical("rank", search_space_dict["rank"])
    alpha = trial.suggest_categorical("alpha", search_space_dict["alpha"])
    seed_shuffle = trial.suggest_categorical("seed", search_space_dict["seed"]) if "seed" in search_space_dict else None
    num_samples = trial.suggest_categorical("num_samples", search_space_dict["num_samples"]) if "num_samples" in search_space_dict else None
    
    # Update training arguments with trial-specific hyperparameters
    dict_update = {
        "learning_rate": learning_rate,
        "output_dir": trial_path,
        "logging_dir": trial_path+'/logs',
    }
    
    training_args_dict.update(dict_update)
    
    # Save the training arguments for this trial
    save_json(training_args_dict, f'{trial_path}/training_args.json')

    # Convert the training arguments to a TrainingArguments object
    training_args = TrainingArguments(**training_args_dict)
    
    
    # Load the model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_id,
        max_seq_length = max_seq_length,
        dtype = torch.float16,
        load_in_4bit = True,
    )

    
    # Prepare the model for training
    model = FastLanguageModel.get_peft_model(
        model,
        r = rank, 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = int(rank*alpha),
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  
        loftq_config = None, 
    )
    
    # Set the padding token to the eos token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # Define the system prompts
    prompt_dict = load_json(prompt_path)
    messages = [
        {"role": "system", "content": prompt_dict[prompt_version]+"\n"}
    ]

    # Load the training and test datasets
    train_dataset = finetuning_dataset_from_jsonl(tokenizer, train_path, inc, messages)
    validation_dataset = finetuning_dataset_from_jsonl(tokenizer, validation_path, inc, messages)
    
    # For ablation study with fewer training samples
    if seed_shuffle and num_samples:
        train_dataset = train_dataset.shuffle(seed_shuffle).select(range(num_samples))

    # Create a data collator for the completion-only language model
    # response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[1:]
    collator =DataCollatorForCompletionOnlyLM(response_template_with_context, tokenizer=tokenizer)
    
    # Create a trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        max_seq_length=max_seq_length,
        dataset_text_field="input",
        tokenizer=tokenizer,
        args=training_args,
        packing= False,
        data_collator=collator,
    )

    # Train the model
    trainer.train()
    
    # Evaluate the model
    eval_result = trainer.evaluate()
    
    # Log evaluation metrics
    trial.set_user_attr("eval_result", eval_result)
    print(eval_result)
    
    # Clean up
    del model
    del trainer
    torch.cuda.empty_cache()
    gc.collect()
    
    return eval_result["eval_loss"]

# Create a grid sampler for the hyperparameters
sampler = optuna.samplers.GridSampler(search_space_dict)

# Define the absolute path for the database file
db_path = os.path.join(base_path, f'{study_name}.db')
storage = f'sqlite:///{db_path}'

# Create an Optuna study with a specific name
study = optuna.create_study(study_name=study_name, direction="minimize", sampler=sampler, storage=storage, load_if_exists=True)

# Calculate the number of trials
n_trials = 1
for key in search_space_dict:
    n_trials *= len(search_space_dict[key])
    
# Optimize the objective function
study.optimize(objective, n_trials=n_trials)

# Print the best hyperparameters
best_trial = study.best_trial
print(f"Best trial hyperparameters: {best_trial.params}")
