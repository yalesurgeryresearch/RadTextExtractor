import torch
from tqdm import tqdm
from ..utils.model_utils import set_seed, load_json, load_jsonl, save_jsonl
from unsloth import FastLanguageModel
import argparse

# set random seed
set_seed(42)

# Set deterministic CUDNN for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
max_seq_length = param_dict["max_seq_length"]
inc = param_dict["inclusion_variable"]
data_labeled = param_dict["data_labeled"]
few_shot = param_dict["few_shot"]

# Paths
model_path = param_dict["model_path"]
inference_path = param_dict["inference_path"]
prompt_path = param_dict["prompt_path"]
prompt_version = param_dict["prompt_version"]
save_path = param_dict["save_path"]

# Load inference data
inference_dataset = load_jsonl(inference_path)

# Define the system prompts
prompt_dict = load_json(prompt_path)
messages = [{"role": "system", "content": prompt_dict[prompt_version] + "\n"}]
# Add few-shot samples if selected
if few_shot:
    few_shot_path = param_dict["few_shot_path"]
    few_shot_samples = load_jsonl(few_shot_path)
    for sample in few_shot_samples:
        messages.append({"role": "user", "content": sample["sentence"] + "\n"})
        messages.append(
            {"role": "assistant", "content": sample["labeled_sentence"] + "\n"}
        )

print(f"Model path: {model_path}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=max_seq_length,
    dtype=torch.float16,
    load_in_4bit=True,
)

# Set the model to evaluation mode, ensure deterministic generation
model.eval()
model.generation_config.temperature = None
model.generation_config.top_p = None
model.generation_config.do_sample = False
FastLanguageModel.for_inference(model)
tokenizer.pad_token_id = tokenizer.eos_token_id

# Prepare all inputs first to avoid repeated tokenization in the loop
formatted_inputs = []
for sample in inference_dataset:
    curr_input = messages + [{"role": "user", "content": sample["sentence"] + "\n"}]
    formatted_input = tokenizer(
        tokenizer.apply_chat_template(curr_input, tokenize=False), return_tensors="pt"
    ).to(device)
    formatted_input["labeled_sentence"] = sample["labeled_sentence"]
    formatted_inputs.append(formatted_input)


result_list = []
for sample in tqdm(inference_dataset):
    if sample[inc]:
        curr_input = messages + [{"role": "user", "content": sample["sentence"] + "\n"}]
        formatted_input = tokenizer(
            tokenizer.apply_chat_template(curr_input, tokenize=False),
            return_tensors="pt",
        ).to(device)

        # Generate the output using the model
        output_ids = model.generate(
            **formatted_input, max_new_tokens=512, use_cache=True
        )
        # Extract the generated tokens (excluding the input prompt)
        generated_ids = output_ids[0, formatted_input["input_ids"].shape[-1] :]
        # Decode the generated output
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        res_dict = {"sentence": sample["sentence"], "prediction": generated_text}
        if data_labeled:
            res_dict["labeled_sentence"] = sample["labeled_sentence"]
        result_list.append(res_dict)

# Save result list to jsonl
save_jsonl(result_list, f"{save_path}/inference_results.jsonl")
