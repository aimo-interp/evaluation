from datasets import load_dataset

# Local CSV path
csv_path = "data/competition_sample/aimo2_robustness_subset.csv"

# Change this to your target dataset repo
repo_id = "michal-stefanik/aimo-interp-challenge-sample"

# Load as a DatasetDict with one split
dataset = load_dataset("csv", data_files={"val": csv_path})

# Push to HF Hub
# Requires:
#   pip install datasets huggingface_hub
#   huggingface-cli login
dataset.push_to_hub(repo_id)
