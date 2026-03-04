# IMO-Symbolic: Math Problem Dataset Generator and Evaluator

This project provides a complete pipeline for generating symbolic-augmented mathematical problem datasets and evaluating Large Language Models (LLMs) on their reasoning capabilities. It covers several complex problems (Dodola Island, Triangle Geometry, Fibonacci sequences, etc.) by parameterizing them to generate unique, verifiable samples.

## Project Structure

- `/solutions/`: Contains `solve_{id}.py` scripts with the core mathematical logic for each problem.
- `/datasets/`: Stores generated `.jsonl` datasets.
- `/results/`: Stores evaluation outputs and logs.
- `run_{id}.py`: Root-level scripts to generate augmented data for specific tasks.
- `evaluate_models.py`: The evaluation engine supporting OpenAI (Standard & Azure) and Google Gemini.
- `run_all.py`: Orchestration script to generate and evaluate all tasks in one command.

## Prerequisites

1. **Install Dependencies**:
   ```bash
   pip install openai google-genai
   ```

2. **Set Environment Variables**:
   Depending on which model you want to evaluate, set the following keys:

   **For Google Gemini**:
   ```bash
   export GEMINI_API_KEY="your-gemini-api-key"
   ```

   **For Azure OpenAI**:
   ```bash
   export AZURE_OPENAI_API_KEY="your-azure-api-key"
   export AZURE_OPENAI_ENDPOINT="https://<your-subdomain>.openai.azure.com/"
   export AZURE_OPENAI_API_VERSION="2025-04-01-preview"
   ```

   **For Standard OpenAI**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

## Usage

### 1. Run the Full Pipeline
The `run_all.py` script automates the generation of datasets and their evaluation against a selected model.

```bash
# Generate 10 samples per task and evaluate using Gemini Thinking Pro
python run_all.py --num_samples 10 --provider gemini --model gemini-2.0-flash-thinking-exp-01-21

# Skip generation and evaluate existing datasets using OpenAI o1
python run_all.py --skip_gen --provider openai --model o1
```

### 2. Generate Data for a Single Task
You can run individual generators located in the root directory.

```bash
# Example: Generate 50 samples for the Triangle Geometry problem
python run_1acac0.py --num_samples 50 --output datasets/my_triangle_data.jsonl
```

### 3. Customize Data Generation
Each `run_{id}.py` script contains logic for sampling parameters (e.g., ranges for side lengths, periods, or number of erased integers). To change how the data is augmented (e.g., changing the range from `[10, 400]` to `[100, 1000]`), simply edit the `get_valid_integer_pairs` or `generate_periods` functions within the respective `run_*.py` script.

## Supported Tasks

| ID | Task Name | Description |
|---|---|---|
| `057f8a` | Dodola | Airline scheduling/gap problem |
| `1acac0` | Triangle | Greatest possible length of altitude |
| `bbd91e` | Alice Board | Mean of remaining numbers after erasure |
| `a1d40b` | Fibonacci Prime | Multiplicity of prime factors in N |
| `480182` | Triangle Segment | Coprime ratio of segments in a triangle |
| `349493` | Delightful Seq | Counting frequency-based sequences |
| `88c219` | Artificial Int | Sum of artificial integers in a range |
| `71beb6` | Digit Sum | Nested sum of digits of a large range |
| `1fce4b` | 3-Digit Divisor | Invariant divisor for repeated numbers |
