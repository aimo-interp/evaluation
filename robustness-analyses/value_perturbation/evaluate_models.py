import argparse
import json
import os
import re
import time

# Attempt to import SDKs
try:
    from openai import OpenAI, AzureOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from google import genai
    from google.genai import types
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

SYSTEM_PROMPT = """You are an expert mathematician. Your task is to solve the following mathematical problem step-by-step.

Work through your logic clearly step-by-step. You are not allowed to execute any code. Provide an analytical solution.
Each problem has exactly one numerical solution (integer). You MUST provide your final answer at the very end on a new line exactly in the format:
The result is: <integer>

Problem:
{problem_text}
"""

def extract_answer(text: str) -> int:
    """Extracts the final numeric answer from the model's text response."""
    # Look for 'The result is: <number>'
    match = re.search(r"The result is:\s*(\d+)", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Fallback to finding the last number in the text if exact string is not found
    numbers = re.findall(r"\b\d+\b", text)
    if numbers:
        return int(numbers[-1])
    return None

def evaluate_openai(client, problem_text: str, model_name: str) -> str:
    """Evaluate using OpenAI's reasoning models (o1, o3-mini)."""
    prompt = SYSTEM_PROMPT.format(problem_text=problem_text)
    
    # Note: o1 models do not support the developer/system role natively in older beta API,
    # and they ignore temperature. So we use a purely user message format.
    # o3-mini supports reasoning_effort
    kwargs = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "reasoning_effort": "high"
    }
    
    if "o3-mini" in model_name:
        kwargs["reasoning_effort"] = "high"
        
    response = client.chat.completions.create(**kwargs)
    print(response)
    return response.choices[0].message.content

def evaluate_gemini(client, problem_text: str, model_name: str) -> str:
    """Evaluate using Google's generative models via genai."""
    prompt = SYSTEM_PROMPT.format(problem_text=problem_text)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
        tools=[] # Empty list ensures no tools (like code execution) are active
    )
    )
    return response.text

def process_dataset(input_file: str, output_file: str, provider: str, model_name: str, delay: int = 5):
    """Loops through the dataset, sends prompts to the specified model, and logs results."""
    
    if provider == "openai":
        if not HAS_OPENAI:
            print("Error: openai library is not installed. Run `pip install openai`.")
            return
        
        # Check if Azure environment variables are set
        if "AZURE_OPENAI_ENDPOINT" in os.environ:
            if "AZURE_OPENAI_API_KEY" not in os.environ:
                print("Error: AZURE_OPENAI_API_KEY environment variable not set.")
                return
                # Quick connectivity check
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("Set the AZURE_OPENAI_API_KEY environment variable.")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            if not endpoint:
                raise EnvironmentError("Set the AZURE_OPENAI_ENDPOINT environment variable.")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
            import urllib.request
            try:
                req = urllib.request.Request(
                    f"{endpoint.rstrip('/')}/openai/models?api-version={api_version}",
                    headers={"api-key": api_key},
                )
                urllib.request.urlopen(req, timeout=10)
                print(f"✓ Successfully connected to {endpoint}")
            except Exception as e:
                print(f"✗ Cannot reach endpoint {endpoint}: {e}")
                print("  Check your network, VPN, firewall, or proxy settings.")
                raise
            api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
            client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version
            )
            print(f"Using Azure OpenAI API (Endpoint: {os.environ['AZURE_OPENAI_ENDPOINT']})")
        else:
            if "OPENAI_API_KEY" not in os.environ:
                print("Error: OPENAI_API_KEY environment variable not set.")
                return
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            print("Using standard OpenAI API")
            
        # default to o1 if not specified
        model_name = model_name or "o1"
    
    elif provider == "gemini":
        if not HAS_GEMINI:
            print("Error: google-genai library is not installed. Run `pip install google-genai`.")
            return
        if "GEMINI_API_KEY" not in os.environ:
            print("Error: GEMINI_API_KEY environment variable not set.")
            return
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        # Default to the experimental thinking model, or the standard pro
        model_name = model_name or "gemini-2.0-flash-thinking-exp-01-21"
    else:
        print(f"Unknown provider: {provider}")
        return

    print(f"Starting evaluation using {provider.upper()} (Model: {model_name})")
    
    correct = 0
    total = 0
    results = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    
    for i, line in enumerate(lines):
        record = json.loads(line)
        problem_text = record["textual_problem"]
        ground_truth = record["numeric_solution"]
        
        print(f"[{i+1}/{len(lines)}] Evaluating: {record['params']}...")
        
        # Simple retry loop
        attempts = 10
        model_response = ""
        for attempt in range(attempts):
            try:                
                if provider == "openai":
                    model_response = evaluate_openai(client, problem_text, model_name)
                else:
                    model_response = evaluate_gemini(client, problem_text, model_name)
                break
            except Exception as e:
                print(f"  Attempt {attempt+1} failed: {e}")
                time.sleep(10)
        
        if not model_response:
            print("  Failed to get response after multiple attempts. Skipping.")
            continue
            
        predicted = extract_answer(model_response)
        is_correct = (predicted == ground_truth)
        
        if is_correct:
            correct += 1
            print(f"  -> CORRECT! Predicted: {predicted}, Ground Truth: {ground_truth}")
        else:
            print(f"  -> WRONG! Predicted: {predicted}, Ground Truth: {ground_truth}")
            
        total += 1
        
        # Save detailed log
        result_record = {
            "textual_problem": problem_text,
            "params": record["params"],
            "ground_truth": ground_truth,
            "predicted": predicted,
            "is_correct": is_correct,
            "model_response": model_response
        }
        results.append(result_record)
        with open(output_file, 'a') as out_f:
            # Write only the NEW record as a single line
            out_f.write(json.dumps(result_record) + "\n")
                
        # Sleep to avoid rate limits
        if i < len(lines) - 1:
            time.sleep(delay)
                
    accuracy = (correct / total * 100) if total > 0 else 0
    print("-" * 50)
    print(f"Evaluation Complete!")
    print(f"Total Evaluated: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Detailed results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Symbolic Math Dataset with LLMs")
    parser.add_argument("--dataset", type=str, default="dodola_dataset.jsonl", help="Path to input dataset JSONL")
    parser.add_argument("--output", type=str, default="evaluation_results.jsonl", help="Path to output results JSONL")
    parser.add_argument("--provider", type=str, choices=["openai", "gemini"], required=True, help="LLM Provider to evaluate")
    parser.add_argument("--model", type=str, default="", help="Specific model name (e.g. o1, o3-mini, gemini-2.0-flash-thinking-exp-01-21)")
    parser.add_argument("--delay", type=int, default=10, help="Delay between API requests to prevent rate limiting (seconds)")
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset {args.dataset} not found.")
        return
        
    process_dataset(args.dataset, args.output, args.provider, args.model, args.delay)

if __name__ == "__main__":
    main()