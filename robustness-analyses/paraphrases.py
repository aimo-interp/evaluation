"""
Robustness analysis: Evaluating LLMs' sensitivity to paraphrased mathematical problems.
Paraphrases are domain-shifted reformulations that preserve structure and final answers.
"""

import csv
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from openai import AzureOpenAI
from google import genai

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Problem:
    id: str
    question: str
    answer: str


@dataclass
class EvaluationResult:
    problem_id: str
    original_question: str
    paraphrased_question: str
    expected_answer: str
    original_response: str
    paraphrased_response: str
    original_correct: bool
    paraphrased_correct: bool

    @property
    def is_detrimental(self) -> bool:
        """Paraphrase is detrimental when original was correct but paraphrase was not."""
        return self.original_correct and not self.paraphrased_correct

    @property
    def is_beneficial(self) -> bool:
        """Paraphrase is beneficial when original was wrong but paraphrase was correct."""
        return not self.original_correct and self.paraphrased_correct


# ---------------------------------------------------------------------------
# 1. Read problems from CSV
# ---------------------------------------------------------------------------

def read_problems(csv_path: str) -> list[Problem]:
    """Read problems from a CSV file with headers (id, question, answer)."""
    problems: list[Problem] = []
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"id", "question", "answer"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV must contain headers: {required}")
        for row in reader:
            problems.append(Problem(
                id=row["id"].strip(),
                question=row["question"].strip(),
                answer=row["answer"].strip(),
            ))
    print(f"Loaded {len(problems)} problems from {csv_path}")
    return problems


# ---------------------------------------------------------------------------
# 2. Paraphrase problems via LLM
# ---------------------------------------------------------------------------

PARAPHRASE_SYSTEM_PROMPT = """\
You are a mathematical problem paraphraser. Your task is to rewrite a math \
problem into a different real-world domain while strictly preserving:
1. The mathematical structure (operations, relationships, constraints).
2. The numerical values and the final answer.

Rules:
- Change the surface-level context/domain (e.g., from apples to cars, from \
  a classroom to a warehouse).
- Do NOT change any numbers, ratios, or the logical steps needed to solve it.
- The answer to the paraphrased problem MUST be identical to the original.
- Output ONLY the paraphrased problem text, nothing else.
"""


def _get_client() -> AzureOpenAI:
    """Instantiate an Azure OpenAI client."""
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set the AZURE_OPENAI_API_KEY environment variable.")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not endpoint:
        raise EnvironmentError("Set the AZURE_OPENAI_ENDPOINT environment variable.")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")

    # Quick connectivity check
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

    return AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )

def _get_gemini_client() -> genai.Client:
    """Instantiate a Google Gemini client."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("Set the GOOGLE_API_KEY environment variable.")
    client = genai.Client(api_key=api_key)
    print("✓ Gemini client initialised")
    return client


def paraphrase_problem(
    client: AzureOpenAI,
    problem: Problem,
    model: str = "gpt-5.2-2025-12-11",
    max_retries: int = 3,
) -> str:
    """Call GPT-5.2 to paraphrase a single problem into a different domain."""
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.7,
                max_completion_tokens=1024,
                messages=[
                    {"role": "system", "content": PARAPHRASE_SYSTEM_PROMPT},
                    {"role": "user", "content": problem.question},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [attempt {attempt}/{max_retries}] Paraphrase failed for "
                  f"problem {problem.id}: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to paraphrase problem {problem.id} after {max_retries} retries.")


def paraphrase_problem_gemini(
    client: genai.Client,
    problem: "Problem",
    model: str = "gemini-2.5-flash",
    max_retries: int = 3,
) -> str:
    """Call Gemini to paraphrase a single problem into a different domain."""
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=problem.question,
                config=genai.types.GenerateContentConfig(
                    system_instruction=PARAPHRASE_SYSTEM_PROMPT,
                    temperature=0.7,
                    max_output_tokens=4096*4,
                ),
            )
            return response.text.strip()
        except Exception as e:
            print(f"  [attempt {attempt}/{max_retries}] Paraphrase failed for "
                  f"problem {problem.id}: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                raise


def paraphrase_all(
    problems: list["Problem"],
    model: str = "gpt-5.2-2025-12-11",
    provider: str = "azure",
) -> dict[str, str]:
    """Return a mapping {problem_id: paraphrased_question} for every problem."""
    if provider == "gemini":
        client = _get_gemini_client()
    else:
        client = _get_client()

    paraphrases: dict[str, str] = {}
    for i, prob in enumerate(problems, 1):
        print(f"Paraphrasing [{i}/{len(problems)}] problem {prob.id} …")
        if provider == "gemini":
            para = paraphrase_problem_gemini(client, prob, model=model)
        else:
            para = paraphrase_problem(client, prob, model=model)
        paraphrases[prob.id] = para
        print(f"  Original:    {prob.question}")
        print(f"  Paraphrased: {para}")
        print()
    return paraphrases


# ---------------------------------------------------------------------------
# 3. Evaluate on original & paraphrased problems
# ---------------------------------------------------------------------------

SOLVE_SYSTEM_PROMPT = """\
You are a precise math problem solver. Solve the given problem step by step, \
then output your final answer on the last line in the exact format:
ANSWER: <your answer>
"""


def solve_problem(client: AzureOpenAI, question: str, model: str = "gpt-5.2-2025-12-11") -> str:
    """Ask the model to solve a problem and return the raw response."""
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_completion_tokens=4096,
        messages=[
            {"role": "system", "content": SOLVE_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content.strip()


def solve_problem_gemini(client: genai.Client, question: str, model: str = "gemini-2.5-flash") -> str:
    """Ask Gemini to solve a problem and return the raw response."""
    response = client.models.generate_content(
        model=model,
        contents=question,
        config=genai.types.GenerateContentConfig(
            system_instruction=SOLVE_SYSTEM_PROMPT,
            temperature=0.0,
            max_output_tokens=2048,
        ),
    )
    return response.text.strip()


def extract_answer(response: str) -> str:
    """Extract the final answer from the model response."""
    for line in reversed(response.splitlines()):
        line = line.strip()
        if line.upper().startswith("ANSWER:"):
            return line.split(":", 1)[1].strip()
    # Fallback: return last non-empty line
    for line in reversed(response.splitlines()):
        if line.strip():
            return line.strip()
    return ""


def normalize(text: str) -> str:
    """Normalize an answer string for comparison."""
    t = text.strip().lower().rstrip(".")
    # Strip dollar signs, commas, percent signs for numeric comparison
    for ch in ("$", ",", "%"):
        t = t.replace(ch, "")
    try:
        return str(float(t))
    except ValueError:
        return t


def answers_match(expected: str, predicted: str) -> bool:
    """Check whether the predicted answer matches the expected one."""
    return normalize(expected) == normalize(predicted)


def _sanitize_model_name(model: str) -> str:
    """Make a model name safe for use in filenames."""
    return model.replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")


def evaluate(
    problems: list["Problem"],
    paraphrases: dict[str, str],
    model: str = "gpt-5.2-2025-12-11",
    output_dir: str = "robustness-analyses/reports",
    provider: str = "azure",
) -> list["EvaluationResult"]:
    """Evaluate the model on both original and paraphrased problems."""
    if provider == "gemini":
        client = _get_gemini_client()
        _solve = lambda q: solve_problem_gemini(client, q, model=model)
    else:
        client = _get_client()
        _solve = lambda q: solve_problem(client, q, model=model)

    results: list[EvaluationResult] = []

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    safe_model = _sanitize_model_name(model)
    csv_path = out / f"evaluation_log_{safe_model}.csv"
    csv_fields = [
        "problem_id",
        "original_question",
        "paraphrased_question",
        "expected_answer",
        "original_predicted_answer",
        "paraphrased_predicted_answer",
        "original_correct",
        "paraphrased_correct",
    ]
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    writer.writeheader()
    csv_file.flush()
    print(f"Writing evaluation log to {csv_path}")

    for i, prob in enumerate(problems, 1):
        print(f"Evaluating [{i}/{len(problems)}] problem {prob.id} …")

        original_resp = _solve(prob.question)
        original_pred = extract_answer(original_resp)

        para_question = paraphrases[prob.id]
        para_resp = _solve(para_question)
        para_pred = extract_answer(para_resp)

        orig_correct = answers_match(prob.answer, original_pred)
        para_correct = answers_match(prob.answer, para_pred)

        result = EvaluationResult(
            problem_id=prob.id,
            original_question=prob.question,
            paraphrased_question=para_question,
            expected_answer=prob.answer,
            original_response=original_resp,
            paraphrased_response=para_resp,
            original_correct=orig_correct,
            paraphrased_correct=para_correct,
        )
        results.append(result)

        writer.writerow({
            "problem_id": prob.id,
            "original_question": prob.question,
            "paraphrased_question": para_question,
            "expected_answer": prob.answer,
            "original_predicted_answer": original_pred,
            "paraphrased_predicted_answer": para_pred,
            "original_correct": orig_correct,
            "paraphrased_correct": para_correct,
        })
        csv_file.flush()

    csv_file.close()
    return results


# ---------------------------------------------------------------------------
# 4. Report generation
# ---------------------------------------------------------------------------

def generate_report(
    results: list[EvaluationResult],
    output_dir: str = "robustness-analyses/reports",
    model: str = "gpt-5.2-2025-12-11",
) -> str:
    """Create a Markdown + JSON report with robustness statistics."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    safe_model = _sanitize_model_name(model)

    total = len(results)
    orig_correct = sum(r.original_correct for r in results)
    para_correct = sum(r.paraphrased_correct for r in results)
    detrimental = [r for r in results if r.is_detrimental]
    beneficial = [r for r in results if r.is_beneficial]
    both_correct = sum(r.original_correct and r.paraphrased_correct for r in results)
    both_wrong = sum(not r.original_correct and not r.paraphrased_correct for r in results)

    stats = {
        "total_problems": total,
        "original_accuracy": orig_correct / total if total else 0,
        "paraphrased_accuracy": para_correct / total if total else 0,
        "accuracy_delta": (para_correct - orig_correct) / total if total else 0,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "detrimental_count": len(detrimental),
        "detrimental_rate": len(detrimental) / total if total else 0,
        "beneficial_count": len(beneficial),
        "beneficial_rate": len(beneficial) / total if total else 0,
    }

    # --- JSON dump ----------------------------------------------------------
    json_path = out / f"paraphrase_robustness_{safe_model}.json"
    detailed = {
        "summary": stats,
        "detrimental_cases": [
            {
                "id": r.problem_id,
                "original_question": r.original_question,
                "paraphrased_question": r.paraphrased_question,
                "expected_answer": r.expected_answer,
                "original_response": r.original_response,
                "paraphrased_response": r.paraphrased_response,
            }
            for r in detrimental
        ],
        "beneficial_cases": [
            {
                "id": r.problem_id,
                "original_question": r.original_question,
                "paraphrased_question": r.paraphrased_question,
                "expected_answer": r.expected_answer,
            }
            for r in beneficial
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2, ensure_ascii=False)

    # --- Markdown report ----------------------------------------------------
    md_lines = [
        "# Paraphrase Robustness Report",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total problems | {total} |",
        f"| Original accuracy | {stats['original_accuracy']:.2%} |",
        f"| Paraphrased accuracy | {stats['paraphrased_accuracy']:.2%} |",
        f"| Accuracy delta | {stats['accuracy_delta']:+.2%} |",
        f"| Both correct | {both_correct} ({both_correct/total:.2%}) |" if total else "",
        f"| Both wrong | {both_wrong} ({both_wrong/total:.2%}) |" if total else "",
        f"| Detrimental (orig ✓ → para ✗) | {len(detrimental)} ({stats['detrimental_rate']:.2%}) |",
        f"| Beneficial  (orig ✗ → para ✓) | {len(beneficial)} ({stats['beneficial_rate']:.2%}) |",
        "",
        "## Detrimental Cases",
        "",
    ]

    if detrimental:
        for r in detrimental:
            md_lines += [
                f"### Problem {r.problem_id}",
                "",
                f"**Original question:** {r.original_question}",
                "",
                f"**Paraphrased question:** {r.paraphrased_question}",
                "",
                f"**Expected answer:** {r.expected_answer}",
                "",
                f"**Original response (correct):**",
                f"```",
                r.original_response,
                f"```",
                "",
                f"**Paraphrased response (incorrect):**",
                f"```",
                r.paraphrased_response,
                f"```",
                "",
            ]
    else:
        md_lines.append("No detrimental cases found.\n")

    md_path = out / f"paraphrase_robustness_{safe_model}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\nReports saved to:\n  {md_path}\n  {json_path}")
    return str(md_path)


# ---------------------------------------------------------------------------
# Main – Stage 1: Paraphrasing
# ---------------------------------------------------------------------------

def run_paraphrase(
    csv_path: str,
    model: str = "gpt-5.2-2025-12-11",
    provider: str = "azure",
) -> str:
    """Stage 1: Read problems, paraphrase them, write a new CSV with a 'paraphrase' column."""
    problems = read_problems(csv_path)

    out = Path(csv_path).parent
    out.mkdir(parents=True, exist_ok=True)
    safe_model = _sanitize_model_name(model)
    out_csv = out / f"paraphrased_problems_{safe_model}.csv"

    if provider == "gemini":
        client = _get_gemini_client()
    else:
        client = _get_client()

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "answer", "paraphrase"])
        writer.writeheader()
        f.flush()
        print(f"Writing paraphrased CSV to {out_csv}")

        for i, prob in enumerate(problems, 1):
            print(f"Paraphrasing [{i}/{len(problems)}] problem {prob.id} …")
            if provider == "gemini":
                para = paraphrase_problem_gemini(client, prob, model=model)
            else:
                para = paraphrase_problem(client, prob, model=model)

            print(f"  Original:    {prob.question}")
            print(f"  Paraphrased: {para}")
            print()

            writer.writerow({
                "id": prob.id,
                "question": prob.question,
                "answer": prob.answer,
                "paraphrase": para,
            })
            f.flush()

    print(f"\nParaphrased CSV saved to: {out_csv}")
    return str(out_csv)


# ---------------------------------------------------------------------------
# Main – Stage 2: Evaluation
# ---------------------------------------------------------------------------

def read_paraphrased_csv(csv_path: str) -> tuple[list[Problem], dict[str, str]]:
    """Read a paraphrased CSV (id, question, answer, paraphrase) and return problems + paraphrases."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    problems: list[Problem] = []
    paraphrases: dict[str, str] = {}

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"id", "question", "answer", "paraphrase"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV must contain headers: {required}")
        for row in reader:
            pid = row["id"].strip()
            problems.append(Problem(
                id=pid,
                question=row["question"].strip(),
                answer=row["answer"].strip(),
            ))
            paraphrases[pid] = row["paraphrase"].strip()

    print(f"Loaded {len(problems)} paraphrased problems from {csv_path}")
    return problems, paraphrases


def run_evaluate(
    paraphrased_csv_path: str,
    model: str = "gpt-5.2-2025-12-11",
    output_dir: str = "robustness-analyses/reports",
    provider: str = "azure",
) -> str:
    """Stage 2: Read paraphrased CSV, evaluate on original & paraphrased, generate report."""
    problems, paraphrases = read_paraphrased_csv(paraphrased_csv_path)
    results = evaluate(problems, paraphrases, model=model, output_dir=output_dir, provider=provider)
    report_path = generate_report(results, output_dir=output_dir, model=model)
    return report_path


if __name__ == "__main__":
    """
    Usage:
      Azure:
        export AZURE_OPENAI_API_KEY="..."
        export AZURE_OPENAI_API_VERSION="2025-04-01-preview"
        export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com"
        python robustness-analyses/paraphrases.py data/problems.csv --provider azure

      Gemini:
        export GOOGLE_API_KEY="..."
        python robustness-analyses/paraphrases.py data/problems.csv --provider gemini --model gemini-2.5-flash
        
    Stage 1 – Paraphrase:
        python robustness-analyses/paraphrases.py paraphrase data/problems.csv --provider azure
        python robustness-analyses/paraphrases.py paraphrase data/problems.csv --provider gemini --model gemini-2.5-flash

    Stage 2 – Evaluate:
        python robustness-analyses/paraphrases.py evaluate robustness-analyses/reports/paraphrased_problems_gpt-5.2-2025-12-11.csv --provider azure
        python robustness-analyses/paraphrases.py evaluate robustness-analyses/reports/paraphrased_problems_gemini-2.5-flash.csv --provider gemini --model gemini-2.5-flash
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate LLM robustness to paraphrased math problems."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- paraphrase sub-command ---
    p_para = subparsers.add_parser(
        "paraphrase",
        help="Stage 1: Paraphrase problems and produce a CSV with a 'paraphrase' column.",
    )
    p_para.add_argument("csv", help="Path to original CSV file with columns: id, question, answer")
    p_para.add_argument("--model", default=None,
                        help="Model used for paraphrasing (default per provider)")
    p_para.add_argument("--provider", choices=["azure", "gemini"], default="azure",
                        help="LLM provider to use (default: azure)")

    # --- evaluate sub-command ---
    p_eval = subparsers.add_parser(
        "evaluate",
        help="Stage 2: Evaluate a model on original & paraphrased problems from a paraphrased CSV.",
    )
    p_eval.add_argument("csv", help="Path to paraphrased CSV (columns: id, question, answer, paraphrase)")
    p_eval.add_argument("--model", default=None,
                        help="Model used for evaluation (default per provider)")
    p_eval.add_argument("--output-dir", default="robustness-analyses/reports",
                        help="Directory for output reports")
    p_eval.add_argument("--provider", choices=["azure", "gemini"], default="azure",
                        help="LLM provider to use (default: azure)")

    args = parser.parse_args()

    # Set sensible default model per provider
    if args.model is None:
        args.model = "gemini-2.5-pro" if args.provider == "gemini" else "gpt-5.2-2025-12-11"

    if args.command == "paraphrase":
        run_paraphrase(args.csv, model=args.model, provider=args.provider)
    elif args.command == "evaluate":
        run_evaluate(args.csv, model=args.model, output_dir=args.output_dir, provider=args.provider)
