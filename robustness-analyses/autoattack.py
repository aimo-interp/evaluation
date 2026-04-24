import asyncio
import hashlib
import json
import os
import pathlib
import random
import re
import uuid as uuid_module
from typing import Annotated

import openai
import typer
import yaml
from pydantic import BaseModel, ConfigDict
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TaskID, TextColumn

app = typer.Typer(help="Generate adversarial variants and evaluate model accuracy.")
console = Console()


# ── Utilities ──────────────────────────────────────────────────────────────────

def load_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: pathlib.Path, record: dict) -> None:
    with open(path, "a", buffering=1) as f:
        f.write(json.dumps(record) + "\n")


def make_seed(master_seed: int, *parts) -> int:
    """Deterministic seed from master_seed + call identifier parts (fits in signed int32)."""
    key = f"{master_seed}:" + ":".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big") & 0x7FFFFFFF


def extract_number(text: str) -> str:
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    return matches[-1] if matches else ""


def normalize_answer(text: str) -> str:
    t = str(text).strip().lower().rstrip(".")
    for ch in ("$", ",", "%"):
        t = t.replace(ch, "")
    try:
        return str(float(t))
    except ValueError:
        return t


def answers_match(expected: str, predicted: str) -> bool:
    return normalize_answer(str(expected)) == normalize_answer(str(predicted))


def parse_variant_output(raw: str) -> tuple[str, str]:
    """Parse model A JSON output into (question, answer). Falls back gracefully."""
    clean = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    try:
        data = json.loads(clean)
        return str(data["question"]), str(data["answer"])
    except (json.JSONDecodeError, KeyError):
        return raw.strip(), extract_number(raw)


def _get_client(provider: str) -> openai.AsyncOpenAI:
    if provider == "google":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set.")
        return openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    if provider == "einfra":
        api_key = os.getenv("EINFRA_AI_TOKEN")
        if not api_key:
            raise EnvironmentError("EINFRA_AI_TOKEN not set.")
        return openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://llm.ai.e-infra.cz/v1/",
        )
    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENROUTER_API_KEY not set.")
        return openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
    if provider == "local":
        base_url = os.getenv("LOCAL_API_BASE_URL", "http://localhost:8000/v1/")
        return openai.AsyncOpenAI(
            api_key=os.getenv("LOCAL_API_KEY", "local"),
            base_url=base_url,
        )
    if provider == "openai":
        import urllib.request
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
        if not api_key:
            raise EnvironmentError("AZURE_OPENAI_API_KEY not set.")
        if not endpoint:
            raise EnvironmentError("AZURE_OPENAI_ENDPOINT not set.")
        try:
            req = urllib.request.Request(
                f"{endpoint.rstrip('/')}/openai/models?api-version={api_version}",
                headers={"api-key": api_key},
            )
            urllib.request.urlopen(req, timeout=10)
            console.log(f"Connected to {endpoint}")
        except Exception as exc:
            console.log(f"[yellow]Warning: cannot reach {endpoint}: {exc}[/yellow]")
        return openai.AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
    raise ValueError(f"Unknown provider: {provider!r}. Choose 'openai', 'google', 'einfra', 'openrouter', or 'local'.")


# ── Config ─────────────────────────────────────────────────────────────────────

class ModelSpec(BaseModel):
    """YAML schema for a single model (attacker or solver)."""
    provider: str
    model: str
    prompt_file: pathlib.Path
    temperature: float = 1.0
    reasoning_effort: str | None = None
    max_tokens: int | None = None
    max_concurrency: int = 20


class RunConfig(BaseModel):
    """Full YAML config schema."""
    attacker: ModelSpec
    solver: ModelSpec
    n_variants: int = 10
    n_initial_predictions: int = 1
    accuracy_drop_threshold: float = 0.1
    n_extra_predictions: int = 9
    max_retries: int = 10
    retry_sleep_secs: float = 30.0
    master_seed: int = 42
    demonstration_probability: float = 0.0
    n_demonstrations: int = 3


class ModelConfig(BaseModel):
    """Runtime model config (after prompt loading and client construction)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    provider: str
    system_prompt: str
    temperature: float
    reasoning_effort: str | None
    max_tokens: int | None
    client: openai.AsyncOpenAI
    sem: asyncio.Semaphore


class Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    attacker: ModelConfig   # model A: generates problem variants
    solver: ModelConfig     # model B: solves variants
    n_variants: int
    n_initial_predictions: int
    accuracy_drop_threshold: float
    n_extra_predictions: int
    max_retries: int
    retry_sleep_secs: float
    master_seed: int
    demonstration_probability: float
    n_demonstrations: int



def compute_base_accuracy_map(records: list[dict]) -> dict[str, float]:
    """Compute per-problem accuracy from a predictions JSONL (groups by 'id')."""
    from collections import defaultdict
    groups: dict[str, list[bool]] = defaultdict(list)
    for r in records:
        groups[str(r["id"])].append(answers_match(str(r["answer"]), str(r["predicted_result"])))
    return {pid: sum(correct) / len(correct) for pid, correct in groups.items()}


def load_run_config(config_path: pathlib.Path) -> RunConfig:
    """Parse and validate the YAML config file."""
    with open(config_path) as f:
        return RunConfig.model_validate(yaml.safe_load(f))


def resolve_prompt(spec: ModelSpec, base_dir: pathlib.Path, role: str) -> str:
    """Resolve prompt_file relative to base_dir and return its contents."""
    path = spec.prompt_file if spec.prompt_file.is_absolute() else base_dir / spec.prompt_file
    if not path.exists():
        console.log(f"[red]{role} prompt file {path} not found.[/red]")
        raise typer.Exit(1)
    console.log(f"Loading {role} prompt from {path}...")
    return path.read_text()


def build_model_config(spec: ModelSpec, system_prompt: str) -> ModelConfig:
    return ModelConfig(
        name=spec.model,
        provider=spec.provider,
        system_prompt=system_prompt,
        temperature=spec.temperature,
        reasoning_effort=spec.reasoning_effort,
        max_tokens=spec.max_tokens,
        client=_get_client(spec.provider),
        sem=asyncio.Semaphore(spec.max_concurrency),
    )



async def call_llm(
    model_cfg: ModelConfig,
    user_content: str,
    call_id: str,
    seed: int | None,
    max_retries: int,
    retry_sleep_secs: float,
) -> tuple[str, str | None, str]:
    """Call the LLM API with retries. Returns (content, reasoning_content, finish_reason)."""
    kwargs: dict = {}
    if seed is not None:
        kwargs["seed"] = seed
    if model_cfg.reasoning_effort is not None:
        if model_cfg.provider == "openrouter":
            kwargs["extra_body"] = {"reasoning": {"effort": model_cfg.reasoning_effort}}
        else:
            kwargs["reasoning_effort"] = model_cfg.reasoning_effort
    if model_cfg.max_tokens is not None:
        kwargs["max_completion_tokens"] = model_cfg.max_tokens

    for attempt in range(1, max_retries + 1):
        try:
            response = await model_cfg.client.chat.completions.create(
                model=model_cfg.name,
                temperature=model_cfg.temperature,
                messages=[
                    {"role": "system", "content": model_cfg.system_prompt},
                    {"role": "user", "content": user_content},
                ],
                timeout=2 * 60 * 60,
                **kwargs,
            )
            msg = response.choices[0].message
            content = msg.content or ""
            reasoning = getattr(msg, "reasoning_content", None)
            finish_reason = response.choices[0].finish_reason
            if not content:
                console.log(f"[yellow][{call_id}] Empty response.[/yellow]")
            return content, reasoning, finish_reason
        except Exception as exc:
            if attempt < max_retries:
                console.log(f"[red][{call_id}] attempt {attempt}/{max_retries}: {exc}[/red]")
                await asyncio.sleep(retry_sleep_secs)
    console.log(f"[red][{call_id}] failed after {max_retries} retries.[/red]")
    raise RuntimeError(f"[{call_id}] failed after {max_retries} retries.")



def _build_demonstrations(variants_map: dict, preds_map: dict, n: int, rng: random.Random) -> str:
    """Format up to n successful past attacks as few-shot demonstrations."""
    successful = [
        v for v in variants_map.values()
        if (v["variant_uuid"], 0) in preds_map
        and not preds_map[(v["variant_uuid"], 0)]["is_correct"]
    ]
    if not successful:
        return ""
    chosen = rng.sample(successful, min(n, len(successful)))
    lines = ["Here are examples of problem variants that successfully fooled the solver:\n"]
    for v in chosen:
        lines.append(f"Original: {v['source_question']}")
        lines.append(json.dumps({"question": v["question"], "answer": v["answer"]}))
        lines.append("")
    lines.append("Now create a variant for the following problem:")
    return "\n".join(lines) + "\n"


async def _generate_variant(
    problem: dict,
    variant_idx: int,
    cfg: Config,
    variants_map: dict,
    preds_map: dict,
) -> dict:
    """Generate one variant of a problem using the attacker model."""
    seed = make_seed(cfg.master_seed, problem["id"], "variant", variant_idx)

    user_content = problem["question"]
    if cfg.demonstration_probability > 0:
        rng = random.Random(make_seed(cfg.master_seed, problem["id"], "demo", variant_idx))
        if rng.random() < cfg.demonstration_probability:
            demos = _build_demonstrations(variants_map, preds_map, cfg.n_demonstrations, rng)
            if demos:
                user_content = demos + user_content

    async with cfg.attacker.sem:
        raw, reasoning, finish_reason = await call_llm(
            model_cfg=cfg.attacker,
            user_content=user_content,
            call_id=f"{problem['id']}/v{variant_idx}",
            seed=seed,
            max_retries=cfg.max_retries,
            retry_sleep_secs=cfg.retry_sleep_secs,
        )
    question, answer = parse_variant_output(raw)
    return {
        "variant_uuid": str(uuid_module.uuid4()),
        "source_id": problem["id"],
        "source_question": problem["question"],
        "source_answer": problem["answer"],
        "variant_idx": variant_idx,
        "question": question,
        "answer": answer,
        "raw_output": raw,
        "reasoning_content": reasoning,
        "finish_reason": finish_reason,
        "model_attacker": cfg.attacker.name,
        "model_attacker_provider": cfg.attacker.provider,
        "model_attacker_temperature": cfg.attacker.temperature,
        "model_attacker_reasoning_effort": cfg.attacker.reasoning_effort,
        "model_attacker_max_tokens": cfg.attacker.max_tokens,
        "model_attacker_seed": seed,
        "model_attacker_system_prompt": cfg.attacker.system_prompt,
    }


async def _generate_prediction(variant: dict, repeat_idx: int, cfg: Config) -> dict:
    """Generate one prediction for a variant using the solver model."""
    seed = make_seed(cfg.master_seed, variant["source_id"], "pred", variant["variant_idx"], repeat_idx)
    async with cfg.solver.sem:
        raw, reasoning, finish_reason = await call_llm(
            model_cfg=cfg.solver,
            user_content=variant["question"],
            call_id=f"{variant['source_id']}/v{variant['variant_idx']}/p{repeat_idx}",
            seed=seed,
            max_retries=cfg.max_retries,
            retry_sleep_secs=cfg.retry_sleep_secs,
        )
    predicted = extract_number(raw)
    return {
        "prediction_uuid": str(uuid_module.uuid4()),
        "variant_uuid": variant["variant_uuid"],
        "repeat_idx": repeat_idx,
        "prediction": raw,
        "predicted_result": predicted,
        "is_correct": answers_match(variant["answer"], predicted),
        "reasoning_content": reasoning,
        "finish_reason": finish_reason,
        "model_solver": cfg.solver.name,
        "model_solver_provider": cfg.solver.provider,
        "model_solver_temperature": cfg.solver.temperature,
        "model_solver_reasoning_effort": cfg.solver.reasoning_effort,
        "model_solver_max_tokens": cfg.solver.max_tokens,
        "model_solver_seed": seed,
        "model_solver_system_prompt": cfg.solver.system_prompt,
    }


def _task_update(progress: Progress, task_id: TaskID, status: str, acc: str = "  —  ") -> None:
    progress.update(task_id, acc=acc, status=status)


async def _generate_variant_and_predict(
    problem: dict,
    variant_idx: int,
    cfg: Config,
    variants_map: dict,
    preds_map: dict,
    variants_file: pathlib.Path,
    predictions_file: pathlib.Path,
    progress: Progress,
    task_id: TaskID,
) -> None:
    """Generate one variant then immediately generate its initial prediction."""
    pid = problem["id"]

    variant = await _generate_variant(problem, variant_idx, cfg, variants_map, preds_map)
    append_jsonl(variants_file, variant)
    variants_map[(pid, variant["variant_idx"])] = variant
    progress.advance(task_id)
    _task_update(progress, task_id, "[cyan]creating variants[/cyan]")

    initial_preds = await asyncio.gather(*[
        _generate_prediction(variant, ri, cfg)
        for ri in range(cfg.n_initial_predictions)
    ])
    for pred in initial_preds:
        append_jsonl(predictions_file, pred)
        preds_map[(pred["variant_uuid"], pred["repeat_idx"])] = pred


async def process_problem(
    problem: dict,
    cfg: Config,
    variants_map: dict,
    preds_map: dict,
    variants_file: pathlib.Path,
    predictions_file: pathlib.Path,
    progress: Progress,
    task_id: TaskID,
    base_accuracy_map: dict[str, float],
) -> None:
    """Process one problem end-to-end."""
    pid = problem["id"]
    n = cfg.n_variants

    _task_update(progress, task_id, "[dim]waiting[/dim]")
    await asyncio.sleep(0)  # yield so the display renders "waiting" before work starts

    # Step 1: for each missing variant, generate and evaluate sequentially so that
    # each variant is assessed before the next one is created.
    missing_vis = [vi for vi in range(n) if (pid, vi) not in variants_map]
    if missing_vis:
        _task_update(progress, task_id, "[cyan]creating variants[/cyan]")
        for vi in missing_vis:
            await _generate_variant_and_predict(
                problem, vi, cfg, variants_map, preds_map,
                variants_file, predictions_file, progress, task_id,
            )

    variants = [variants_map[(pid, vi)] for vi in range(n)]

    n_init = cfg.n_initial_predictions

    # Step 2: fallback — predict for any existing variants that somehow lack initial predictions.
    missing_init = [
        (v, ri) for v in variants
        for ri in range(n_init)
        if (v["variant_uuid"], ri) not in preds_map
    ]
    if missing_init:
        _task_update(progress, task_id, "[yellow]solving[/yellow]")
        new_preds = await asyncio.gather(*[_generate_prediction(v, ri, cfg) for v, ri in missing_init])
        for p in new_preds:
            append_jsonl(predictions_file, p)
            preds_map[(p["variant_uuid"], p["repeat_idx"])] = p

    # Compute accuracy after initial predictions.
    initial_preds = [
        preds_map[(v["variant_uuid"], ri)]
        for v in variants
        for ri in range(n_init)
        if (v["variant_uuid"], ri) in preds_map
    ]
    accuracy = sum(p["is_correct"] for p in initial_preds) / len(initial_preds) if initial_preds else None
    acc_str = f"{accuracy:>5.0%}" if accuracy is not None else "  —  "

    # Step 3: decide whether to extend.
    already_extending = any((v["variant_uuid"], n_init) in preds_map for v in variants)
    if already_extending:
        should_extend = True
    else:
        ref = base_accuracy_map.get(str(pid))
        should_extend = accuracy is not None and ref is not None and accuracy < ref - cfg.accuracy_drop_threshold

    # Step 4: generate extra predictions if accuracy dropped.
    if should_extend:
        _task_update(progress, task_id, "[magenta]drilling deeper[/magenta]", acc_str)
        missing_extra = [
            (v, ri) for v in variants
            for ri in range(n_init, n_init + cfg.n_extra_predictions)
            if (v["variant_uuid"], ri) not in preds_map
        ]
        if missing_extra:
            extra_preds = await asyncio.gather(*[_generate_prediction(v, ri, cfg) for v, ri in missing_extra])
            for p in extra_preds:
                append_jsonl(predictions_file, p)
                preds_map[(p["variant_uuid"], p["repeat_idx"])] = p

        # Recompute accuracy over all predictions after drilling.
        all_preds = [
            preds_map[(v["variant_uuid"], ri)]
            for v in variants
            for ri in range(n_init + cfg.n_extra_predictions)
            if (v["variant_uuid"], ri) in preds_map
        ]
        final_accuracy = sum(p["is_correct"] for p in all_preds) / len(all_preds) if all_preds else accuracy
        acc_str = f"{final_accuracy:>5.0%}" if final_accuracy is not None else "  —  "
        _task_update(progress, task_id, "[bold red]attacked ✓[/bold red]", acc_str)
    else:
        _task_update(progress, task_id, "[green]done[/green]", acc_str)



@app.command()
def attack(
    base_problems_file: Annotated[
        pathlib.Path,
        typer.Argument(help="JSONL file with base problems (fields: id, question, answer)."),
    ],
    output_dir: Annotated[
        pathlib.Path,
        typer.Argument(help="Output directory. Will be created if it doesn't exist. Receives variants.jsonl, predictions.jsonl, and a copy of the config."),
    ],
    base_predictions_file: Annotated[
        pathlib.Path,
        typer.Option("--base-predictions", help="JSONL of baseline predictions (fields: id, answer, predicted_result). Used to compute per-problem reference accuracy."),
    ],
    config_file: Annotated[
        pathlib.Path,
        typer.Option("--config", "-c", help="YAML config file."),
    ] = pathlib.Path("configs/autoattack.yaml"),
) -> None:
    """Generate adversarial variants with model A and evaluate model B's accuracy."""

    if not config_file.exists():
        console.log(f"[red]Config file {config_file} not found.[/red]")
        raise typer.Exit(1)

    console.log(f"Loading config from {config_file}...")
    run_cfg = load_run_config(config_file)

    output_dir.mkdir(parents=True, exist_ok=True)
    variants_file = output_dir / "variants.jsonl"
    predictions_file = output_dir / "predictions.jsonl"

    import shutil
    shutil.copy(config_file, output_dir / "config.yaml")
    console.log(f"Output directory: {output_dir}")

    console.log(f"Loading base problems from {base_problems_file}...")
    problems = load_jsonl(base_problems_file)
    if not problems:
        console.log("[red]No problems found.[/red]")
        raise typer.Exit(1)
    console.log(f"Loaded {len(problems)} problems.")

    base_preds = load_jsonl(base_predictions_file)
    base_accuracy_map = compute_base_accuracy_map(base_preds)
    console.log(f"Loaded base accuracy for {len(base_accuracy_map)} problems from {base_predictions_file}.")

    existing_variants = load_jsonl(variants_file)
    existing_predictions = load_jsonl(predictions_file)
    if existing_variants:
        console.log(f"[yellow]Resuming: found {len(existing_variants)} existing variants.[/yellow]")
    if existing_predictions:
        console.log(f"[yellow]Resuming: found {len(existing_predictions)} existing predictions.[/yellow]")

    variants_map: dict[tuple, dict] = {(v["source_id"], v["variant_idx"]): v for v in existing_variants}
    preds_map: dict[tuple, dict] = {(p["variant_uuid"], p["repeat_idx"]): p for p in existing_predictions}

    base_dir = config_file.parent
    console.log(f"Initialising model A client ({run_cfg.attacker.provider}/{run_cfg.attacker.model})...")
    console.log(f"Initialising model B client ({run_cfg.solver.provider}/{run_cfg.solver.model})...")
    cfg = Config(
        attacker=build_model_config(
            run_cfg.attacker,
            resolve_prompt(run_cfg.attacker, base_dir, "attacker"),
        ),
        solver=build_model_config(
            run_cfg.solver,
            resolve_prompt(run_cfg.solver, base_dir, "solver"),
        ),
        n_variants=run_cfg.n_variants,
        n_initial_predictions=run_cfg.n_initial_predictions,
        accuracy_drop_threshold=run_cfg.accuracy_drop_threshold,
        n_extra_predictions=run_cfg.n_extra_predictions,
        max_retries=run_cfg.max_retries,
        retry_sleep_secs=run_cfg.retry_sleep_secs,
        master_seed=run_cfg.master_seed,
        demonstration_probability=run_cfg.demonstration_probability,
        n_demonstrations=run_cfg.n_demonstrations,
    )

    pid_width = max((len(str(p["id"])) for p in problems), default=10)

    async def _run() -> None:
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold cyan]{{task.fields[pid]:<{pid_width}}}[/bold cyan]"),
            BarColumn(bar_width=20),
            MofNCompleteColumn(),
            TextColumn("[dim]acc[/dim] [bold]{task.fields[acc]}[/bold]"),
            TextColumn("{task.fields[status]}"),
            console=console,
            refresh_per_second=8,
        ) as progress:
            task_ids: dict = {
                p["id"]: progress.add_task(
                    "",
                    total=run_cfg.n_variants,
                    completed=sum(
                        1 for vi in range(run_cfg.n_variants)
                        if (p["id"], vi) in variants_map
                    ),
                    pid=str(p["id"]),
                    acc="  —  ",
                    status="[dim]waiting[/dim]",
                )
                for p in problems
            }
            await asyncio.gather(*[
                process_problem(
                    problem, cfg, variants_map, preds_map,
                    variants_file, predictions_file,
                    progress, task_ids[problem["id"]],
                    base_accuracy_map,
                )
                for problem in problems
            ])

    asyncio.run(_run())
    console.log("[green]All problems processed.[/green]")


if __name__ == "__main__":
    app()
