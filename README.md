# robustness-eval

## Set up:

```shell
git clone ...
cd ...
uv venv
source .venv/bin/activate
uv sync
```


## Generate variants

```shell
python robustness-analyses/main.py augment \
    data/aimo-2025-reference.jsonl \
    prompts/paraphrase.txt \
    data/ \
    --api-model "gpt-5.2-2025-12-11" \
    --n-variants 10
```


## Generate predictions

Predictions for base problems:

```shell
python robustness-analyses/main.py predict \
    data/aimo-2025-reference.jsonl \
    predictions/ \
    --api-model "gpt-5.2-2025-12-11" \
    --n-repeats 100
```

Predictions for augmented problems:

```shell
python robustness-analyses/main.py predict \
    data/aimo-2025-reference___paraphrase=gpt-5.2-2025-12-11.jsonl \
    predictions/ \
    --api-model "gpt-5.2-2025-12-11" \
    --n-repeats 10
```


## Evaluate 

Evaluate standalone base prediction file:

```shell
python robustness-analyses/main.py eval \
    predictions/aimo-2025-reference___eval=gpt-5.2-2025-12-11.jsonl
```

Evaluate augmented prediction file:

```shell
python robustness-analyses/main.py eval \
    predictions/aimo-2025-reference___paraphrase=gpt-5.2-2025-12-11___eval=gpt-5.2-2025-12-11.jsonl
```

Or compare the augmented to the base:

```shell
python robustness-analyses/main.py eval \
    predictions/aimo-2025-reference___paraphrase=gpt-5.2-2025-12-11___eval=gpt-5.2-2025-12-11.jsonl \
    --base-pred-file predictions/aimo-2025-reference___eval=gpt-5.2-2025-12-11.jsonl
```
