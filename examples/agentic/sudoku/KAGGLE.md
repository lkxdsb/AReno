# Kaggle Training Checklist

These commands are intended to be run from the AReno repository root in a
Kaggle notebook with GPU enabled. They have been checked against the repository
CLI, but have not been executed on Kaggle by this CPU-only development pass.

## 1. Record the environment

```bash
!nvidia-smi
!nvcc --version
!python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_version", torch.version.cuda)
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

Install AReno using the repository script, then capture its checks:

```bash
!bash scripts/install.sh
!areno check
!areno env --json | tee /kaggle/working/areno-env.json
```

Use `--attn-backend native` on Tesla T4. AReno documents automatic fallback for
flash-attn-unsupported GPUs such as T4, but selecting `native` explicitly makes
the experiment command reproducible.

## 2. Generate and audit frozen data

Start smaller than the final target because each Sudoku row can require many
model requests:

```bash
!python examples/agentic/sudoku/dataset_generator.py \
  --output /kaggle/working/sudoku-train-smoke.jsonl \
  --count 30 --seed 2026 --split train --max-actions 8

!python examples/agentic/sudoku/dataset_generator.py \
  --output /kaggle/working/sudoku-validation.jsonl \
  --count 300 --seed 2027 --split validation

!python examples/agentic/sudoku/dataset_generator.py \
  --output /kaggle/working/sudoku-test.jsonl \
  --count 300 --seed 2028 --split test

!python examples/agentic/sudoku/audit_dataset.py \
  /kaggle/working/sudoku-train-smoke.jsonl \
  /kaggle/working/sudoku-validation.jsonl \
  /kaggle/working/sudoku-test.jsonl \
  --verify-solutions
```

The audit must return `"ok": true`. Next inspect the exact normalized training
contract:

```bash
!python .agents/skills/areno-run-training/scripts/inspect_dataset.py \
  --dataset-path /kaggle/working/sudoku-train-smoke.jsonl \
  --loader examples/agentic/sudoku/dataset_loader.py \
  --algo gspo
```

Do not continue unless this reports `"ok": true`.

## 3. Run one real trainer step

```bash
!areno train \
  --ckpt Qwen/Qwen3-0.6B \
  --model-hub modelscope \
  --dataset-path /kaggle/working/sudoku-train-smoke.jsonl \
  --dataset-loader-fn examples/agentic/sudoku/dataset_loader.py \
  --reward-fn-path examples/agentic/sudoku/reward.py \
  --agent-fn examples/agentic/sudoku/run_agent.py \
  --algo gspo \
  --tp-size 1 --world-size 1 \
  --batch-size 1 --n-samples 2 --mini-bs 1 \
  --max-running-prompts 2 \
  --max-new-tokens 64 --max-context-len 8192 \
  --attn-backend native \
  --max-steps 1 \
  --save-interval 1 \
  --metrics-log-dir /kaggle/working/metrics-smoke \
  --save-path /kaggle/working/checkpoint-smoke
```

Success means an optimization step actually advances and metrics are written;
model download or initialization alone is not sufficient. If memory is tight,
reduce `--max-running-prompts` to `1` before reducing semantic limits.

## 4. Scale in stages

After the one-step run succeeds:

1. Try `--max-steps 5` on the 30-record smoke split.
2. Inspect several trajectories and the reward distribution.
3. Generate 300 training records with the same train seed and repeat.
4. Only then consider 3,000 records or a larger model.

Example 300-record generation:

```bash
!python examples/agentic/sudoku/dataset_generator.py \
  --output /kaggle/working/sudoku-train-300.jsonl \
  --count 300 --seed 2026 --split train
```

Re-run `audit_dataset.py` and `inspect_dataset.py` whenever the dataset file
changes. Keep the validation and test files frozen and never pass them to
`areno train`.

## 5. What to retain

Save these artifacts together:

- repository commit from `git rev-parse HEAD`;
- `/kaggle/working/areno-env.json` and `nvidia-smi` output;
- exact training command and dataset seeds;
- metrics directory and checkpoint directory;
- replayable episode JSONL when available.

`evaluate.py` can summarize replayable episodes by difficulty:

```bash
!python examples/agentic/sudoku/evaluate.py \
  /kaggle/working/sudoku-episodes.jsonl --json
```

Run the same evaluation settings on the base checkpoint and trained checkpoint.
Dataset audit or a successful training loss step does not by itself establish
improved Sudoku solve rate.
