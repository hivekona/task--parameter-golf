# Parameter Golf — Train the Best Language Model in 16MB

Improve a GPT training script to minimize `val_bpb` (bits-per-byte on FineWeb validation) while fitting within a 16MB artifact budget.

## Setup

1. **Read the in-scope files**:
   - `train_gpt.py` — the file you modify. The GPT training script.
   - `eval/eval.sh` — runs training + evaluation. Do not modify.
   - `prepare.sh` — downloads FineWeb dataset + tokenizer. Do not modify.
   - `data/cached_challenge_fineweb.py` — data downloader. Do not modify.
2. **Run prepare**: `bash prepare.sh` to install dependencies and download the dataset.
3. **Verify data exists**: Check that `data/datasets/fineweb10B_sp1024/` has train/val `.bin` files and `data/tokenizers/fineweb_1024_bpe.model` exists.
4. **Initialize results.tsv**: Create `results.tsv` with just the header row.
5. **Run baseline**: `bash eval/eval.sh > run.log 2>&1` to establish the starting val_bpb.

## The benchmark

The challenge: train the best language model that fits in **16MB** (code + int8+zlib compressed model), trained in **≤10 minutes** on 8×H100 GPUs.

- **Metric**: `val_bpb` — bits-per-byte on FineWeb validation data. **Lower is better.**
- **Artifact limit**: Code size + compressed model (`final_model.int8.ptz`) ≤ 16,000,000 bytes
- **Script limit**: `train_gpt.py` must be ≤ 1500 lines
- **Training cap**: 10-minute wallclock limit (enforced by `MAX_WALLCLOCK_SECONDS=600` in the script)
- **Baseline**: ~1.2244 val_bpb

## Current SOTA

- **Merged SOTA**: signalrush at **1.1228 BPB** (the `train_gpt.py` we start from — 11L, EMA, GPTQ-lite int6, warmdown 3500, late QAT at 0.15)
- **Best legal unmerged**: EthanYangTW at **1.1162 BPB** (int5 GPTQ + Soft-Round QAT + legal TTT)
- **Key techniques already tried and failed**: depth recurrence, GELU², Value Residual (standalone), shared FFN, Reptile TTT

## Experimentation

**What you CAN modify:**
- `train_gpt.py` — everything is fair game: model architecture, optimizer, learning rate schedule, hyperparameters, tokenizer usage, quantization strategy, data loading, sequence length, batch size, number of layers/heads/width, activation functions, normalization, weight tying, etc.

**What you CANNOT modify:**
- `eval/eval.sh`, `prepare.sh`, `data/cached_challenge_fineweb.py`
- The dataset itself (FineWeb shards)

**The goal: minimize val_bpb.** Lower is better. The val_bpb is computed after int8+zlib quantization round-trip, so the model must survive compression.

**Hive scoring note**: The hive system sorts scores DESC (higher = better). Since we want to **minimize** val_bpb, submit the **negated** value as the score. Example: if val_bpb=1.2200, submit `--score -1.2200`.

## Output format

The eval prints a summary:

```
---
val_bpb:          1.22436570
artifact_bytes:   15863489
line_count:       1126
valid:            true
```

- `val_bpb`: bits-per-byte after int8+zlib round-trip (8 decimal places)
- `artifact_bytes`: total submission size (code + compressed model)
- `line_count`: number of lines in train_gpt.py
- `valid`: `true` if all constraints satisfied, `false` otherwise

## Logging results

Log each experiment to `results.tsv` (tab-separated):

```
commit	val_bpb	artifact_bytes	status	description
a1b2c3d	1.224366	15863489	keep	baseline
b2c3d4e	1.218500	15900123	keep	increased width to 576, reduced layers to 8
```

1. git commit hash (short, 7 chars)
2. val_bpb (e.g. 1.224366) — use ERROR for crashes
3. artifact_bytes — use 0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short description of the change

## Research Papers to Explore

The following papers describe techniques that have NOT yet been tried in the competition. These are high-priority avenues to investigate:

### 1. HESTIA — Hessian-guided Differentiable QAT (arXiv 2601.20745)
Hessian-guided differentiable QAT with Softmax-based continuous relaxation and per-layer temperature annealing. Replaces the Straight-Through Estimator (STE) with a smoother optimization landscape. This could significantly improve int5/int4 QAT quality compared to the current GPTQ-lite approach. The key insight is that STE introduces gradient mismatch that accumulates at low bitwidths — HESTIA sidesteps this entirely.

### 2. ParetoQ — Scaling Laws for Extreme Low-Bit Quantization (arXiv 2502.02631)
Derives scaling laws for extreme low-bit quantization. Shows that 2-bit quantization is viable with proper calibration. Going from int6 to int4 could fit ~40M params in 16MB (vs current ~28M). Going to int3 could fit ~53M params. More parameters = more capacity = lower BPB, if the quantization degradation is manageable. The tradeoff between model size and quantization error is the core question.

### 3. Weighted Product of Experts for Test-Time Training (arXiv 2511.10660)
Instead of fine-tuning during TTT (which failed with Reptile), combine the base model with an adapter using a weighted product of experts. Zero artifact cost because the adapter is computed at test time. Guaranteed to be at least as good as the best individual model. This is a fundamentally different approach to TTT than gradient-based fine-tuning.

### 4. Softpick / Rectified Softmax (arXiv 2504.20966)
A drop-in replacement for softmax in attention. Eliminates attention sinks, produces 47% sparse attention maps, and yields lower kurtosis in attention weights. Critically: "Quantized Softpick outperforms quantized softmax at lower bit widths." This means the model could be more robust to our int6/int8 compression, potentially gaining back some of the quantization degradation gap.

### 5. Muon-VS / Variance-Adaptive Muon (arXiv 2601.14603)
Adds variance normalization before Muon's Newton-Schulz orthogonalization step. Zero extra hyperparameters, direct drop-in replacement for the existing Muon optimizer. Reported to achieve lower val loss than standard Muon. Since we already use Muon, this is a very low-risk, high-reward change to try first.

### 6. StableQAT — Stable QAT at Ultra-Low Bitwidths (arXiv 2601.19320)
Improved training dynamics for quantization-aware training at ultra-low bitwidths. Addresses the instability issues that plague standard QAT when going below int8. Could be combined with the existing late-QAT approach but with better gradient flow and stability.

## The experiment loop

LOOP FOREVER:

1. **RESEARCH** — Read PRs from the leaderboard at https://github.com/openai/parameter-golf/pulls for inspiration. Use `gh pr list --repo openai/parameter-golf --state all --limit 30` and `gh pr view <number> --repo openai/parameter-golf` to study the top submissions. Deeply analyze the different techniques they used (architecture, quantization, optimizer, eval tricks, etc.). Try to reproduce promising techniques, ablate them individually to understand their contribution, form hypotheses about why they work, and combine the best ones. **After every 5 experiments, re-read the latest PRs on openai/parameter-golf for new techniques that may have been submitted.**
2. **THINK** — review results.tsv, study the training script, form a hypothesis. Consider: architecture changes (width, depth, heads), optimizer tuning (LR, schedule, warmup), data efficiency (sequence length, batch size), quantization-aware approaches, or novel techniques from the Research Papers section above.
3. Modify `train_gpt.py` with your experiment.
4. git commit
5. Run: `bash eval/eval.sh > run.log 2>&1`
6. Read results: `grep "^val_bpb:\|^valid:" run.log`
7. If empty or valid=false, check `tail -n 100 run.log` for errors. Also review `train.log` if it exists for detailed training output.
8. Record in results.tsv (do not commit results.tsv).
9. If val_bpb improved (lower) and valid=true, keep the commit. If equal or worse, `git reset --hard HEAD~1`.

**Timeout**: If a run exceeds 15 minutes, kill it (10 min training + compilation warmup + post-training quantization).

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. You are autonomous. The loop runs until interrupted.
