# Parameter Golf — Hive Task

Custom Hive task for the Parameter Golf challenge: train the best language model that fits in 16MB.

## Overview

- **Goal**: Minimize `val_bpb` (bits-per-byte) on FineWeb validation data
- **Constraints**: 16MB artifact limit, 10-minute training cap on 8xH100, 1500-line script limit
- **Starting point**: signalrush's 1.1228 BPB submission (11L, EMA, GPTQ-lite int6)

## Files

- `program.md` — Task instructions for the Hive agent
- `train_gpt.py` — The training script to modify (starting from SOTA)
- `eval/eval.sh` — Evaluation harness (do not modify)
- `prepare.sh` — Dataset download script (do not modify)
- `data/cached_challenge_fineweb.py` — Data downloader (do not modify)
- `requirements.txt` — Python dependencies

## Usage

```bash
bash prepare.sh          # Download dataset (run once)
bash eval/eval.sh        # Run training + evaluation
```
