# TalentGrid — Fine-Tuned Structured Extraction

Fine-tunes an open-weight model (Qwen2.5-3B-Instruct) with QLoRA to extract
a fixed schema (skills, experience-per-skill, education, certifications,
career gaps) from freeform resume/cover-letter text — and demonstrates it
against a prompt-only baseline in a Streamlit app.

## Project layout

```
talentgrid/
├── data/
│   └── generate_data.py      # builds the synthetic train/val/test dataset
├── scripts/
│   ├── train_qlora.py        # QLoRA fine-tuning
│   └── evaluate.py           # baseline vs fine-tuned comparison + metrics
├── app/
│   └── app.py                # Streamlit side-by-side demo
├── requirements.txt
└── README.md
```

## Why this approach (maps to the client brief)

| Brief requirement | How this addresses it |
|---|---|
| Open-weight / freely fine-tunable base model | Qwen2.5-3B-Instruct (Apache 2.0) |
| Fits a free-tier / single consumer GPU | 4-bit QLoRA — trains on a single T4 (Colab free tier) |
| No real baseline comparison (prior failure) | `evaluate.py` runs the SAME base model zero/few-shot-prompted as the baseline, on the same held-out test set |
| Overfitting to narrow industry/format sample (prior failure) | Dataset spans 8 industries × 5 text formats; one entire (industry, format) bucket is withheld from training and evaluated separately as a generalization check |
| Schema drift / invented values (prior failure) | Training data explicitly teaches null + `"n/a — not mentioned"` for absent fields; `evaluate.py` computes a hallucination rate |
| Catastrophic forgetting (prior failure) | QLoRA freezes base weights; only small LoRA adapters (r=16) are trained, few epochs, early stopping on val loss |
| Streamlit app, side-by-side, with confidence | `app/app.py` |

## Setup (from a clean environment)

```bash
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Requires a CUDA GPU with ~8GB+ VRAM for the 3B model in 4-bit (a free Colab
T4 works). CPU-only will run but is very slow — fine for testing the data
generation script, not for training/inference at scale.

## Step 1 — generate the dataset

```bash
cd data
python generate_data.py --n_per_bucket 40 --out_dir ./out
```

Produces `out/train.jsonl`, `out/val.jsonl`, `out/test.jsonl`. Increase
`--n_per_bucket` for a larger dataset if time allows.

## Step 2 — fine-tune

```bash
cd ../scripts
python train_qlora.py \
    --train_file ../data/out/train.jsonl \
    --val_file ../data/out/val.jsonl \
    --output_dir ./qlora-adapter \
    --epochs 2
```

Hyperparameters (documented, also written to `qlora-adapter/run_info.json`
after training):
- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Method: QLoRA, 4-bit NF4 quantization, LoRA r=16, alpha=32, dropout=0.05
- LoRA target modules: q/k/v/o/gate/up/down projections
- Epochs: 2 (kept low deliberately — anti-overfitting/forgetting)
- LR: 2e-4, effective batch size: batch_size × grad_accum
- Max sequence length: 1024
- Model selection: best checkpoint by `eval_loss` on the val set

## Step 3 — evaluate against baseline

```bash
python evaluate.py \
    --test_file ../data/out/test.jsonl \
    --adapter_dir ./qlora-adapter \
    --out_report ./eval_report.json
```

Reports, for both baseline and fine-tuned, on both the general test split
and the held-out generalization bucket:
- Skills precision / recall / F1
- Certifications precision / recall / F1
- Education exact-match accuracy
- Career-gap-detection accuracy
- Hallucination rate (fraction of non-null predicted fields with no textual
  evidence in the source)
- JSON parse failure rate

## Step 4 — run the Streamlit demo

```bash
cd ../app
streamlit run app.py -- --adapter_dir ../scripts/qlora-adapter
```

Paste or upload resume text; the app shows the baseline and fine-tuned
extractions side by side with confidence badges (🟢 high / 🟡 medium /
🟠 low / 🔘 not mentioned).

## Known limitations

- The dataset is synthetically generated (no access to TalentGrid's real
  applications), so real-world messiness — typos, OCR artifacts from
  scanned resumes, non-English text — is not represented. Real deployment
  would need a sample of true production text before rollout.
- The hallucination check is a heuristic (token-overlap against source
  text), not a human-verified audit — good for a fast automated signal,
  but a human QA pass on a sample is still recommended before production use.
- Only one base model size was tuned within the project's time/compute
  budget; a 7B base model would likely improve accuracy further at the
  cost of slower inference.
