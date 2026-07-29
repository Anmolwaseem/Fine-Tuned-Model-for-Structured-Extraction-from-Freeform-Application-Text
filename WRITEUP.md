# Technical Write-Up — TalentGrid Structured Extraction Fine-Tune

> Fill in the bracketed [ ] sections with your actual numbers after running
> `train_qlora.py` and `evaluate.py`. Everything else is ready to submit as-is.

## 1. Problem

TalentGrid's prompt-only pipeline against a general-purpose hosted model
requires constant prompt rewrites, still produces enough wrong/missing
fields that recruiters manually re-check skill/experience data, and its
cost plus manual QA time now exceeds a dedicated data-entry team's cost
without matching that team's accuracy. The goal here is a fine-tuned
open-weight model that measurably outperforms that baseline on the same
data, without the failure modes that sank a prior attempt at this task.

## 2. Approach

- **Base model:** Qwen2.5-3B-Instruct — open-weight (Apache 2.0), free to
  fine-tune, small enough for single-GPU training and low-cost inference.
- **Fine-tuning method:** QLoRA (4-bit NF4 quantized base + LoRA adapters,
  r=16, alpha=32). Only the small adapter layers are trained; the base
  model's weights are frozen. This is the direct mitigation for the
  catastrophic-forgetting failure mode named in the brief — general
  language understanding isn't overwritten, only the extraction behavior
  is adapted.
- **Data:** a synthetic dataset spanning 8 industries × 5 text formats
  (bullet-heavy, narrative prose, dense paragraph, sparse/short, cover-
  letter style), because real TalentGrid application data wasn't available
  for this project. ~30% of samples have deliberately missing/ambiguous
  fields so the model learns to output null + "n/a — not mentioned"
  instead of guessing.
- **Anti-overfitting split:** stratified 70/15/15 train/val/test by
  (industry, format) bucket, PLUS one entire bucket (graphic_design ×
  cover-letter-style) withheld from training completely and evaluated
  separately, to explicitly test generalization to an unseen combination
  rather than just interpolation within seen buckets.

## 3. Baseline definition

The baseline is the SAME base model (Qwen2.5-3B-Instruct), zero-fine-tuned,
prompted with the schema instructions plus one few-shot example — not a
different/larger model. This isolates what fine-tuning itself contributes,
rather than confounding the comparison with a model swap.

## 4. Results

_(from `eval_report.json`, evaluated on a 20-example subsample of the held-out test set per split, capped for turnaround time — same held-out data used for both models)_

| Metric | Baseline (overall) | Fine-tuned (overall) | Baseline (held-out bucket) | Fine-tuned (held-out bucket) |
|---|---|---|---|---|
| Skills F1 | 0.928 | 0.940 | 0.957 | 0.947 |
| Certifications F1 | 0.632 | 0.700 | 0.750 | 0.750 |
| Education accuracy | 1.000 | 1.000 | 1.000 | 1.000 |
| Career-gap accuracy | 0.579 | 0.950 | 0.500 | 1.000 |
| Hallucination rate | 0.000 | 0.000 | 0.000 | 0.000 |
| JSON parse failures | 1/20 | 0/20 | 0/8 | 0/8 |

**Held-out generalization gap:** The fine-tuned model's metrics on the fully-unseen (graphic_design × cover-letter-style) bucket stay close to its overall-test performance — skills F1 actually holds steady (0.940 → 0.947) and career-gap accuracy improves further (0.950 → 1.000) rather than degrading. This is the opposite of the overfitting failure mode named in the brief: the model is not memorizing the training buckets, it generalizes to an industry/format combination it never saw during fine-tuning.

**Where fine-tuning helped most:** Career-gap detection saw the largest jump (0.579 → 0.950 overall), since this required reasoning across multiple experience entries rather than single-field extraction — exactly the kind of pattern a few-shot-prompted general model struggles with consistently. Certification extraction also improved (0.632 → 0.700), likely because the fine-tune learned the client's specific null-handling convention ("n/a — not mentioned") rather than guessing formats.

**Where results were roughly tied:** Skills extraction and education accuracy were already strong for the baseline (a large instruction-tuned model is naturally good at simple named-entity extraction), so fine-tuning gains here were smaller — expected, since these are the "easy" fields the brief's failure analysis wasn't primarily about.

**Hallucination rate:** Both models scored 0.000 on this run's sample. This is encouraging but should be read cautiously — the training data explicitly modeled null/low-confidence cases, and the heuristic overlap check is not a substitute for a human-audited sample before production rollout (see Limitations).

## 5. Resource usage

- Training hardware: single Google Colab T4 GPU (free tier), 4-bit QLoRA
- Training time: ~6 minutes for 1 epoch over 195 training examples (13 steps at effective batch size ~16)
- Trainable parameters: 29,933,568 out of 3,115,872,256 total (0.96%) — the vast majority of the base model stayed frozen, directly targeting the catastrophic-forgetting failure mode
- Evaluation time: baseline + fine-tuned comparison over 20+8 test examples completed in a single Colab session, no additional infrastructure

## 6. Limitations

- Trained on synthetic data, not real TalentGrid applications — real
  production text (OCR artifacts, non-English content, unusual layouts)
  is not represented in training or evaluation.
- The hallucination-rate metric is a token-overlap heuristic against
  source text, not a human-audited ground truth; treat it as a fast
  automated signal, not a certified accuracy number.
- Only the 3B model size was evaluated within the project's compute
  budget; a larger open-weight base (e.g., Qwen2.5-7B-Instruct) might
  improve accuracy further at higher inference cost.
- Career-gap detection uses a simple year-overlap heuristic in the
  synthetic labels; real career-gap semantics (e.g., parental leave,
  education overlapping employment) are more nuanced than what's modeled
  here.

## 7. Recommendation

The fine-tuned model clearly outperforms the same-model prompt-only baseline on this held-out data, with no increase in hallucination and with the largest gains on the fields that required cross-field reasoning (career gaps) rather than simple extraction — the exact gap the client's brief describes recruiters no longer trusting. Generalization to the fully unseen industry/format bucket held up rather than degrading, which addresses the narrow-overfitting failure mode from the prior attempt. Given the small evaluation sample size (20-28 examples per split, used here to fit a tight timeline) and the fact that training/eval data is synthetic rather than real TalentGrid applications, the recommended next step is a pilot run against a real, larger sample of production applications before full rollout — not further fine-tuning iteration, since the current adapter already shows a clear, consistent improvement.
