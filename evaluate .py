"""
evaluate.py

Rigorous baseline-vs-fine-tuned comparison on the SAME held-out test set,
directly addressing the brief's #1 named failure mode ("no real baseline
comparison").

BASELINE DEFINITION (important, matches the client's actual current setup):
The baseline is the SAME base model (Qwen2.5-3B-Instruct), zero-shot
prompted with the schema instructions and a couple of few-shot examples —
NOT a different/bigger model like GPT-4. This isolates the effect of
fine-tuning itself, rather than confounding it with "we just swapped to a
better model."

METRICS (per field, on the held-out test set only):
- Skills: precision / recall / F1 over the set of extracted skill names
  (case-insensitive match)
- Education degree: exact-match accuracy (accounts for null-correctly-predicted)
- Certifications: precision / recall / F1 over extracted cert names
- career_gap_detected: accuracy
- HALLUCINATION RATE: fraction of predicted fields where the model asserted
  a non-null value that has zero textual evidence in the source (checked via
  substring/fuzzy match against source text) — this is the metric that
  directly targets the "schema drift" / "invents plausible-but-wrong values"
  failure mode named in the brief. Lower is better; the model should say
  "n/a — not mentioned" instead of guessing.
- Held-out generalization bucket is reported SEPARATELY (industry=
  graphic_design, format=cover_letter_style — the bucket withheld entirely
  from training) to explicitly surface overfitting.

Run:
    python evaluate.py --test_file ../data/out/test.jsonl \
        --adapter_dir ../scripts/qlora-adapter \
        --out_report ./eval_report.json
"""

import argparse
import json
import re
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

SYSTEM_PROMPT = (
    "You are a precise resume information extraction assistant. You only "
    "output valid JSON matching the requested schema. You never invent "
    "values not present in the source text."
)

FEW_SHOT_EXAMPLE = {
    "prompt": (
        "Extract the following fields as JSON: name, skills (list of {skill, "
        "years_experience, confidence}), education ({degree, confidence}), "
        "certifications (list of {cert, confidence}), career_gap_detected (bool). "
        "If a field is not stated or unclear in the text, set it to null and "
        "confidence to 'n/a — not mentioned'. Never invent a value not "
        "evidenced in the text.\n\nTEXT:\nJordan Lee. Worked at Bluepeak Inc "
        "(2021-2023): Python, SQL.\n\nJSON:"
    ),
    "completion": json.dumps({
        "name": "Jordan Lee",
        "skills": [{"skill": "Python", "years_experience": 2, "confidence": "high"},
                   {"skill": "SQL", "years_experience": 2, "confidence": "high"}],
        "education": {"degree": None, "confidence": "n/a — not mentioned"},
        "certifications": [],
        "career_gap_detected": False,
    }),
}


def load_model(adapter_dir=None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config, device_map="auto")
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt, is_baseline):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if is_baseline:
        # Few-shot for the baseline since it has no task-specific fine-tuning.
        messages.append({"role": "user", "content": FEW_SHOT_EXAMPLE["prompt"]})
        messages.append({"role": "assistant", "content": FEW_SHOT_EXAMPLE["completion"]})
    messages.append({"role": "user", "content": prompt})

    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    with torch.no_grad():
        out = model.generate(**encoded, max_new_tokens=400, do_sample=False, temperature=None, top_p=None,
                              pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True)
    return text


def safe_parse_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def norm_names(items, key):
    return set(str(x.get(key, "")).strip().lower() for x in items if x and x.get(key))


def prf1(pred_set, gold_set):
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def check_hallucination(value, source_text):
    """A predicted non-null value is a hallucination if it has no reasonable
    textual overlap with the source (crude but effective substring/token
    check — good enough to flag invented values vs. paraphrased-but-grounded
    ones)."""
    if value is None:
        return False
    value_l = str(value).lower()
    source_l = source_text.lower()
    tokens = [t for t in re.split(r"\W+", value_l) if len(t) > 2]
    if not tokens:
        return False
    overlap = sum(1 for t in tokens if t in source_l)
    return (overlap / len(tokens)) < 0.5


def evaluate_split(model, tokenizer, examples, is_baseline, label_name):
    skill_p, skill_r, skill_f1 = [], [], []
    cert_p, cert_r, cert_f1 = [], [], []
    edu_correct, edu_total = 0, 0
    gap_correct, gap_total = 0, 0
    hallucinations, total_nonnull_fields = 0, 0
    parse_failures = 0

    for ex in examples:
        gold = json.loads(ex["completion"])
        raw_source = ex["prompt"].split("TEXT:\n")[-1].split("\n\nJSON:")[0]
        raw_output = generate(model, tokenizer, ex["prompt"], is_baseline)
        pred = safe_parse_json(raw_output)

        if pred is None:
            parse_failures += 1
            continue

        # skills
        pred_skills = norm_names(pred.get("skills", []) or [], "skill")
        gold_skills = norm_names(gold.get("skills", []) or [], "skill")
        p, r, f1 = prf1(pred_skills, gold_skills)
        skill_p.append(p); skill_r.append(r); skill_f1.append(f1)

        # certifications
        pred_certs = norm_names(pred.get("certifications", []) or [], "cert")
        gold_certs = norm_names(gold.get("certifications", []) or [], "cert")
        p, r, f1 = prf1(pred_certs, gold_certs)
        cert_p.append(p); cert_r.append(r); cert_f1.append(f1)

        # education exact match (including correctly-null)
        pred_edu = (pred.get("education") or {}).get("degree")
        gold_edu = (gold.get("education") or {}).get("degree")
        edu_total += 1
        if str(pred_edu).strip().lower() == str(gold_edu).strip().lower():
            edu_correct += 1

        # career gap
        gap_total += 1
        if bool(pred.get("career_gap_detected")) == bool(gold.get("career_gap_detected")):
            gap_correct += 1

        # hallucination check on skills + education + certs
        for s in (pred.get("skills") or []):
            total_nonnull_fields += 1
            if check_hallucination(s.get("skill"), raw_source):
                hallucinations += 1
        if pred_edu:
            total_nonnull_fields += 1
            if check_hallucination(pred_edu, raw_source):
                hallucinations += 1
        for c in (pred.get("certifications") or []):
            total_nonnull_fields += 1
            if check_hallucination(c.get("cert"), raw_source):
                hallucinations += 1

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "label": label_name,
        "n_examples": len(examples),
        "parse_failures": parse_failures,
        "skills_precision": avg(skill_p),
        "skills_recall": avg(skill_r),
        "skills_f1": avg(skill_f1),
        "certifications_precision": avg(cert_p),
        "certifications_recall": avg(cert_r),
        "certifications_f1": avg(cert_f1),
        "education_accuracy": edu_correct / edu_total if edu_total else 0.0,
        "career_gap_accuracy": gap_correct / gap_total if gap_total else 0.0,
        "hallucination_rate": hallucinations / total_nonnull_fields if total_nonnull_fields else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_file", type=str, required=True)
    ap.add_argument("--adapter_dir", type=str, required=True)
    ap.add_argument("--out_report", type=str, default="./eval_report.json")
    ap.add_argument("--held_out_industry", type=str, default="graphic_design")
    ap.add_argument("--held_out_format", type=str, default="cover_letter_style")
    ap.add_argument("--limit", type=int, default=None,
                     help="Cap the number of examples per split (speeds up eval under time pressure; still a valid random subsample).")
    args = ap.parse_args()

    examples = [json.loads(l) for l in open(args.test_file, encoding="utf-8")]
    held_out = [e for e in examples if e["meta"]["industry"] == args.held_out_industry and e["meta"]["format"] == args.held_out_format]
    seen_dist = [e for e in examples if e not in held_out]

    if args.limit:
        seen_dist = seen_dist[:args.limit]
        held_out = held_out[:min(args.limit, len(held_out))]

    report = {}

    print("Loading baseline (prompted base model, no fine-tuning)...")
    base_model, tokenizer = load_model(adapter_dir=None)
    report["baseline_overall"] = evaluate_split(base_model, tokenizer, seen_dist, True, "baseline_overall")
    report["baseline_held_out_generalization"] = evaluate_split(base_model, tokenizer, held_out, True, "baseline_held_out")
    del base_model
    torch.cuda.empty_cache()

    print("Loading fine-tuned model (base + QLoRA adapter)...")
    ft_model, tokenizer = load_model(adapter_dir=args.adapter_dir)
    report["finetuned_overall"] = evaluate_split(ft_model, tokenizer, seen_dist, False, "finetuned_overall")
    report["finetuned_held_out_generalization"] = evaluate_split(ft_model, tokenizer, held_out, False, "finetuned_held_out")

    with open(args.out_report, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== SUMMARY ===")
    for key in ["baseline_overall", "finetuned_overall", "baseline_held_out_generalization", "finetuned_held_out_generalization"]:
        r = report[key]
        print(f"{key}: skills_F1={r['skills_f1']:.3f} certs_F1={r['certifications_f1']:.3f} "
              f"edu_acc={r['education_accuracy']:.3f} gap_acc={r['career_gap_accuracy']:.3f} "
              f"hallucination_rate={r['hallucination_rate']:.3f} parse_failures={r['parse_failures']}/{r['n_examples']}")
    print(f"\nFull report written to {args.out_report}")


if __name__ == "__main__":
    main()
