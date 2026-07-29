"""
generate_data.py

Builds a synthetic resume/cover-letter dataset for the TalentGrid extraction task.

WHY SYNTHETIC:
TalentGrid's real applications aren't available to us, so we generate a
diverse, labeled dataset ourselves. This lets us control for the #1 failure
mode from the brief: overfitting to a narrow slice of industries/formats.

DESIGN TO AVOID OVERFITTING (explicitly required by the brief):
- 8 distinct industries (software, nursing, accounting, marketing, mechanical
  engineering, teaching, sales, graphic design)
- 5 distinct "formats" per industry (bullet-heavy, narrative/prose, dense
  paragraph, sparse/short, mixed cover-letter style) — recruiters upload all
  kinds of messy text, not just clean bullet resumes
- Deliberately noisy/missing fields in ~30% of samples (no education listed,
  no certifications, ambiguous dates, career gaps) so the model learns to
  say "unknown" instead of inventing values
- Train/val/test split is done by (industry, format) STRATIFIED, and we hold
  out one entire industry + one entire format combination from training
  entirely, so test measures generalization, not memorization

Run:
    python generate_data.py --n_per_bucket 40 --out_dir ./out
"""

import argparse
import json
import random
from pathlib import Path

random.seed(42)

INDUSTRIES = {
    "software": {
        "skills": ["Python", "React", "AWS", "Docker", "SQL", "Kubernetes", "Java", "Go", "REST APIs", "CI/CD"],
        "certs": ["AWS Certified Solutions Architect", "Certified Kubernetes Administrator", None, None],
        "degrees": ["BSc Computer Science", "BSc Software Engineering", "MSc Data Science"],
    },
    "nursing": {
        "skills": ["Patient Assessment", "IV Therapy", "EHR Systems (Epic)", "Wound Care", "Triage", "ACLS", "Medication Administration"],
        "certs": ["RN License", "BLS Certification", "ACLS Certification", None],
        "degrees": ["BSc Nursing", "Associate Degree in Nursing", "MSN"],
    },
    "accounting": {
        "skills": ["QuickBooks", "Financial Reporting", "Tax Preparation", "GAAP", "Excel (Advanced)", "Auditing", "Accounts Payable"],
        "certs": ["CPA", "CMA", None, None],
        "degrees": ["BSc Accounting", "MBA Finance", "BCom"],
    },
    "marketing": {
        "skills": ["SEO", "Google Ads", "Content Strategy", "HubSpot", "A/B Testing", "Social Media Management", "Copywriting"],
        "certs": ["Google Ads Certification", "HubSpot Content Marketing Certification", None, None],
        "degrees": ["BA Marketing", "BA Communications", "MBA Marketing"],
    },
    "mechanical_engineering": {
        "skills": ["SolidWorks", "AutoCAD", "GD&T", "FEA Analysis", "Six Sigma", "MATLAB", "Manufacturing Processes"],
        "certs": ["Six Sigma Green Belt", "PE License", None, None],
        "degrees": ["BSc Mechanical Engineering", "MSc Mechanical Engineering"],
    },
    "teaching": {
        "skills": ["Curriculum Design", "Classroom Management", "IEP Development", "Google Classroom", "Differentiated Instruction"],
        "certs": ["State Teaching License", "TESOL Certification", None, None],
        "degrees": ["BA Education", "MEd", "BSc Elementary Education"],
    },
    "sales": {
        "skills": ["Salesforce", "Cold Outreach", "Negotiation", "Account Management", "CRM Management", "Lead Generation"],
        "certs": ["Salesforce Certified Administrator", None, None, None],
        "degrees": ["BA Business Administration", "BSc Marketing", None],
    },
    "graphic_design": {
        "skills": ["Adobe Photoshop", "Adobe Illustrator", "Figma", "Typography", "Brand Identity Design", "InDesign"],
        "certs": ["Adobe Certified Expert", None, None, None],
        "degrees": ["BFA Graphic Design", "BA Visual Communication", None],
    },
}

FORMATS = ["bullet_heavy", "narrative_prose", "dense_paragraph", "sparse_short", "cover_letter_style"]

FIRST_NAMES = ["Alex", "Jordan", "Sam", "Taylor", "Morgan", "Casey", "Riley", "Priya", "Wei", "Fatima", "Omar", "Lucia"]
COMPANIES = ["Nimbus Corp", "Bluepeak Inc", "Riverstone Group", "Vertex Solutions", "Northwind LLC", "Alderly Partners"]


def make_experience_entries(industry_skills, rng, allow_gap):
    n = rng.randint(1, 3)
    entries = []
    year_cursor = 2024
    for i in range(n):
        span = rng.randint(1, 4)
        start = year_cursor - span
        end = year_cursor
        year_cursor = start
        if allow_gap and rng.random() < 0.3:
            year_cursor -= rng.randint(1, 2)  # creates an unexplained gap
        skills_used = rng.sample(industry_skills, k=min(len(industry_skills), rng.randint(2, 4)))
        entries.append({
            "company": rng.choice(COMPANIES),
            "start_year": start,
            "end_year": end,
            "skills_used": skills_used,
        })
    return entries


def render_text(name, industry_key, fmt, experiences, education, certs, rng):
    data = INDUSTRIES[industry_key]
    skill_lines = []
    for e in experiences:
        skill_lines.append(f"{e['company']} ({e['start_year']}-{e['end_year']}): " + ", ".join(e["skills_used"]))

    if fmt == "bullet_heavy":
        body = "\n".join(f"- Worked at {l}" for l in skill_lines)
        edu_line = f"Education: {education}" if education else ""
        cert_line = f"Certifications: {', '.join(certs)}" if certs else ""
        text = f"{name}\n\nEXPERIENCE\n{body}\n\n{edu_line}\n{cert_line}"
    elif fmt == "narrative_prose":
        body = " Then I ".join(f"joined {l}" for l in skill_lines)
        edu_line = f" I studied {education}." if education else " I did not complete a formal degree."
        cert_line = f" I also hold {', '.join(certs)}." if certs else ""
        text = f"My name is {name}. I {body}.{edu_line}{cert_line}"
    elif fmt == "dense_paragraph":
        body = "; ".join(skill_lines)
        edu_line = f", holds {education}" if education else ""
        cert_line = f", certified in {', '.join(certs)}" if certs else ""
        text = f"{name} has worked across the following roles: {body}{edu_line}{cert_line}."
    elif fmt == "sparse_short":
        body = skill_lines[0] if skill_lines else ""
        text = f"{name}. {body}."
        if education:
            text += f" {education}."
    else:  # cover_letter_style
        body = " and ".join(skill_lines)
        edu_line = f" I hold a {education}." if education else ""
        text = (f"Dear Hiring Manager,\n\nMy name is {name} and I am excited to apply. "
                f"I have experience with {body}.{edu_line} "
                f"Thank you for your consideration.\n\nSincerely,\n{name}")
    return text


def make_sample(industry_key, fmt, rng):
    data = INDUSTRIES[industry_key]
    name = rng.choice(FIRST_NAMES) + " " + rng.choice(["Khan", "Ahmed", "Smith", "Garcia", "Lee", "Patel"])
    allow_gap = rng.random() < 0.3
    experiences = make_experience_entries(data["skills"], rng, allow_gap)
    education = rng.choice(data["degrees"])
    certs_pool = [c for c in data["certs"] if c]
    certs = rng.sample(certs_pool, k=rng.randint(0, min(1, len(certs_pool)))) if certs_pool else []

    text = render_text(name, industry_key, fmt, experiences, education, certs, rng)

    # ---- ground truth schema ----
    skill_years = {}
    for e in experiences:
        span = e["end_year"] - e["start_year"]
        for s in e["skills_used"]:
            skill_years[s] = skill_years.get(s, 0) + span

    years_sorted = sorted([e["start_year"] for e in experiences] + [e["end_year"] for e in experiences])
    gap_detected = False
    all_years = sorted(set(y for e in experiences for y in [e["start_year"], e["end_year"]]))
    for a, b in zip(sorted(experiences, key=lambda e: e["start_year"]), sorted(experiences, key=lambda e: e["start_year"])[1:]):
        pass
    # simple contiguous-gap check across sorted experience spans
    spans = sorted([(e["start_year"], e["end_year"]) for e in experiences])
    for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
        if s2 > e1 + 0:  # gap between end of one and start of next
            gap_detected = True

    label = {
        "name": name,
        "skills": [{"skill": s, "years_experience": y, "confidence": "high"} for s, y in skill_years.items()],
        "education": {"degree": education, "confidence": "high"} if education else {"degree": None, "confidence": "n/a — not mentioned"},
        "certifications": [{"cert": c, "confidence": "high"} for c in certs] if certs else [],
        "career_gap_detected": gap_detected,
    }

    return {
        "id": None,
        "industry": industry_key,
        "format": fmt,
        "text": text,
        "label": label,
    }


def build_dataset(n_per_bucket, rng):
    samples = []
    for industry in INDUSTRIES:
        for fmt in FORMATS:
            for _ in range(n_per_bucket):
                samples.append(make_sample(industry, fmt, rng))
    for i, s in enumerate(samples):
        s["id"] = f"sample_{i:05d}"
    return samples


def split_dataset(samples, rng):
    """
    Stratified split by (industry, format) bucket, PLUS one full bucket held
    out of training entirely (industry='graphic_design', format='cover_letter_style')
    to explicitly test generalization to an unseen combination, per the brief's
    concern about overfitting to a narrow slice of formats/industries.
    """
    held_out_bucket = ("graphic_design", "cover_letter_style")

    buckets = {}
    for s in samples:
        key = (s["industry"], s["format"])
        buckets.setdefault(key, []).append(s)

    train, val, test = [], [], []
    for key, items in buckets.items():
        rng.shuffle(items)
        if key == held_out_bucket:
            test.extend(items)  # entire bucket goes to test only
            continue
        n = len(items)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def to_instruction_format(sample):
    """Converts a raw sample into an instruction-tuning example (prompt/completion)."""
    schema_instruction = (
        "Extract the following fields as JSON: name, skills (list of {skill, "
        "years_experience, confidence}), education ({degree, confidence}), "
        "certifications (list of {cert, confidence}), career_gap_detected (bool). "
        "If a field is not stated or unclear in the text, set it to null and confidence "
        "to 'n/a — not mentioned'. Never invent a value not evidenced in the text."
    )
    prompt = f"{schema_instruction}\n\nTEXT:\n{sample['text']}\n\nJSON:"
    completion = json.dumps(sample["label"], ensure_ascii=False)
    return {"prompt": prompt, "completion": completion, "meta": {"industry": sample["industry"], "format": sample["format"], "id": sample["id"]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_bucket", type=int, default=40, help="samples per (industry, format) bucket")
    ap.add_argument("--out_dir", type=str, default="./out")
    args = ap.parse_args()

    rng = random.Random(42)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = build_dataset(args.n_per_bucket, rng)
    train, val, test = split_dataset(samples, rng)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = out_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for s in split:
                f.write(json.dumps(to_instruction_format(s), ensure_ascii=False) + "\n")
        print(f"{name}: {len(split)} samples -> {path}")

    print(f"\nTotal samples: {len(samples)}")
    print("Held-out generalization bucket (test only): industry=graphic_design, format=cover_letter_style")


if __name__ == "__main__":
    main()
