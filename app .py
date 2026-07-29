"""
app.py — Streamlit demo for TalentGrid resume extraction.

Lets a user paste/upload resume text and see the CURRENT baseline
(prompted general-purpose-style pipeline, same base model zero/few-shot)
and the FINE-TUNED model's extractions side by side, with per-field
confidence indicators — directly matching the deliverable requirement.

Run:
    streamlit run app.py -- --adapter_dir ../scripts/qlora-adapter
"""

import json
import re
import sys

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
SYSTEM_PROMPT = (
    "You are a precise resume information extraction assistant. You only "
    "output valid JSON matching the requested schema. You never invent "
    "values not present in the source text."
)
SCHEMA_INSTRUCTION = (
    "Extract the following fields as JSON: name, skills (list of {skill, "
    "years_experience, confidence}), education ({degree, confidence}), "
    "certifications (list of {cert, confidence}), career_gap_detected (bool). "
    "If a field is not stated or unclear in the text, set it to null and confidence "
    "to 'n/a — not mentioned'. Never invent a value not evidenced in the text."
)
FEW_SHOT_PROMPT = (
    SCHEMA_INSTRUCTION + "\n\nTEXT:\nJordan Lee. Worked at Bluepeak Inc "
    "(2021-2023): Python, SQL.\n\nJSON:"
)
FEW_SHOT_COMPLETION = json.dumps({
    "name": "Jordan Lee",
    "skills": [{"skill": "Python", "years_experience": 2, "confidence": "high"},
               {"skill": "SQL", "years_experience": 2, "confidence": "high"}],
    "education": {"degree": None, "confidence": "n/a — not mentioned"},
    "certifications": [],
    "career_gap_detected": False,
})


def parse_cli_adapter_dir(default="./qlora-adapter"):
    if "--adapter_dir" in sys.argv:
        idx = sys.argv.index("--adapter_dir")
        return sys.argv[idx + 1]
    return default


@st.cache_resource(show_spinner="Loading base model (baseline)...")
def load_baseline():
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                     bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config, device_map="auto")
    model.eval()
    return model, tok


@st.cache_resource(show_spinner="Loading fine-tuned model...")
def load_finetuned(adapter_dir):
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                     bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.pad_token or tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config, device_map="auto")
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tok


def run_extraction(model, tokenizer, resume_text, is_baseline):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if is_baseline:
        messages.append({"role": "user", "content": FEW_SHOT_PROMPT})
        messages.append({"role": "assistant", "content": FEW_SHOT_COMPLETION})
    user_prompt = f"{SCHEMA_INSTRUCTION}\n\nTEXT:\n{resume_text}\n\nJSON:"
    messages.append({"role": "user", "content": user_prompt})

    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    with torch.no_grad():
        out = model.generate(**encoded, max_new_tokens=400, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.decode(out[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True)
    match = re.search(r"\{.*\}", decoded, re.DOTALL)
    if not match:
        return None, decoded
    try:
        return json.loads(match.group(0)), decoded
    except json.JSONDecodeError:
        return None, decoded


def confidence_badge(conf):
    if not conf or "n/a" in str(conf).lower():
        return "🔘 not mentioned"
    conf_l = str(conf).lower()
    if "high" in conf_l:
        return "🟢 high"
    if "med" in conf_l:
        return "🟡 medium"
    if "low" in conf_l:
        return "🟠 low"
    return f"⚪ {conf}"


def render_extraction(container, result, raw_output, label):
    container.markdown(f"#### {label}")
    if result is None:
        container.error("Model did not return valid JSON.")
        container.code(raw_output)
        return

    container.markdown(f"**Name:** {result.get('name') or '_not detected_'}")

    container.markdown("**Skills**")
    skills = result.get("skills") or []
    if not skills:
        container.caption("No skills extracted.")
    for s in skills:
        container.markdown(f"- {s.get('skill')} — {s.get('years_experience', '?')} yrs — {confidence_badge(s.get('confidence'))}")

    edu = result.get("education") or {}
    container.markdown("**Education**")
    container.markdown(f"- {edu.get('degree') or '_not mentioned_'} — {confidence_badge(edu.get('confidence'))}")

    container.markdown("**Certifications**")
    certs = result.get("certifications") or []
    if not certs:
        container.caption("None extracted.")
    for c in certs:
        container.markdown(f"- {c.get('cert')} — {confidence_badge(c.get('confidence'))}")

    container.markdown("**Career Gap Detected**")
    container.markdown("⚠️ Yes" if result.get("career_gap_detected") else "✅ No")


def main():
    st.set_page_config(page_title="TalentGrid Extraction Comparison", layout="wide")
    st.title("TalentGrid: Baseline vs Fine-Tuned Extraction")
    st.caption("Paste resume/cover-letter text below to compare the current prompt-only pipeline "
               "against the fine-tuned model, side by side, with confidence indicators.")

    adapter_dir = parse_cli_adapter_dir()

    default_text = ("Dear Hiring Manager,\n\nMy name is Alex Khan and I am excited to apply. "
                    "I have experience with Nimbus Corp (2021-2023): Python, AWS, Docker. "
                    "I hold a BSc Computer Science. Thank you for your consideration.\n\nSincerely,\nAlex Khan")
    resume_text = st.text_area("Resume / cover letter text", value=default_text, height=220)
    uploaded = st.file_uploader("...or upload a .txt file", type=["txt"])
    if uploaded:
        resume_text = uploaded.read().decode("utf-8")

    if st.button("Run extraction", type="primary"):
        with st.spinner("Loading models and running extraction (first run is slower)..."):
            base_model, base_tok = load_baseline()
            baseline_result, baseline_raw = run_extraction(base_model, base_tok, resume_text, is_baseline=True)

            ft_model, ft_tok = load_finetuned(adapter_dir)
            ft_result, ft_raw = run_extraction(ft_model, ft_tok, resume_text, is_baseline=False)

        col1, col2 = st.columns(2)
        render_extraction(col1, baseline_result, baseline_raw, "Current Baseline (prompt-only)")
        render_extraction(col2, ft_result, ft_raw, "Fine-Tuned Model")

        with st.expander("Raw model outputs"):
            st.code(baseline_raw, language="json")
            st.code(ft_raw, language="json")

    st.divider()
    st.caption("Confidence legend: 🟢 high · 🟡 medium · 🟠 low · 🔘 not mentioned in source text")


if __name__ == "__main__":
    main()
