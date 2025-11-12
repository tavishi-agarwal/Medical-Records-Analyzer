import re
import io
import os
import tempfile
from typing import Dict, Optional, Tuple

import streamlit as st
import pandas as pd
import pdfplumber
from PIL import Image

# Optional imports: pytesseract, pdf2image, openai, matplotlib
try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    from pdf2image import convert_from_bytes
except Exception:
    convert_from_bytes = None

try:
    import openai
except Exception:
    openai = None

import matplotlib.pyplot as plt

# -----------------------------
# Helper functions
# -----------------------------

TEST_PATTERNS = {
    # pattern: (human-friendly-name, unit-if-detected)
    r"\bTotal\s*Cholesterol[:\s]*([0-9]+\.?[0-9]*)\s*(mg/dL|mmol/L)?": ("Cholesterol (Total)", None),
    r"\bCholesterol[:\s]*([0-9]+\.?[0-9]*)\s*(mg/dL|mmol/L)?": ("Cholesterol (Total)", None),
    r"\bHDL[:\s]*([0-9]+\.?[0-9]*)\s*(mg/dL|mmol/L)?": ("HDL Cholesterol", None),
    r"\bLDL[:\s]*([0-9]+\.?[0-9]*)\s*(mg/dL|mmol/L)?": ("LDL Cholesterol", None),
    r"\bTriglycerides[:\s]*([0-9]+\.?[0-9]*)\s*(mg/dL|mmol/L)?": ("Triglycerides", None),
    r"\bGlucose[:\s]*([0-9]+\.?[0-9]*)\s*(mg/dL|mmol/L)?": ("Glucose (Fasting)", None),
    r"\bFasting\s*Glucose[:\s]*([0-9]+\.?[0-9]*)\s*(mg/dL|mmol/L)?": ("Glucose (Fasting)", None),
    r"\bHbA1c[:\s]*([0-9]+\.?[0-9]*)\s*(%|mmol/mol)?": ("HbA1c", None),
    r"\bHemoglobin[:\s]*([0-9]+\.?[0-9]*)\s*(g/dL)?": ("Hemoglobin", None),
    r"\bWBC[:\s]*([0-9]+\.?[0-9]*)\s*(10\^9/L|x10\^9/L)?": ("WBC", None),
    r"\bPlatelet[s]?:[:\s]*([0-9]+\.?[0-9]*)\s*(10\^9/L|x10\^9/L)?": ("Platelets", None),
    r"\bAST[:\s]*([0-9]+\.?[0-9]*)\s*(U/L)?": ("AST (SGOT)", None),
    r"\bALT[:\s]*([0-9]+\.?[0-9]*)\s*(U/L)?": ("ALT (SGPT)", None),
    r"\bCreatinine[:\s]*([0-9]+\.?[0-9]*)\s*(mg/dL|umol/L)?": ("Creatinine", None),
    r"\beGFR[:\s]*([0-9]+\.?[0-9]*)\s*(mL/min/1.73m2)?": ("eGFR", None),
    r"\bTSH[:\s]*([0-9]+\.?[0-9]*)\s*(uIU/mL|mIU/L)?": ("TSH", None),
    r"\bVitamin\s*D[:\s]*([0-9]+\.?[0-9]*)\s*(ng/mL|nmol/L)?": ("Vitamin D", None),
}

# Some fallback generic pattern for lines like "HDL 55 mg/dL"
GENERIC_PATTERN = r"\b([A-Za-z%\s]{2,25})\s+([0-9]+\.?[0-9]*)\s*(mg/dL|mmol/L|%|U/L|ng/mL|mmol/L|g/dL|umol/L|mIU/L|10\^9/L|x10\^9/L)?\b"

NORMAL_RANGES = {
    "Cholesterol (Total)": (0, 200),  # mg/dL
    "HDL Cholesterol": (40, 100),
    "LDL Cholesterol": (0, 100),
    "Triglycerides": (0, 150),
    "Glucose (Fasting)": (70, 100),
    "HbA1c": (0, 5.6),
    "Hemoglobin": (12, 17.5),
    "WBC": (4.0, 11.0),
    "Platelets": (150, 450),
    "AST (SGOT)": (0, 40),
    "ALT (SGPT)": (0, 40),
    "Creatinine": (0.6, 1.3),
    "eGFR": (60, 200),
    "TSH": (0.4, 4.0),
    "Vitamin D": (20, 50),
}

# -----------------------------
# Text extraction
# -----------------------------

def extract_text_from_pdf_bytes(data: bytes) -> str:
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
    except Exception as e:
        st.exception(e)
    # If text is empty and pdf2image+pytesseract available, try OCR
    if (not text.strip()) and convert_from_bytes and pytesseract:
        st.info("No text found in PDF - trying OCR with pdf2image + pytesseract (may require poppler)")
        try:
            images = convert_from_bytes(data)
            for img in images:
                text += pytesseract.image_to_string(img) + "\n"
        except Exception as e:
            st.warning("pdf2image OCR failed: " + str(e))
    return text


def extract_text_from_image_file(image_bytes: bytes) -> str:
    if not pytesseract:
        raise RuntimeError("pytesseract not installed. Install pytesseract to enable OCR for images.")
    img = Image.open(io.BytesIO(image_bytes))
    return pytesseract.image_to_string(img)


# -----------------------------
# Parsing
# -----------------------------

def parse_tests_from_text(text: str) -> Dict[str, Dict[str, Optional[float]]]:
    results = {}

    # First apply specific patterns
    for pat, (name, _) in TEST_PATTERNS.items():
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            try:
                val = float(m.group(1))
            except Exception:
                continue
            results[name] = {"value": val, "unit": (m.group(2) if m.lastindex and m.lastindex >=2 else None)}

    # Generic fallback
    for m in re.finditer(GENERIC_PATTERN, text):
        label = m.group(1).strip()
        val = m.group(2)
        unit = m.group(3)
        # Normalize label somewhat (remove extra spaces)
        key = label.title()
        # Map some common shorthand to our canonical names
        mapping = {
            "Hdl": "HDL Cholesterol",
            "Ldl": "LDL Cholesterol",
            "Triglycerides": "Triglycerides",
            "Cholesterol": "Cholesterol (Total)",
            "HbA1c": "HbA1c",
            "Hemoglobin": "Hemoglobin",
            "Wbc": "WBC",
            "Platelets": "Platelets",
            "Ast": "AST (SGOT)",
            "Alt": "ALT (SGPT)",
            "Creatinine": "Creatinine",
            "Egfr": "eGFR",
            "Tsh": "TSH",
            "Vitamin D": "Vitamin D",
        }
        if key in mapping:
            results[mapping[key]] = {"value": float(val), "unit": unit}

    return results


# -----------------------------
# Insights generation (rule-based + LLM optional)
# -----------------------------

def simple_rule_insights(parsed: Dict[str, Dict[str, Optional[float]]]) -> Dict[str, str]:
    insights = {}
    for k, v in parsed.items():
        val = v.get("value")
        if val is None:
            continue
        if k in NORMAL_RANGES:
            lo, hi = NORMAL_RANGES[k]
            if val < lo:
                insights[k] = f"{k} is low ({val}). Consider evaluation for deficiency or other causes."
            elif val > hi:
                insights[k] = f"{k} is high ({val}). Lifestyle changes or clinical follow-up may be needed."
            else:
                insights[k] = f"{k} is within the reference range ({val})."
        else:
            insights[k] = f"{k}: {val}"
    return insights


def call_llm_for_insights(report_text: str, user_note: str = "") -> str:
    if not openai:
        return "OpenAI library not installed or OPENAI_API_KEY not configured. Install openai and set the key to use AI insights."
    api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None
    if not api_key:
        return "OpenAI API key not found. Set OPENAI_API_KEY environment variable or Streamlit secrets."
    openai.api_key = api_key
    prompt = (
        "You are a medical assistant (non-diagnostic). "
        "Given the following extracted lab report text, provide a clear, concise medical interpretation of the main abnormal findings and practical lifestyle/management suggestions. "
        "Be explicit about uncertainties and recommend clinical follow-up where appropriate.\n\n"
        f"Report text:\n{report_text}\n\nUser note: {user_note}\n\nRespond with bullet points."
    )
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant. Always advise clinical consultation for diagnoses."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=700,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"LLM call failed: {e}"


# -----------------------------
# Streamlit UI
# -----------------------------

def app():
    st.set_page_config(page_title="Smart Medical Report Analyzer", layout="wide")
    st.title("🩺 Smart Medical Report Analyzer")
    st.markdown(
        "Upload a medical report (PDF or image) and get extracted lab values, quick rule-based insights, and optional AI-generated recommendations.\n\n"
        "**Important:** This tool provides informational insights only and is NOT a medical diagnosis. Always consult a healthcare professional."
    )

    uploaded = st.file_uploader("Upload medical report (PDF, PNG, JPG)", type=["pdf", "png", "jpg", "jpeg"])
    user_note = st.text_input("Any context for the report? (e.g., fasting sample, patient age/sex)")

    if uploaded is not None:
        bytes_data = uploaded.read()
        text = ""
        if uploaded.type == "application/pdf" or uploaded.name.lower().endswith('.pdf'):
            text = extract_text_from_pdf_bytes(bytes_data)
        else:
            if pytesseract:
                text = extract_text_from_image_file(bytes_data)
            else:
                st.warning("Install pytesseract to enable OCR for images")

        if not text.strip():
            st.error("No text could be extracted from the uploaded file. Try a different file or install pdf2image/pytesseract for OCR.")
            return

        st.subheader("📄 Extracted Text")
        with st.expander("Show extracted text"):
            st.text_area("Report text", text, height=300)

        parsed = parse_tests_from_text(text)
        if not parsed:
            st.warning("No common lab values were parsed automatically. You can still ask the AI for insights or paste key values manually.")

        # Display parsed results in a dataframe
        df_rows = []
        for k, v in parsed.items():
            val = v.get('value')
            unit = v.get('unit') or ""
            lohi = NORMAL_RANGES.get(k)
            df_rows.append({"Test": k, "Value": val, "Unit": unit or "", "Reference": f"{lohi[0]} - {lohi[1]}" if lohi else ""})
        if df_rows:
            df = pd.DataFrame(df_rows)
            st.subheader("🔬 Parsed Lab Values")
            st.dataframe(df, width=800)

            # Simple insights
            insights = simple_rule_insights(parsed)
            st.subheader("💡 Quick Rule-Based Insights")
            for k, v in insights.items():
                if "high" in v.lower() or "low" in v.lower():
                    st.warning(f"{k}: {v}")
                else:
                    st.success(f"{k}: {v}")

            # Plot basic bar chart for numeric values
            st.subheader("📊 Lab Values Chart")
            numeric = {k: v['value'] for k, v in parsed.items() if isinstance(v.get('value'), (int, float))}
            if numeric:
                fig, ax = plt.subplots(figsize=(8, 4))
                names = list(numeric.keys())
                vals = [numeric[n] for n in names]
                ax.barh(names, vals)
                ax.set_xlabel('Value')
                st.pyplot(fig)

        # AI insights
        st.subheader("🤖 AI Insights (optional)")
        if openai:
            if st.button("Generate AI insights from extracted report"):
                with st.spinner("Calling AI..."):
                    ai_output = call_llm_for_insights(text, user_note)
                    st.markdown(ai_output)
        else:
            st.info("OpenAI SDK not available in this environment. Install openai to enable AI insights.")

        # Downloadable summary
        st.subheader("📥 Downloadable Summary")
        summary_lines = ["Smart Medical Report Analyzer - Summary\n"]
        if parsed:
            for k, v in parsed.items():
                val = v.get('value')
                unit = v.get('unit') or ""
                summary_lines.append(f"{k}: {val} {unit}\n")
        if 'insights' in locals():
            summary_lines.append("\nQuick insights:\n")
            for k, v in insights.items():
                summary_lines.append(f"- {k}: {v}\n")
        if openai and 'ai_output' in locals():
            summary_lines.append("\nAI Insights:\n")
            summary_lines.append(ai_output + "\n")

        summary_text = "\n".join(summary_lines)
        st.download_button("Download summary (TXT)", summary_text, file_name="report_summary.txt")

    st.sidebar.header("Setup & Notes")
    st.sidebar.markdown(
        "- This app extracts visible numeric values from uploaded reports using heuristics and regex.\n"
        "- OCR requires pytesseract; PDF OCR requires pdf2image (plus poppler).\n"
        "- To enable AI insights, set your OPENAI_API_KEY as an environment variable or in Streamlit secrets.\n"
        "- **Not a substitute for professional medical advice.**"
    )


if __name__ == "__main__":
    app()
