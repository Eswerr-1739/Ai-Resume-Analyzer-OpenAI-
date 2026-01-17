import streamlit as st
from src.analyzer import compute_similarity, section_scoring, llm_skill_extraction, save_reports
from src.utils import extract_text_from_file
import os

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
st.title("📄 AI Resume / Job Description Analyzer")

# -------------------------------
st.sidebar.header("Upload Files")
resume_file = st.sidebar.file_uploader("Upload Resume (PDF, DOCX, TXT)", type=["pdf","docx","txt"])
jd_file = st.sidebar.file_uploader("Upload Job Description (PDF, DOCX, TXT)", type=["pdf","docx","txt"])

if st.sidebar.button("Analyze"):

    if resume_file is None or jd_file is None:
        st.warning("Please upload both Resume and Job Description files.")
    else:
        # Save files to temp path
        resume_path = os.path.join("data", resume_file.name)
        jd_path = os.path.join("data", jd_file.name)
        os.makedirs("data", exist_ok=True)
        with open(resume_path, "wb") as f:
            f.write(resume_file.getbuffer())
        with open(jd_path, "wb") as f:
            f.write(jd_file.getbuffer())

        # Extract text
        resume_text = extract_text_from_file(resume_path)
        jd_text = extract_text_from_file(jd_path)

        # Compute similarity
        similarity_score = compute_similarity(resume_text, jd_text)
        st.metric("Resume-JD Similarity", f"{similarity_score:.2f}%")

        # Section scoring
        section_scores = section_scoring(resume_text, jd_text)
        st.subheader("📊 Section Scores")
        st.dataframe(section_scores)

        # LLM Skill extraction
        with st.spinner("Extracting skills via AI..."):
            skill_data = llm_skill_extraction(resume_text, jd_text)

        st.subheader("💡 Skills Matched")
        st.dataframe(skill_data)

        # Save reports
        save_reports(skill_data, section_scores, similarity_score)
        st.success("Reports saved in output folder ✅")
