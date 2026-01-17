import os
import json
from fuzzywuzzy import fuzz
from .utils import clean_text, extract_sections
import openai

# -------------------------------
# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------
def compute_similarity(resume_text, jd_text):
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)
    return fuzz.token_set_ratio(resume_clean, jd_clean)

# -------------------------------
def section_scoring(resume_text, jd_text):
    resume_sections = extract_sections(resume_text)
    jd_sections = extract_sections(jd_text)
    scores = {}
    for sec in resume_sections:
        if sec in jd_sections:
            scores[sec] = fuzz.token_set_ratio(clean_text(resume_sections[sec]), clean_text(jd_sections[sec]))
        else:
            scores[sec] = 0
    return scores

# -------------------------------
def llm_skill_extraction(resume_text, jd_text):
    """
    Extract skills automatically using GPT model
    """
    prompt = f"""
You are an AI recruiter assistant.

1. Analyze the following Resume text:
{resume_text}

2. Compare it with the following Job Description:
{jd_text}

3. Extract all relevant skills mentioned in the resume and match them to JD.
4. Output a JSON list with this format:
[
  {{
    "Skill": "Skill Name",
    "Mentioned_in_Resume": true/false,
    "Mentioned_in_JD": true/false,
    "Relevance_Score": 0-100
  }}
]
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response.choices[0].message.content
    try:
        skill_data = json.loads(content)
    except json.JSONDecodeError:
        skill_data = {"raw_output": content}

    return skill_data

# -------------------------------
def save_reports(skill_data, section_scores, similarity_score, output_folder="output"):
    import pandas as pd
    os.makedirs(output_folder, exist_ok=True)

    # Skills CSV & JSON
    df_skills = pd.DataFrame(skill_data)
    df_skills.to_csv(os.path.join(output_folder, "skills.csv"), index=False)
    with open(os.path.join(output_folder, "skills.json"), "w") as f:
        json.dump(skill_data, f, indent=4)

    # Sections CSV
    df_sections = pd.DataFrame(list(section_scores.items()), columns=["Section", "Score (%)"])
    df_sections.to_csv(os.path.join(output_folder, "section_scores.csv"), index=False)

    # Summary
    summary = {
        "Resume-JD Similarity": similarity_score
    }
    with open(os.path.join(output_folder, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    return output_folder
