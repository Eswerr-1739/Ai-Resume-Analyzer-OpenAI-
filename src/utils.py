import spacy
import pdfplumber
from docx import Document

nlp = spacy.load("en_core_web_sm")

# -------------------------------
def clean_text(text):
    """Clean text with Spacy: lowercase, lemmatize, remove stop words & non-alpha"""
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])

# -------------------------------
def extract_text_from_file(file_path):
    """Extract text from TXT, PDF, DOCX"""
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif file_path.endswith(".pdf"):
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    else:
        raise ValueError("Unsupported file type. Use TXT, PDF, or DOCX.")

# -------------------------------
def extract_sections(text):
    """
    Simple rule-based section extractor
    Returns dict: {section_name: content}
    """
    sections = {}
    current_section = "General"
    lines = text.splitlines()
    sections[current_section] = ""

    for line in lines:
        line_strip = line.strip().lower()
        if any(h in line_strip for h in ["experience", "work experience", "education", "skills", "projects"]):
            current_section = line_strip.title()
            sections[current_section] = ""
        else:
            sections[current_section] += line + " "
    return sections
