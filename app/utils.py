import spacy
import pandas as pd
from PyPDF2 import PdfReader
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load('en_core_web_md')
ruler = nlp.add_pipe("entity_ruler")

nlp.pipe_names

# patterns
skill_path = "./data/jz_skill_patterns.jsonl"
ruler.from_disk(skill_path)

email_pattern = [{'label': 'EMAIL', 
                  'pattern': [{'TEXT': {'REGEX': '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'}}]}]
ruler.add_patterns(email_pattern)

education_pattern = [
    {"label": "EDUCATION", "pattern": [{"LOWER": {"IN": ["bsc", "bachelor", "bachelor's", "b.a", "b.s"]}}, {"IS_ALPHA": True, "OP": "*"}]},
    {"label": "EDUCATION", "pattern": [{"LOWER": {"IN": ["msc", "master", "master's", "m.a", "m.s"]}}, {"IS_ALPHA": True, "OP": "*"}]},
    {"label": "EDUCATION", "pattern": [{"LOWER": {"IN": ["phd", "ph.d", "doctor", "doctorate"]}}, {"IS_ALPHA": True, "OP": "*"}]}]
ruler.add_patterns(education_pattern)

web_patterns = [
    {"label": "WEBSITE", "pattern": [{"TEXT": {"REGEX": "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"}}]}]
ruler.add_patterns(web_patterns)

work_experience_patterns = [
    {"label": "JOB_TITLE", "pattern": [{"TEXT": {"REGEX": "(?i)\b(manager|engineer|developer|director)\b"}}]},
    {"label": "COMPANY_NAME", "pattern": [{"TEXT": {"REGEX": "(?i)\b(Inc\.?|Ltd\.?|LLC|Corporation)\b"}}]},
    {"label": "DATE", "pattern": [{"TEXT": {"REGEX": "\b(19|20)\d{2}\b"}}, {"TEXT": {"REGEX": "to"}}, {"TEXT": {"REGEX": "\b(19|20)\d{2}\b"}}]},
    {"label": "JOB_RESPONSIBILITY", "pattern": [{"TEXT": {"REGEX": "(?i)\b(responsible for|duties included|managed)\b"}}]}]

ruler.add_patterns(work_experience_patterns)


# preprocessing
from spacy.lang.en.stop_words import STOP_WORDS
def preprocessing(sentence):

    stopwords    = list(STOP_WORDS)
    doc          = nlp(sentence)
    clean_tokens = []
    
    for token in doc:
        if token.text not in stopwords and token.pos_ != 'PUNCT' and token.pos_ != 'SYM' and \
            token.pos_ != 'SPACE':
                clean_tokens.append(token.lemma_.lower().strip())
                
    return " ".join(clean_tokens)

from PyPDF2 import PdfReader
def pdfReader(cv_path):
    reader = PdfReader(cv_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + " "  # Concatenate all pages
    text = preprocessing(text)
    doc = nlp(text)

    skills = []
    education = []  # Corrected variable name
    email = []
    website = []
    job_title = []
    company_name = []
    date = []
    job_responsibility = []

    for ent in doc.ents:
        if ent.label_ == 'SKILL':
            skills.append(ent.text)
        elif ent.label_ == 'EDUCATION':
            education.append(ent.text)  # Ensure consistency in variable naming
        elif ent.label_ == 'EMAIL':
            email.append(ent.text)
        elif ent.label_ == 'WEBSITE':
            website.append(ent.text)
        elif ent.label_ == 'JOB_TITLE':
            job_title.append(ent.text)
        elif ent.label_ == 'COMPANY_NAME':
            company_name.append(ent.text)
        elif ent.label_ == 'DATE':
            date.append(ent.text)
        elif ent.label_ == 'JOB_RESPONSIBILITY':
            job_responsibility.append(ent.text)

    dict = {'skills': list(set(skills)), 
            'education_status': list(set(education)), 
            'email': list(set(email)), 
            'website': list(set(website)),
            'job_title': list(set(job_title)),
            'company_name': list(set(company_name)),
            'date': list(set(date)),
            'job_responsibility': list(set(job_responsibility)),
            }

    return dict
