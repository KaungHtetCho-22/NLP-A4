import spacy
import pandas as pd
from PyPDF2 import PdfReader
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load('en_core_web_md')
ruler = nlp.add_pipe("entity_ruler")

nlp.pipe_names

# patterns
skill_path       = "./data/jz_skill_patterns.jsonl"
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

# work_patterns = [
#     # General pattern for job titles - looking for noun phrases that could be job titles
#     {"label": "JOB_TITLE", "pattern": [{"POS": "PROPN"}, {"POS": "NOUN", "OP": "?"}]},
#     # General pattern for companies - looking for proper nouns that could indicate company names
#     {"label": "COMPANY", "pattern": [{"POS": "PROPN"}, {"POS": "PROPN", "OP": "?"}]},
#     # To catch patterns like 'Company Name, Job Title, Years'
#     {"label": "WORK_EXPERIENCE", "pattern": [{"POS": "PROPN"}, {"POS": "PROPN", "OP": "?"}, {"POS": "PUNCT"}, {"POS": "PROPN"}, {"POS": "NOUN", "OP": "?"}, {"POS": "PUNCT"}, {"SHAPE": "dddd"}]},]
# ruler.add_patterns(work_patterns)

# "(\\+?\\d{1,3})?[-\\s.]?(\\d{1,4})?[-\\s.]?(\\d{2,4})[-\\s.]?(\\d{2,4})[-\\s.]?(\\d{2,4})"

# date_pattern = [
#     {"label": "DATE", "pattern": [
#         {"LOWER": {"IN": ["january", "february", "march", "april", "may", "june", "july", 
#                           "august", "september", "october", "november", "december"]}},
#         {"SHAPE": "dddd"},
#         {"LOWER": "-", "OP": "?"},
#         {"LOWER": {"IN": ["-", "–"]}, "OP": "?"},
#         {"LOWER": {"IN": ["present", "january", "february", "march", "april", "may", "june", 
#                           "july", "august", "september", "october", "november", "december"]}, "OP": "?"},
#         {"SHAPE": "dddd", "OP": "?"}
#     ]}
# ]

# mobile_pattern = [
#     # {"label": "MOBILE", "pattern": [{"TEXT": {"REGEX": "(\+?\d{1,3})?[-\s.]?(\d{1,4})?[-\s.]?(\d{2,4})[-\s.]?(\d{2,4})[-\s.]?(\d{2,4})"}}]},
#     {"label": "MOBILE", "pattern": [{"TEXT": {"REGEX": "\+?(\d{7,15})"}}]}]
# ruler.add_patterns(mobile_pattern)

work_pattern = [
    # Pattern for Job Title
    {"label": "JOB_TITLE", "pattern": [{"POS": "PROPN"}, {"POS": "PROPN", "OP": "?"}]},
    
    # Pattern for Organization
    {"label": "COMPANY", "pattern": [{"POS": "PROPN"}, {"POS": "PROPN", "OP": "?"}, {"POS": "PROPN", "OP": "?"}]}
    
    # Pattern for Location
    # {"label": "LOCATION", "pattern": [{"POS": "PROPN"}, {"POS": "PROPN", "OP": "?"}]},
    
    # Pattern for Dates
    # {"label": "DATE", "pattern": [{"SHAPE": "ddd"}, {"SHAPE": "dddd"}]},
    # {"label": "DATE", "pattern": [{"SHAPE": "dd"}, {"ORTH": "-"}, {"SHAPE": "dddd"}]}
    
    # Job Descriptions are more complex and may need more sophisticated patterns or machine learning approaches.
]
ruler.add_patterns(work_pattern)




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
        text += page.extract_text() + " "  
    text = preprocessing(text)
    doc = nlp(text)
    
    name            = []
    skills          = []
    education       = []  
    email           = []
    website         = []
    org             = []
    # mobile           = []
    job_title       = []
    company         = []
    # work_experience = []

    for ent in doc.ents:
        if ent.label_ == 'SKILL':
            skills.append(ent.text)
        elif ent.label_ == 'EDUCATION':
            education.append(ent.text)  
        elif ent.label_ == 'EMAIL':
            email.append(ent.text)
        elif ent.label_ == 'WEBSITE':
            website.append(ent.text)
        elif ent.label_ == 'ORG':
            org.append(ent.text)
        elif ent.label_ == 'PERSON':
            name.append(ent.text)
        # elif ent.label_ == 'MOBILE':
        #     mobile.append(ent.text)
        elif ent.label_ == 'JOB_TITLE':
            job_title.append(ent.text)
        elif ent.label_ == 'COMPANY':
            company.append(ent.text)
        # elif ent.label_ == 'WORK_EXPERIENCE':
        #     work_experience.append(ent.text)

    dict = {'email': list(set(email)),
            'name': list(set(name)),
            # 'mobile': list(set(mobile)),
            'website': list(set(website)),
            'education_status': list(set(education)), 
            'org':list(set(org)),
            'skills': list(set(skills)),
            'job_title': list(set(job_title)),
            'company': list(set(company))
            # 'work_experience': list(set(work_experience)),
            }

    return dict
