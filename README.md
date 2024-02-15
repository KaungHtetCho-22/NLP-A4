# NLP-A4
NLP assignment from AIT

**Kaung Htet Cho (st124092)**

## Table of Contents
1. [Task1](#Task1)
2. [Task2](#Task2)


## Task1
### Implementation

- Extended resumer parser with another five categories (skills, education, email, website and contact number).
- Utilized Spacy + Custom + Regex 
- All ruler patterns are defined in './app/utils.py'
- implemented based on Chaky's resume

## Task2
### Web app documentation

The Website can be accessed on http://localhost:8000. User can upload Resume (only PDF format) and pass the preprocesssing function and get the clean text. By defining customized entity patterns (skills, education, email, website, mobile_number) for specific name entity recognition and shown the extracted results with the table and then users can download the results as .csv file.


