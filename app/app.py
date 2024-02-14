from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import spacy
from utils import *

# Initialize Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit
app.secret_key = 'super_secret_key'

# Load Spacy model and add entity ruler
nlp = spacy.load('en_core_web_md')
ruler = nlp.add_pipe("entity_ruler", before="ner")

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Entity extraction logic here (unchanged from your initial implementation)

# Route for displaying the upload form and processing uploads
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            extracted_data = pdfReader(filepath)
            os.remove(filepath)  # Clean up the uploaded file
            return render_template('upload.html', extracted_data=extracted_data)
        else:
            flash('Invalid file type. Please upload a PDF file.')
            return redirect(url_for('upload_file'))
    return render_template('upload.html')

# Ensure the UPLOAD_FOLDER exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

if __name__ == '__main__':
    app.run(debug=True)
