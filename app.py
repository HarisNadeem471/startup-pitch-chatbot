from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from rag_core.pdf_reader import extract_text_from_pdf
from rag_core.embed_store import chunk_text, embed_and_store
from rag_core.retriever import retrieve_top_chunks
from rag_core.generate import generate_answer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_and_process():
    pdf = request.files['pdf']
    question = request.form['question']

    if not pdf:
        return "No file uploaded", 400

    filename = secure_filename(pdf.filename)
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pdf.save(pdf_path)

    # Step 1: Extract text
    text = extract_text_from_pdf(pdf_path)

    # Step 2: Chunk + Embed + Store
    chunks = chunk_text(text)
    embed_and_store(chunks)

    # Step 3: Retrieve + Generate
    top_chunks = retrieve_top_chunks(question)
    answer = generate_answer(top_chunks, question)

    return render_template('index.html', answer=answer)

if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
