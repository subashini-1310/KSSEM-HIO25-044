from flask import Flask, request, jsonify, render_template
from model_utils import paraphrase_text
from PyPDF2 import PdfReader
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def extract_text_from_pdf(file):
    """Extract text from PDF using PyPDF2"""
    text = ""
    try:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        print("PDF extraction failed:", e)
    return text.strip()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/humanize", methods=["POST"])
def humanize():
    text = request.form.get("text", "")
    file = request.files.get("file")

    if file and file.filename.endswith(".pdf"):
        text = extract_text_from_pdf(file)

    if not text.strip():
        return jsonify({"error": "No content found in file or text box"}), 400

    humanized = paraphrase_text(text)
    return jsonify({"humanized_text": humanized})


if __name__ == "__main__":
    app.run(debug=True)
