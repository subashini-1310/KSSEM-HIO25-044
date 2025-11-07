# app.py
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from model_utils import paraphrase_text, semantic_similarity, lexical_overlap

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/humanize", methods=["POST"])
def humanize_text():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        humanized = paraphrase_text(text)
        sem_sim = semantic_similarity(text, humanized)
        lex_ov = lexical_overlap(text, humanized)

        # helpful score interpretation
        result = {
            "original": text,
            "humanized": humanized,
            "semantic_similarity": sem_sim,   # close to 1 => meaning preserved
            "lexical_overlap": lex_ov        # lower => more changed wording
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
