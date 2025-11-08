# model_utils.py
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import torch

# Load models once
PARAPHRASE_MODEL = "Vamsi/T5_Paraphrase_Paws"
tokenizer = AutoTokenizer.from_pretrained(PARAPHRASE_MODEL, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(PARAPHRASE_MODEL)
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

def paraphrase_text(text, num_return_sequences=1):
    """
    Generate a humanized (paraphrased) version of the text
    using T5 Paraphrase model with controlled randomness.
    """
    # Input preparation
    input_text = "paraphrase: " + text.strip()
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)

    # Generate output with balanced parameters (less distortion)
    outputs = model.generate(
        inputs,
        max_length=256,
        num_return_sequences=num_return_sequences,
        num_beams=5,            # deterministic beam search instead of random sampling
        early_stopping=True
    )

    # Decode
    paraphrased = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return paraphrased.strip()

def calculate_similarity(text1, text2):
    """
    Compute semantic similarity between two pieces of text
    using SentenceTransformer cosine similarity.
    """
   
    text1, text2 = text1.strip().lower(), text2.strip().lower()

    emb1 = similarity_model.encode(text1, convert_to_tensor=True, normalize_embeddings=True)
    emb2 = similarity_model.encode(text2, convert_to_tensor=True, normalize_embeddings=True)


    similarity_score = util.cos_sim(emb1, emb2).item()
    return round(similarity_score, 3)

if __name__ == "__main__":
    sample_text = "AI is transforming industries by automating tasks and improving decision-making."
    humanized = paraphrase_text(sample_text)
    sim = calculate_similarity(sample_text, humanized)
    
    print("Original:", sample_text)
    print("Paraphrased:", humanized)
    print("Similarity Score:", sim)
