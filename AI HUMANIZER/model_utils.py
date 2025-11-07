# model_utils.py
import math
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import re

PARAPHRASE_MODEL = "Vamsi/T5_Paraphrase_Paws"   # can swap later
SIM_MODEL = "all-MiniLM-L6-v2"

# device-aware
device = "cuda" if torch.cuda.is_available() else "cpu"

# Use slow tokenizer to avoid fast-tokenizer conversion issues
tokenizer = AutoTokenizer.from_pretrained(PARAPHRASE_MODEL, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(PARAPHRASE_MODEL).to(device)

# Semantic similarity model
sbert = SentenceTransformer(SIM_MODEL, device=device)

# Helpers
def _split_to_chunks(text, max_tokens=180):
    """
    Very simple chunker: split by sentences and group until token length approx max_tokens.
    Avoids cutting mid-sentence.
    """
    # naive sentence split (keeps punctuation)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current = ""
    for s in sentences:
        if not s:
            continue
        cand = (current + " " + s).strip() if current else s
        tok_len = len(tokenizer.encode(cand, truncation=False))
        if tok_len <= max_tokens:
            current = cand
        else:
            if current:
                chunks.append(current)
            # if single sentence longer than max_tokens, we force-slice by tokens
            if len(tokenizer.encode(s)) > max_tokens:
                toks = tokenizer.encode(s, add_special_tokens=False)
                # slice into token ranges and decode back
                n = max_tokens
                for i in range(0, len(toks), n):
                    piece = tokenizer.decode(toks[i:i+n], clean_up_tokenization_spaces=True)
                    chunks.append(piece)
                current = ""
            else:
                current = s
    if current:
        chunks.append(current)
    return chunks

def paraphrase_text(text, max_length=256, num_beams=5):
    """
    Paraphrase input text robustly by chunking long input and generating
    paraphrase for each chunk using beam search for consistent outputs.
    """
    if not text.strip():
        return ""

    chunks = _split_to_chunks(text, max_tokens=180)
    paraphrased_chunks = []
    for chunk in chunks:
        prompt = "paraphrase: " + chunk
        inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        # use beam search (deterministic) for higher quality; avoid sampling randomness
        outputs = model.generate(
            inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        paraphrased_chunks.append(decoded.strip())

    # join with space (preserve sentence boundaries)
    humanized = " ".join(paraphrased_chunks).replace("  ", " ").strip()
    return humanized

def lexical_overlap(a, b):
    """
    Simple token-set Jaccard overlap (casefolded, punctuation removed)
    Lower lexical overlap means more changed wording.
    """
    # remove punctuation, tokenize by whitespace
    def tokens(s):
        s = re.sub(r'[^\w\s]', '', s).lower()
        return set(s.split())
    ta = tokens(a)
    tb = tokens(b)
    if not ta and not tb:
        return 1.0
    inter = ta.intersection(tb)
    union = ta.union(tb)
    return round(len(inter) / len(union) if union else 0.0, 3)

def semantic_similarity(a, b):
    """Return cosine similarity between two sentences (0..1)."""
    if not a.strip() or not b.strip():
        return 0.0
    emb1 = sbert.encode(a, convert_to_tensor=True)
    emb2 = sbert.encode(b, convert_to_tensor=True)
    score = util.pytorch_cos_sim(emb1, emb2).item()
    # clamp and round
    score = max(min(score, 1.0), -1.0)
    # map -1..1 to 0..1 if needed, but SBERT returns 0..1 for similar texts typically
    return round(float(score), 4)
