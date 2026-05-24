import os
import json
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm


HTML_DIR = "data/html"
META_DIR = "data/meta"
OUTPUT_PATH = "output/docs_clean.json"

os.makedirs("output", exist_ok=True)


STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


def clean_text(text: str) -> str:
    # lowercase + remove non-alphanum + collapse spaces
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_tokens(tokens):
    cleaned_tokens = []
    for t in tokens:
        t2 = re.sub(r"[^a-z0-9]", "", t)
        if t2 and t2 not in STOPWORDS and len(t2) >= 3:
            cleaned_tokens.append(STEMMER.stem(t2))
    return cleaned_tokens

docs = []
html_files = sorted([f for f in os.listdir(HTML_DIR) if f.endswith(".html")])

for fname in tqdm(html_files, desc="Processing HTML"):
    doc_id = int(fname.replace(".html", ""))
    html_path = os.path.join(HTML_DIR, fname)
    meta_path = os.path.join(META_DIR, f"{doc_id}.json")

    try:
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except:
        continue

    # Fast text extraction
    try:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        raw_text = soup.get_text(separator=" ", strip=True)
    except:
        raw_text = ""

    cleaned_text = clean_text(raw_text)
    tokens = cleaned_text.split()
    tokens = preprocess_tokens(tokens)

    docs.append({
        "id": doc_id,
        "url": meta.get("url", ""),
        "clean_text": cleaned_text,
        "tokens": tokens,
        "length": len(tokens)
    })

# Filter very short docs
docs = [d for d in docs if d["length"] >= 50]

# Save output
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(docs, f, ensure_ascii=False, indent=2)

print(f"Done! Total docs after filtering: {len(docs)}")
