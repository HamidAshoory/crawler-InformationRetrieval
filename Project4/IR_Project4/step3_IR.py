import json
import math
import time
import os
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
import pickle

# Load docs_clean.json (Output of Step 2)
DOCS_PATH = "output/docs_clean.json"

with open(DOCS_PATH, "r", encoding="utf-8") as f:
    docs = json.load(f)

print("Loaded docs:", len(docs))

# Build TF Index using project3
# Structure: {term: [(doc_id, tf), ...]}
tf_index = defaultdict(list)
doc_ids = []

for doc in docs:
    doc_id = doc["id"]
    doc_ids.append(doc_id)

    tokens = doc["tokens"]
    tf = Counter(tokens)

    for term, freq in tf.items():
        tf_index[term].append((doc_id, freq))

print("TF index vocabulary size:", len(tf_index))


def compute_idf(inverted_index, total_docs):
    idf_dict = {}
    for term, postings in inverted_index.items():
        df = len(postings)
        idf_dict[term] = math.log10(total_docs / df) if df > 0 else 0
    return idf_dict

def compute_doc_lengths(tf_index, idf_dict):
    doc_sq_sum = defaultdict(float)

    for term, postings in tf_index.items():
        idf = idf_dict.get(term, 0)
        for doc_id, tf in postings:
            raw_weight = tf * idf
            doc_sq_sum[doc_id] += raw_weight ** 2

    doc_lengths = {doc: math.sqrt(val) for doc, val in doc_sq_sum.items()}
    return doc_lengths

def build_normalized_tfidf_index(tf_index, idf_dict, doc_lengths):
    tfidf_index = defaultdict(list)

    for term, postings in tf_index.items():
        idf = idf_dict.get(term, 0)
        for doc_id, tf in postings:
            raw_weight = tf * idf
            doc_len = doc_lengths.get(doc_id, 0)
            norm_weight = raw_weight / doc_len if doc_len > 0 else 0
            tfidf_index[term].append((doc_id, norm_weight))

    return tfidf_index

def compute_query_vector(query_tokens, idf_dict):
    query_tf = Counter(query_tokens)

    query_weights = {}
    query_sq_sum = 0.0

    for term, tf in query_tf.items():
        if term in idf_dict:
            idf = idf_dict[term]
            weight = tf * idf
            query_weights[term] = weight
            query_sq_sum += weight ** 2

    query_len = math.sqrt(query_sq_sum)

    norm_query_vector = {}
    if query_len > 0:
        for term, weight in query_weights.items():
            norm_query_vector[term] = weight / query_len

    return norm_query_vector


def preprocess_pipeline(tokens):
    out = []
    for t in tokens:
        t = t.lower()
        t = "".join(ch for ch in t if ch.isalnum())
        if t:
            out.append(t)
    return out


def search(query_text, index, idf_dict, top_k=10):
    tokens = word_tokenize(query_text)
    processed_query = preprocess_pipeline(tokens)

    query_vector = compute_query_vector(processed_query, idf_dict)

    scores = defaultdict(float)

    for term, q_weight in query_vector.items():
        if term in index:
            for doc_id, d_weight in index[term]:
                scores[doc_id] += q_weight * d_weight

    ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_results[:top_k]


# Build Final Index (Normalized TF-IDF)
start = time.time()

total_docs = len(doc_ids)
idf_values = compute_idf(tf_index, total_docs)
doc_lengths = compute_doc_lengths(tf_index, idf_values)
final_index = build_normalized_tfidf_index(tf_index, idf_values, doc_lengths)

end = time.time()
print("\nSTEP 3 Index Build Done")
print(f"Time Taken: {end - start:.2f} seconds")
print("Final vocab size:", len(final_index))


# Test Search
print("\n--- Step 3: Testing Retrieval ---")
test_query = "oil prices in international markets"
print(f"Query: '{test_query}'")

t0 = time.time()
results = search(test_query, final_index, idf_values, top_k=10)
t1 = time.time()

print(f"Search Time: {t1 - t0:.6f} seconds")
print("\nTop 10 Results:")
print(f"{'Rank':<5} {'Score':<10} {'Doc ID':<15}")
print("-" * 35)

for i, (doc_id, score) in enumerate(results):
    print(f"{i+1:<5} {score:.4f}     {doc_id:<15}")

    # show url of top results
doc_map = {d["id"]: d["url"] for d in docs}

print("\nTop URLs:")
for doc_id, score in results[:10]:
    print(doc_id, score, doc_map.get(doc_id, "N/A"))

os.makedirs("output", exist_ok=True)

with open("output/final_index.pkl", "wb") as f:
    pickle.dump(final_index, f)

with open("output/idf_values.pkl", "wb") as f:
    pickle.dump(idf_values, f)

print("final_index and idf_values saved successfully")