import time
import statistics
from step3_IR import search
import json
import pickle

with open("output/docs_clean.json", "r", encoding="utf-8") as f:
    docs = json.load(f)

id_to_url = {d["id"]: d["url"] for d in docs}

with open("output/final_index.pkl", "rb") as f:
    final_index = pickle.load(f)

with open("output/idf_values.pkl", "rb") as f:
    idf_values = pickle.load(f)

QUERIES = [
    "computer engineering faculty members",
    "international students admission",
    "research centers at ferdowsi university",
    "graduate programs in science",
    "campus facilities and libraries",
]

TOP_K = 10
times = []

print("="*60)
print("STEP 4: Query Timing + Top Results")
print("="*60)

for i, q in enumerate(QUERIES, 1):
    t0 = time.time()
    results = search(q, final_index, idf_values, top_k=TOP_K)
    t1 = time.time()
    
    elapsed = t1 - t0
    times.append(elapsed)

    seen_urls = set()
    filtered_results = []
    for doc_id, score in results:
        url = id_to_url.get(doc_id, "N/A")
        if url not in seen_urls:
            filtered_results.append((doc_id, score, url))
            seen_urls.add(url)

    print(f"\n[{i}] Query: {q}")
    print(f"Time: {elapsed:.6f} seconds")
    print("-"*50)

    for rank, (doc_id, score, url) in enumerate(filtered_results, 1):
        print(f"{rank:02d}. score={score:.4f} | id={doc_id} | {url}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Avg time: {statistics.mean(times):.6f} sec")
print(f"Min time: {min(times):.6f} sec")
print(f"Max time: {max(times):.6f} sec")
