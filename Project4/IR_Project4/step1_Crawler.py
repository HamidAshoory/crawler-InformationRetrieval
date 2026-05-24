import os
import re
import json
import time
from collections import deque
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

START_URL = "https://en.um.ac.ir/"
ALLOWED_DOMAIN = "en.um.ac.ir"

MAX_PAGES = 3000          
DELAY = 0.3              
TIMEOUT = 20

DATA_DIR = "data"
HTML_DIR = os.path.join(DATA_DIR, "html")
META_DIR = os.path.join(DATA_DIR, "meta")
STATE_DIR = os.path.join(DATA_DIR, "state")

os.makedirs(HTML_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)

FRONTIER_PATH = os.path.join(STATE_DIR, "frontier.json")
VISITED_PATH = os.path.join(STATE_DIR, "visited.json")


def canonicalize(url: str) -> str:
    """Remove fragment + query to reduce duplicates."""
    p = urlparse(url)
    scheme = "https" 
    return urlunparse((scheme, p.netloc, p.path, "", "", ""))
def is_valid_url(url: str) -> bool:
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https"):
            return False
        if not p.netloc.endswith(ALLOWED_DOMAIN):
            return False

        # skip non-text files
        if re.search(r"\.(pdf|jpg|jpeg|png|gif|zip|rar|mp4|mp3|svg|doc|docx|xls|xlsx)$", p.path.lower()):
            return False

        # skip empty path weirdness
        return True
    except:
        return False

def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return default
    return default

def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def get_next_id(meta_dir=META_DIR) -> int:
    """Next doc id based on saved meta files."""
    ids = []
    for fname in os.listdir(meta_dir):
        if fname.endswith(".json"):
            try:
                ids.append(int(fname.replace(".json", "")))
            except:
                pass
    return (max(ids) + 1) if ids else 0


session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) IR-Crawler/1.0"
})

def fetch_html(url: str):
    try:
        r = session.get(url, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        ct = r.headers.get("Content-Type", "")
        if "text/html" not in ct:
            return None
        return r.text
    except:
        return None

# Resume state
visited = set(load_json(VISITED_PATH, []))
frontier_list = load_json(FRONTIER_PATH, [START_URL])

# canonicalize loaded frontier
frontier = deque([canonicalize(u) for u in frontier_list if isinstance(u, str)])
if not frontier:
    frontier = deque([START_URL])

next_id = get_next_id()
saved_pages = next_id

print("=== Resume State ===")
print("Already visited:", len(visited))
print("Already saved:", saved_pages)
print("Frontier size:", len(frontier))
print("====================")

# Crawl loop
pbar = tqdm(total=MAX_PAGES, initial=saved_pages)

while frontier and saved_pages < MAX_PAGES:
    url = frontier.popleft()
    url = canonicalize(url)

    if url in visited:
        continue
    if not is_valid_url(url):
        continue

    html = fetch_html(url)
    visited.add(url)

    if len(visited) % 30 == 0:
        save_json(VISITED_PATH, sorted(list(visited)))
        save_json(FRONTIER_PATH, list(frontier))

    if html is None:
        continue

    # Save HTML
    with open(os.path.join(HTML_DIR, f"{saved_pages}.html"), "w", encoding="utf-8") as f:
        f.write(html)

    # Save meta
    meta = {"id": saved_pages, "url": url}
    with open(os.path.join(META_DIR, f"{saved_pages}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    saved_pages += 1
    pbar.update(1)

    # Extract links
    soup = BeautifulSoup(html, "lxml")
    for a in soup.select("a[href]"):
        href = a.get("href", "").strip()
        if not href:
            continue
        nxt = urljoin(url, href)
        nxt = nxt.split("#")[0]
        nxt = canonicalize(nxt)

        if is_valid_url(nxt) and nxt not in visited:
            frontier.append(nxt)

    time.sleep(DELAY)

pbar.close()

# Final save state
save_json(VISITED_PATH, sorted(list(visited)))
save_json(FRONTIER_PATH, list(frontier))

print("\nDone.")
print("Saved pages:", saved_pages)
print("Visited URLs:", len(visited))
print("Remaining frontier:", len(frontier))

