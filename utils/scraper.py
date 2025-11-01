import os
import requests
import re
from bs4 import BeautifulSoup
from ddgs import DDGS  # new library (replaces duckduckgo_search)
from PyPDF2 import PdfReader

# === Paths ===
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "kb_docs")

# === Helpers ===
def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name.strip())

# === Search ===
def search_web(query, max_results=10):
    """Search the web using DDGS and return a list of URLs."""
    urls = []
    try:
        with DDGS() as ddgs:
            for result in ddgs.text(query, max_results=max_results):
                link = result.get("href") or result.get("url")
                if link and link.startswith("http"):
                    urls.append(link)
    except Exception as e:
        print(f"⚠️ Search error: {e}")

    print(f"🔎 Found {len(urls)} URLs for '{query}'")
    return urls

# === Download PDFs ===
def download_pdf(url, folder):
    """Download a PDF and save it locally, even if content-type is wrong."""
    try:
        filename = sanitize_filename(url.split("/")[-1])
        if not filename.endswith(".pdf"):
            filename += ".pdf"
        file_path = os.path.join(folder, filename)

        response = requests.get(url, timeout=25, allow_redirects=True)

        # check first few bytes for PDF header
        if response.status_code == 200 and (response.content[:4] == b"%PDF" or "pdf" in response.headers.get("Content-Type", "").lower()):
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"✅ Downloaded: {filename}")
            return file_path
        else:
            print(f"⚠️ Skipped (not PDF content): {url}")
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
    return None


# === Extract PDF Text ===
def extract_text_from_pdf(pdf_path):
    """Extract text from a downloaded PDF."""
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        print(f"❌ Error reading {pdf_path}: {e}")
    return text


# === Cleanup Helpers ===
def is_effectively_empty_file(path):
    """Return True if file is zero bytes or contains only whitespace when decoded as text.
    We try reading as text and consider files with only whitespace as empty (e.g., blank PDFs converted to empty .txt).
    """
    try:
        if os.path.getsize(path) == 0:
            return True
        # Try to read small portion as text to detect whitespace-only
        with open(path, "rb") as f:
            content = f.read()
            # If it's binary that's fine; try decode utf-8 with errors ignored
            text = content.decode("utf-8", errors="ignore")
            if text.strip() == "":
                return True
    except Exception as e:
        # If we can't read it, don't delete just in case
        print(f"⚠️ Could not inspect file for emptiness: {path} ({e})")
    return False


def clean_kb_docs(root_dir):
    """Walk `root_dir` and remove empty files and empty directories.

    - Removes files that are zero bytes or text-only whitespace.
    - After file removals, removes directories that are empty.
    """
    removed_files = 0
    removed_dirs = 0

    # First pass: remove empty files
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            if is_effectively_empty_file(fpath):
                try:
                    os.remove(fpath)
                    removed_files += 1
                    print(f"🧹 Removed empty file: {fpath}")
                except Exception as e:
                    print(f"⚠️ Failed to remove file {fpath}: {e}")

    # Second pass: remove empty directories (bottom-up)
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        try:
            # skip root_dir itself
            if dirpath == root_dir:
                continue
            if not os.listdir(dirpath):
                os.rmdir(dirpath)
                removed_dirs += 1
                print(f"🧹 Removed empty directory: {dirpath}")
        except Exception as e:
            print(f"⚠️ Failed to remove directory {dirpath}: {e}")

    print(f"🧽 Cleanup complete: removed {removed_files} empty files and {removed_dirs} empty directories from {root_dir}")

# === Main Scraper ===
def scrape_topic(query, max_results=10):
    """Search, download, and extract text from PDFs."""
    topic_dir = os.path.join(BASE_DIR, sanitize_filename(query))
    os.makedirs(topic_dir, exist_ok=True)

    urls = search_web(query, max_results=max_results)
    pdf_texts = []

    for url in urls:
        if ".pdf" in url.lower():
            pdf_path = download_pdf(url, topic_dir)
            if pdf_path:
                text = extract_text_from_pdf(pdf_path)
                if text:
                    out_txt = pdf_path.replace(".pdf", ".txt")
                    with open(out_txt, "w", encoding="utf-8") as f:
                        f.write(text)
                    pdf_texts.append(out_txt)

    print(f"\n📂 Saved {len(pdf_texts)} text files under {topic_dir}")
    # Cleanup any empty files or empty folders created during scraping for this topic
    try:
        clean_kb_docs(topic_dir)
    except Exception as e:
        print(f"⚠️ Cleanup failed for {topic_dir}: {e}")

    return pdf_texts

# === Allianz-specific Scraper ===
def scrape_allianz_docs(query="insurance technology 2025 site:allianz.com", max_results=10):
    """Wrapper to scrape Allianz documents."""
    return scrape_topic(query, max_results=max_results)


# === Example Run ===
if __name__ == "__main__":
    # multiple queries to run
    queries = [
        "insurance technology 2025 site:swissre.com",
        "insurance trends 2025 site:munichre.com",
        "insurance market report 2025 site:oecd.org",
        "insurance report 2025 site:iii.org",
    ]

    def scrape_queries(queries_list, max_results=10):
        all_files = {}
        for q in queries_list:
            print(f"\n=== Scraping: {q} ===")
            files = scrape_topic(q, max_results=max_results)
            all_files[q] = files
        return all_files

    scrape_queries(queries, max_results=10)

    # Final cleanup across kb_docs root to remove any stray empty files/dirs
    try:
        clean_kb_docs(BASE_DIR)
    except Exception as e:
        print(f"⚠️ Final cleanup failed for {BASE_DIR}: {e}")
