# web_researcher.py
# Agentic web research module
# Takes a topic, searches the web, fetches content,
# and returns clean chunked text ready for embedding into Endee

from duckduckgo_search import DDGS
import wikipedia
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict
import time


# Same chunk settings as document_processor for consistency
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " ", ""]
)


def search_duckduckgo(topic: str, max_results: int = 5) -> List[Dict]:
    """
    Search DuckDuckGo for a topic and return top results.
    Returns list of dicts with title, url, and snippet.
    """
    results = []
    try:
        with DDGS() as ddgs:
            search_results = ddgs.text(
                topic,
                max_results=max_results
            )
            for r in search_results:
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")
                })
        print(f"[WebResearcher] Found {len(results)} DuckDuckGo results.")
    except Exception as e:
        print(f"[WebResearcher] DuckDuckGo search error: {e}")
    return results


def fetch_wikipedia_content(topic: str, sentences: int = 50) -> str:
    """
    Fetch detailed Wikipedia article for the topic.
    sentences=50 gives a comprehensive summary.
    """
    content = ""
    try:
        # Auto-suggest finds closest matching article
        page = wikipedia.page(topic, auto_suggest=True)
        content = wikipedia.summary(topic, sentences=sentences)
        print(f"[WebResearcher] Wikipedia content fetched: {page.title}")
    except wikipedia.exceptions.DisambiguationError as e:
        # If topic is ambiguous, take the first option
        try:
            content = wikipedia.summary(e.options[0], sentences=sentences)
            print(f"[WebResearcher] Disambiguation resolved to: {e.options[0]}")
        except Exception:
            print("[WebResearcher] Wikipedia disambiguation failed.")
    except wikipedia.exceptions.PageError:
        print(f"[WebResearcher] No Wikipedia page found for: {topic}")
    except Exception as e:
        print(f"[WebResearcher] Wikipedia error: {e}")
    return content


def build_research_content(topic: str) -> str:
    """
    Combines DuckDuckGo snippets + Wikipedia into
    one rich text body for the given topic.
    """
    full_content = f"RESEARCH TOPIC: {topic}\n\n"

    # Wikipedia gives deep structured knowledge
    print("[WebResearcher] Fetching Wikipedia content...")
    wiki_content = fetch_wikipedia_content(topic)
    if wiki_content:
        full_content += f"=== Wikipedia ===\n{wiki_content}\n\n"

    # DuckDuckGo gives recent/diverse web snippets
    print("[WebResearcher] Searching DuckDuckGo...")
    ddg_results = search_duckduckgo(topic, max_results=5)
    if ddg_results:
        full_content += "=== Web Search Results ===\n"
        for i, result in enumerate(ddg_results):
            full_content += f"\n[Source {i+1}] {result['title']}\n"
            full_content += f"URL: {result['url']}\n"
            full_content += f"{result['snippet']}\n"
            time.sleep(0.5)  # polite delay between requests

    print(f"[WebResearcher] Total content length: {len(full_content)} characters.")
    return full_content


def research_topic(topic: str) -> Dict:
    """
    Master function — researches a topic and returns
    everything needed for embedding and storing in Endee.

    Returns:
        {
            "topic": "Quantum Computing",
            "chunks": ["chunk1...", "chunk2...", ...],
            "metadata": [{"text": "...", "source": "web_research", "topic": "Quantum Computing", "chunk_id": 0}, ...]
        }
    """
    print(f"[WebResearcher] Starting research on: {topic}")

    # Build combined content from all sources
    raw_content = build_research_content(topic)

    if not raw_content.strip():
        print("[WebResearcher] No content found for topic.")
        return {}

    # Chunk the content
    chunks = text_splitter.split_text(raw_content)
    print(f"[WebResearcher] Created {len(chunks)} chunks from research.")

    # Build metadata
    metadata = [
        {
            "text": chunk,
            "source": "web_research",
            "topic": topic,
            "chunk_id": i
        }
        for i, chunk in enumerate(chunks)
    ]

    return {
        "topic": topic,
        "chunks": chunks,
        "metadata": metadata
    }