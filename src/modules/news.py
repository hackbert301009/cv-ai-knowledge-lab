"""Live News — KI/CV News aus RSS-Feeds und arXiv."""
import streamlit as st
import feedparser
from datetime import datetime
from src.components import hero, section_header, divider, info_box


# Kuratiertes Feed-Set — schnell & relevant
FEEDS = {
    "arXiv CV":          "http://export.arxiv.org/rss/cs.CV",
    "arXiv ML":          "http://export.arxiv.org/rss/cs.LG",
    "Hugging Face Blog": "https://huggingface.co/blog/feed.xml",
    "OpenAI Blog":       "https://openai.com/blog/rss.xml",
    "DeepMind Blog":     "https://deepmind.google/blog/rss.xml",
    "Google AI Blog":    "https://blog.research.google/feeds/posts/default",
    "Anthropic News":    "https://www.anthropic.com/news/rss.xml",
    "MIT News AI":       "https://news.mit.edu/topic/mitartificial-intelligence2-rss.xml",
}


@st.cache_data(ttl=1800)  # 30 Minuten Cache
def fetch_feed(url: str, max_items: int = 10):
    """Feed laden, kompakte Liste zurückgeben."""
    try:
        feed = feedparser.parse(url)
        items = []
        for entry in feed.entries[:max_items]:
            items.append({
                "title": entry.get("title", "Ohne Titel"),
                "link":  entry.get("link", "#"),
                "summary": entry.get("summary", "")[:300],
                "published": entry.get("published", entry.get("updated", "")),
            })
        return items
    except Exception as e:
        return [{"title": f"⚠️ Fehler beim Laden: {e}", "link": "#", "summary": "", "published": ""}]


def render():
    hero(
        eyebrow="Live · Modul 23",
        title="Live News",
        sub="Aktuelle Forschung, Releases und Industrienews aus der KI- und CV-Welt — "
            "direkt aus arXiv, OpenAI, DeepMind, HuggingFace & Co."
    )

    info_box(
        "Diese Seite zieht live von RSS-Feeds. Beim ersten Aufruf kann es ein paar Sekunden dauern. "
        "Die Daten werden 30 Minuten gecached.",
        kind="info",
    )

    selected = st.multiselect(
        "Quellen wählen",
        options=list(FEEDS.keys()),
        default=["arXiv CV", "Hugging Face Blog", "OpenAI Blog"],
    )

    max_items = st.slider("Max. Items pro Quelle", 3, 15, 5)

    if not selected:
        st.info("Wähle mindestens eine Quelle aus.")
        return

    divider()

    for source in selected:
        section_header(f"📰 {source}")
        items = fetch_feed(FEEDS[source], max_items)
        for it in items:
            with st.container():
                st.markdown(f"#### [{it['title']}]({it['link']})")
                if it["published"]:
                    st.caption(f"📅 {it['published']}")
                if it["summary"]:
                    # einfache HTML-Bereinigung
                    summary = it["summary"]
                    import re
                    summary = re.sub(r"<[^>]+>", "", summary)
                    st.markdown(summary[:280] + ("..." if len(summary) > 280 else ""))
                st.markdown("---")

    divider()
    info_box(
        "Tipp: Speichere dir interessante Paper in Zotero oder einer Notiz-App. "
        "Die Halbwertszeit von KI-News ist kurz — was zählt, ist was du wirklich verstanden hast.",
        kind="tip",
    )
