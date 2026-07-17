"""Live News — KI/CV News aus RSS-Feeds und arXiv."""
import re
from datetime import datetime, timezone

import feedparser
import requests
import streamlit as st

from src.components import hero, section_header, divider, info_box


# Kuratiertes Feed-Set — schnell & relevant
FEEDS = {
    "arXiv CV":          "https://rss.arxiv.org/rss/cs.CV",
    "arXiv ML":          "https://rss.arxiv.org/rss/cs.LG",
    "Hugging Face Blog": "https://huggingface.co/blog/feed.xml",
    "OpenAI News":       "https://openai.com/news/rss.xml",
    "DeepMind Blog":     "https://deepmind.google/blog/rss.xml",
    "Google Research":   "https://research.google/blog/rss/",
    "Anthropic News":    "https://www.anthropic.com/news/rss.xml",
    "MIT News AI":       "https://news.mit.edu/topic/mitartificial-intelligence2-rss.xml",
}

_HEADERS = {"User-Agent": "cv-ai-knowledge-lab/1.0 (+https://github.com/hackbert301009)"}
_TAG_RE = re.compile(r"<[^>]+>")


def _clean(text: str, limit: int = 280) -> str:
    """HTML entfernen und auf Laenge kuerzen."""
    text = _TAG_RE.sub("", text or "").strip()
    return text[:limit] + ("…" if len(text) > limit else "")


def _format_date(entry) -> str:
    """Publikationsdatum robust formatieren (struct_time bevorzugt)."""
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed:
        try:
            return datetime(*parsed[:6], tzinfo=timezone.utc).strftime("%d.%m.%Y")
        except (ValueError, TypeError):
            pass
    return entry.get("published", entry.get("updated", ""))


@st.cache_data(ttl=1800, show_spinner=False)  # 30 Minuten Cache — NUR Erfolge werden gecached
def _fetch_feed_cached(url: str, max_items: int) -> list[dict]:
    """Feed mit Timeout laden. Wirft bei Netz-/HTTP-Fehler (wird NICHT gecached)."""
    resp = requests.get(url, timeout=8, headers=_HEADERS)
    resp.raise_for_status()
    feed = feedparser.parse(resp.content)
    items = []
    for entry in feed.entries[:max_items]:
        items.append({
            "title":     entry.get("title", "Ohne Titel"),
            "link":      entry.get("link", "#"),
            "summary":   _clean(entry.get("summary", "")),
            "published": _format_date(entry),
        })
    return items


def fetch_feed(url: str, max_items: int = 10) -> tuple[list[dict], str | None]:
    """Feed laden. Gibt (items, fehlermeldung) zurueck. Fehler werden nicht gecached."""
    try:
        items = _fetch_feed_cached(url, max_items)
        if not items:
            return [], "Feed lieferte keine Eintraege (evtl. umgezogen oder leer)."
        return items, None
    except requests.Timeout:
        return [], "Zeitueberschreitung — Quelle antwortet nicht (Timeout 8 s)."
    except requests.RequestException as exc:
        return [], f"Netzwerk-/HTTP-Fehler: {exc}"
    except Exception as exc:  # pragma: no cover — feedparser-Randfaelle
        return [], f"Unerwarteter Fehler: {exc}"


def render():
    hero(
        eyebrow="Live",
        title="Live News",
        sub="Aktuelle Forschung, Releases und Industrienews aus der KI- und CV-Welt — "
            "direkt aus arXiv, OpenAI, DeepMind, HuggingFace & Co."
    )

    info_box(
        "Diese Seite zieht live von RSS-Feeds mit 8-Sekunden-Timeout. Beim ersten Aufruf kann es "
        "ein paar Sekunden dauern. Erfolgreiche Antworten werden 30 Minuten gecached, Fehler nicht.",
        kind="info",
    )

    selected = st.multiselect(
        "Quellen wählen",
        options=list(FEEDS.keys()),
        default=["arXiv CV", "Hugging Face Blog", "OpenAI News"],
    )

    max_items = st.slider("Max. Items pro Quelle", 3, 15, 5)

    if not selected:
        st.info("Wähle mindestens eine Quelle aus.")
        return

    if st.button("🔄 Feeds neu laden (Cache leeren)"):
        _fetch_feed_cached.clear()
        st.rerun()

    divider()

    seen_links: set[str] = set()
    for source in selected:
        section_header(f"📰 {source}")
        with st.spinner(f"Lade {source} …"):
            items, error = fetch_feed(FEEDS[source], max_items)

        if error:
            st.warning(f"⚠️ {source}: {error}")
            continue

        for it in items:
            # Duplikate ueber Quellen hinweg ueberspringen
            if it["link"] in seen_links and it["link"] != "#":
                continue
            seen_links.add(it["link"])
            with st.container():
                st.markdown(f"#### [{it['title']}]({it['link']})")
                if it["published"]:
                    st.caption(f"📅 {it['published']}")
                if it["summary"]:
                    st.markdown(it["summary"])
                st.markdown("---")

    divider()
    info_box(
        "Tipp: Speichere dir interessante Paper in Zotero oder einer Notiz-App. "
        "Die Halbwertszeit von KI-News ist kurz — was zählt, ist was du wirklich verstanden hast.",
        kind="tip",
    )
