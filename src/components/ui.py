"""
Wiederverwendbare UI-Komponenten und Layout-Helpers.
"""
import streamlit as st
from pathlib import Path


def inject_css():
    """Globales CSS einbinden — einmal pro Session."""
    css_path = Path(__file__).parent.parent.parent / "assets" / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def hero(eyebrow: str, title: str, sub: str = ""):
    """Großer Hero-Header mit Gradient-Title."""
    st.markdown(
        f"""
        <div style="margin: 0.5rem 0 2rem 0;">
          <span class="eyebrow">{eyebrow}</span>
          <div class="hero-title">{title}</div>
          <p class="hero-sub">{sub}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, sub: str = ""):
    """Sektions-Header (kleiner als Hero)."""
    st.markdown(
        f"""
        <div style="margin: 1.5rem 0 1rem 0;">
          <h2 style="font-weight:700; letter-spacing:-0.02em; margin-bottom:0.3rem;">{title}</h2>
          {f'<p style="color:#9CA3AF; margin:0;">{sub}</p>' if sub else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def divider():
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


def stat_tile(num: str, label: str):
    return f"""
    <div class="stat-tile">
      <div class="stat-num">{num}</div>
      <div class="stat-label">{label}</div>
    </div>
    """


def card(icon: str, title: str, desc: str, tags: list[str] | None = None, tag_colors: list[str] | None = None) -> str:
    """HTML für eine Karte. Mehrere Karten am besten in cols rendern."""
    tags = tags or []
    tag_colors = tag_colors or []
    tag_html = ""
    for i, tag in enumerate(tags):
        color_cls = tag_colors[i] if i < len(tag_colors) else ""
        tag_html += f'<span class="card-tag {color_cls}">{tag}</span>'
    return f"""
    <div class="card">
      <span class="card-icon">{icon}</span>
      <div class="card-title">{title}</div>
      <div class="card-desc">{desc}</div>
      <div>{tag_html}</div>
    </div>
    """


def render_card_grid(cards: list[str], cols: int = 3):
    """Liste von Karten-HTMLs in einem Grid rendern."""
    columns = st.columns(cols)
    for i, card_html in enumerate(cards):
        with columns[i % cols]:
            st.markdown(card_html, unsafe_allow_html=True)


def info_box(text: str, kind: str = "info"):
    """Hübsche Info-Box. kind: info | success | warn | tip."""
    palette = {
        "info":    ("#3B82F6", "rgba(59, 130, 246, 0.1)",   "ℹ️"),
        "success": ("#10B981", "rgba(16, 185, 129, 0.1)",    "✅"),
        "warn":    ("#F59E0B", "rgba(245, 158, 11, 0.1)",    "⚠️"),
        "tip":     ("#A78BFA", "rgba(167, 139, 250, 0.1)",   "💡"),
    }
    color, bg, icon = palette.get(kind, palette["info"])
    st.markdown(
        f"""
        <div style="
            background:{bg};
            border-left:3px solid {color};
            border-radius:8px;
            padding:1rem 1.2rem;
            margin:1rem 0;
            color:#E5E7EB;">
          <strong>{icon} </strong>{text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def level_badge(level: str) -> str:
    colors = {"Anfänger": "green", "Fortgeschritten": "amber", "Experte": "pink"}
    return f'<span class="card-tag {colors.get(level, "")}">{level}</span>'
