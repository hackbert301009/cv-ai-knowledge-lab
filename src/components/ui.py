"""
Wiederverwendbare UI-Komponenten und Layout-Helpers.
"""
import streamlit as st
from pathlib import Path
from typing import Sequence


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


def lab_header(title: str, desc: str = ""):
    """Header für interaktive Lab-Sektionen mit farbiger Markierung."""
    st.markdown(
        f"""
        <div class="lab-header">
          <span class="lab-pill">🧪 Lab</span>
          <h3 style="margin:0.5rem 0 0.25rem 0; font-weight:700;">{title}</h3>
          {f'<p style="color:#9CA3AF; margin:0; font-size:0.9rem;">{desc}</p>' if desc else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def video_embed(youtube_id: str, title: str = "Lernvideo", caption: str = ""):
    """Bettet ein YouTube-Video als responsiven iframe ein."""
    st.markdown(
        f"""
        <div class="video-wrap">
          <iframe
            src="https://www.youtube-nocookie.com/embed/{youtube_id}?rel=0&modestbranding=1"
            title="{title}"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowfullscreen
            loading="lazy">
          </iframe>
        </div>
        {f'<p class="video-caption">{caption}</p>' if caption else ''}
        """,
        unsafe_allow_html=True,
    )


def key_concept(emoji: str, term: str, explanation: str):
    """Hervorgehobener Schlüsselbegriff mit Erklärung."""
    st.markdown(
        f"""
        <div class="key-concept">
          <span class="key-emoji">{emoji}</span>
          <div>
            <div class="key-term">{term}</div>
            <div class="key-exp">{explanation}</div>
          </div>
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


def math_box(latex: str, label: str = ""):
    """Visuell hervorgehobene Formel-Box."""
    lbl = f'<div class="math-label">{label}</div>' if label else ""
    st.markdown(
        f"""
        <div class="math-box">
          {lbl}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.latex(latex)


def step_list(steps: list[tuple[str, str]]):
    """Nummerierte Schritt-Liste mit Titel + Beschreibung."""
    items = ""
    for i, (title, desc) in enumerate(steps, 1):
        items += f"""
        <div class="step-item">
          <div class="step-num">{i}</div>
          <div>
            <div class="step-title">{title}</div>
            <div class="step-desc">{desc}</div>
          </div>
        </div>
        """
    st.markdown(f'<div class="step-list">{items}</div>', unsafe_allow_html=True)


def _as_markdown_list(items: Sequence[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def render_learning_block(
    *,
    key_prefix: str,
    section_title: str = "Lernpfad & Übungen",
    section_sub: str = "Guided -> Challenge -> Debug -> Mini-Projekt",
    progression: list[tuple[str, str, str, str, str]] | None = None,
    mcq_question: str | None = None,
    mcq_options: list[str] | None = None,
    mcq_correct_option: str | None = None,
    mcq_success_message: str = "Richtig.",
    mcq_retry_message: str = "Nicht korrekt. Prüfe den Abschnitt erneut.",
    open_question: str | None = None,
    code_task: str | None = None,
    code_language: str = "python",
    community_rows: list[dict[str, str]] | None = None,
    cheat_sheet: list[str] | None = None,
    key_takeaways: list[str] | None = None,
    common_errors: list[str] | None = None,
):
    """Rendert einen standardisierten Lernblock für Module."""
    section_header(section_title, section_sub)

    if progression:
        cards = [
            card(icon, title, desc, [level], [color])
            for icon, title, desc, level, color in progression
        ]
        render_card_grid(cards, cols=min(4, max(1, len(cards))))

    st.markdown("### Mischformat-Übung")
    if mcq_question and mcq_options and mcq_correct_option:
        selected = st.radio(mcq_question, mcq_options, key=f"{key_prefix}_mcq")
        if st.button("Antwort prüfen", key=f"{key_prefix}_mcq_check"):
            if selected == mcq_correct_option:
                st.success(mcq_success_message)
            else:
                st.warning(mcq_retry_message)

    if open_question:
        st.text_area(open_question, key=f"{key_prefix}_open_question", height=90)

    if code_task:
        st.code(code_task, language=code_language)

    if community_rows:
        st.markdown("### Community & Peer-Feedback")
        st.table(community_rows)

    if cheat_sheet:
        st.markdown("### Cheat Sheet")
        st.markdown(_as_markdown_list(cheat_sheet))

    if key_takeaways:
        st.markdown("### Key Takeaways")
        st.markdown(_as_markdown_list(key_takeaways))

    if common_errors:
        st.markdown("### Häufige Fehler")
        st.markdown("\n".join(f"{idx}. {item}" for idx, item in enumerate(common_errors, 1)))


def render_quiz_checkpoint(
    *,
    key_prefix: str,
    module_id: str | None = None,
    title: str = "Quiz & Checkpoint",
    question: str,
    options: list[str],
    correct_option: str,
    checklist: list[str] | None = None,
    capstone_prompt: str | None = None,
):
    """Standardisierter Abschlussblock mit Quiz, Selbstcheck und Mini-Transfer."""
    section_header(title, "Verstehe ich das Thema wirklich?")
    tracked_module_id = module_id or key_prefix
    st.session_state.setdefault("quiz_completed_modules", [])
    quiz_done = tracked_module_id in st.session_state.get("quiz_completed_modules", [])

    if quiz_done:
        st.success("Checkpoint bereits bestanden. Du kannst direkt weitermachen oder erneut testen.")

    selected = st.radio(question, options, key=f"{key_prefix}_quiz_choice")
    if st.button("Quiz prüfen", key=f"{key_prefix}_quiz_check"):
        if selected == correct_option:
            completed = list(st.session_state.get("quiz_completed_modules", []))
            if tracked_module_id not in completed:
                completed.append(tracked_module_id)
            st.session_state.quiz_completed_modules = completed
            st.success("Richtig. Du kannst zum naechsten Modul weitergehen.")
        else:
            st.warning("Noch nicht korrekt. Wiederhole die Kerngedanken und probiere es erneut.")

    if checklist:
        st.markdown("### Checkpoint")
        for idx, item in enumerate(checklist):
            st.checkbox(item, key=f"{key_prefix}_check_{idx}")

    if capstone_prompt:
        st.markdown("### Mini-Transfer")
        st.text_area(
            "Formuliere kurz deinen Loesungsansatz",
            value=capstone_prompt,
            height=120,
            key=f"{key_prefix}_transfer",
        )
