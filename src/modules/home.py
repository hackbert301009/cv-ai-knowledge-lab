"""Startseite — Landing Page mit Hero, Stats, Modul-Grid und Featured Videos."""
import streamlit as st
from src.components import (
    hero, divider, section_header, stat_tile, card, render_card_grid,
    info_box, video_embed, key_concept,
)
from src.registry import MODULES, modules_by_category, get_module


def render():
    hero(
        eyebrow="Open-Source · Master Edition 2026",
        title="Computer Vision &amp; KI Knowledge Lab",
        sub="Ein vollständiges Wissenshub von Mathe-Grundlagen über klassische Bildverarbeitung "
            "bis zu Transformern, Diffusion Models und multimodaler KI — interaktiv, mit Code, "
            "Visualisierungen und direkten Lernvideos."
    )

    # ---------- Stats ----------
    cols = st.columns(4)
    stats = [
        (str(len(MODULES)), "Module"),
        ("200+", "Kapitel & Themen"),
        ("100+", "Code-Beispiele"),
        ("∞",   "Lernpotenzial"),
    ]
    for col, (num, label) in zip(cols, stats):
        col.markdown(stat_tile(num, label), unsafe_allow_html=True)

    divider()

    # ---------- Personal Hub ----------
    section_header("Dein Personal Hub", "Fortschritt, Favoriten und schneller Wiedereinstieg.")
    trackable_ids = [m.id for m in MODULES if m.id != "home"]
    completed = set(st.session_state.get("completed_modules", []))
    favorites = st.session_state.get("favorite_modules", [])
    recent = st.session_state.get("visited_modules", [])
    done_count = len(completed.intersection(trackable_ids))
    progress_pct = int((done_count / max(1, len(trackable_ids))) * 100)

    hcols = st.columns(4)
    hub_stats = [
        (f"{progress_pct}%", "Lernfortschritt"),
        (str(done_count), "Abgeschlossen"),
        (str(len(favorites)), "Favoriten"),
        (str(len(recent)), "Zuletzt besucht"),
    ]
    for col, (num, label) in zip(hcols, hub_stats):
        col.markdown(stat_tile(num, label), unsafe_allow_html=True)

    st.progress(progress_pct / 100)

    last_id = st.session_state.get("last_module", "home")
    last_mod = get_module(last_id)
    if last_mod is not None and last_mod.id != "home":
        if st.button(f"▶️ Weiterlernen: {last_mod.icon} {last_mod.title}", key="home_resume_last"):
            st.session_state.current_module = last_mod.id
            st.rerun()

    q1, q2 = st.columns(2)
    with q1:
        st.markdown("**⭐ Deine Favoriten**")
        if favorites:
            for fav_id in favorites[:4]:
                fav = get_module(fav_id)
                if fav is None:
                    continue
                if st.button(f"{fav.icon} {fav.title}", key=f"home_fav_{fav_id}", use_container_width=True):
                    st.session_state.current_module = fav_id
                    st.rerun()
        else:
            st.caption("Noch keine Favoriten markiert.")
    with q2:
        st.markdown("**🕘 Zuletzt besucht**")
        if recent:
            for rec_id in recent[:4]:
                rec = get_module(rec_id)
                if rec is None:
                    continue
                if st.button(f"{rec.icon} {rec.title}", key=f"home_recent_{rec_id}", use_container_width=True):
                    st.session_state.current_module = rec_id
                    st.rerun()
        else:
            st.caption("Noch keine Module besucht.")

    divider()

    # ---------- Quick-Start Guide ----------
    section_header("Dein Einstieg", "In 3 Schritten vom Nullwissen zum funktionierenden Code.")

    qs_cols = st.columns(3)
    qs_steps = [
        ("1️⃣", "Wähle deinen Einstiegspunkt",
         "Bist du Anfänger → starte mit **Mathe-Crashkurs**. "
         "Hast du ML-Grundlagen → spring direkt zu **CNNs**. "
         "Kennst Du Deep Learning → geh direkt zu **Transformern** oder **Diffusion**.",
         "green"),
        ("2️⃣", "Lerne interaktiv",
         "Jedes Modul hat interaktive Labs mit Schiebereglern und Live-Demos. "
         "Verändere Parameter und sieh sofort das Ergebnis. "
         "Keine Installation nötig — alles läuft im Browser.",
         "blue"),
        ("3️⃣", "Schau die Videos",
         "Jedes Modul enthält kuratierte Videos von 3Blue1Brown, Andrej Karpathy, "
         "Computerphile und anderen Topexperten — direkt eingebettet, keine Suche nötig.",
         "amber"),
    ]
    for col, (num, title, desc, color) in zip(qs_cols, qs_steps):
        with col:
            st.markdown(card(num, title, desc, [], []), unsafe_allow_html=True)

    divider()

    # ---------- Was macht dieses Lab besonders ----------
    section_header("Was du hier findest", "Sechs Kernbereiche, die zusammen ein vollständiges Bild ergeben.")

    cards = [
        card("📐", "Mathematische Fundamente",
             "Lineare Algebra, Analysis, Wahrscheinlichkeit — verständlich erklärt mit Anwendungsbezug und interaktiven Visualisierungen.",
             ["Grundlagen"], ["green"]),
        card("🖼️", "Klassische Bildverarbeitung",
             "Filter, Faltung, Kantenerkennung, Features, Segmentierung — die zeitlosen Basics mit interaktiven Kernel-Editoren.",
             ["CV-Klassik"], ["blue"]),
        card("🤖", "Deep Learning Tiefe",
             "Von Perzeptron bis ConvNeXt: alles über CNNs, Training, Backpropagation mit Live-Demos und Loss-Visualisierungen.",
             ["Deep Learning"], ["amber"]),
        card("⚡", "Transformer &amp; Attention",
             "ViT, Swin, DINO — die Architektur, die alles verändert hat. Mit interaktiver Attention-Matrix.",
             ["State-of-the-Art"], ["pink"]),
        card("🌊", "Generative &amp; Diffusion KI",
             "Stable Diffusion, FLUX, GANs, VAEs — wie Maschinen kreativ werden. Mit Noise-Schedule-Demo.",
             ["Generativ"], ["pink"]),
        card("🌐", "Multimodal &amp; VLMs",
             "CLIP, LLaVA, GPT-4o, Sora — Bild, Text, Video, Audio in einem Modell. Aktueller Stand 2026.",
             ["Modern"], ["pink"]),
    ]
    render_card_grid(cards, cols=3)

    divider()

    # ---------- Featured Videos ----------
    section_header("Empfohlene Einstiegs-Videos",
                   "Die drei besten kostenlosen Videos für einen schnellen Überblick.")

    v1, v2, v3 = st.columns(3)
    with v1:
        st.markdown("**🧠 Was ist ein neuronales Netz?**")
        video_embed("aircAruvnKk", "But what is a neural network? — 3Blue1Brown",
                    "3Blue1Brown · ~19 Min · Pflichtanschauen")
    with v2:
        st.markdown("**👁️ Wie CNNs funktionieren**")
        video_embed("FmpDIaiMIeA", "How CNNs work",
                    "Brandon Rohrer · ~17 Min · Sehr verständlich")
    with v3:
        st.markdown("**⚡ Attention & Transformer**")
        video_embed("eMlx5fFNoYc", "Attention in Transformers — 3Blue1Brown",
                    "3Blue1Brown · ~26 Min · Neue Serie 2024")

    divider()

    # ---------- Lernpfad-Vorschau ----------
    section_header("Empfohlener Lernpfad", "Vier Phasen, vom Anfänger zum Experten.")

    pcols = st.columns(4)
    phases = [
        ("1️⃣", "Mathematik &amp; Pixel",
         "Lineare Algebra, Analysis, Wahrscheinlichkeit. Verstehen, was ein Bild eigentlich ist.",
         "~3–5h"),
        ("2️⃣", "Klassische CV",
         "Faltung, Kanten, Features, Segmentierung. Algorithmen, die seit Jahrzehnten funktionieren.",
         "~4–6h"),
        ("3️⃣", "Deep Learning",
         "Neuronale Netze, CNNs, Training. Die Brücke zur modernen KI. Eigene Netze schreiben.",
         "~8–12h"),
        ("4️⃣", "State-of-the-Art",
         "Transformer, Diffusion, VLMs. Was 2024–2026 passiert und wie du es nutzt.",
         "~6–10h"),
    ]
    for col, (num, title, desc, time) in zip(pcols, phases):
        with col:
            st.markdown(card(num, title, f"{desc} <br><br>**Zeitaufwand:** {time}"),
                        unsafe_allow_html=True)

    divider()

    # ---------- Schlüssel-Konzepte ----------
    section_header("Schlüssel-Ideen", "Die wichtigsten Konzepte auf einen Blick.")

    kc1, kc2 = st.columns(2)
    with kc1:
        key_concept("🔲", "Faltung / Convolution",
                    "Ein Kernel gleitet über ein Bild und berechnet gewichtete Summen. "
                    "Basis von CNNs — und von der gesamten modernen Computer Vision.")
        key_concept("⬅️", "Backpropagation",
                    "Kettenregel der Analysis angewendet auf tiefe Netze. "
                    "Gradienten fließen rückwärts und passen Gewichte an. Seit 1986 unverändert.")
        key_concept("⚡", "Self-Attention",
                    "Jedes Element einer Sequenz kommuniziert direkt mit jedem anderen. "
                    "Basis aller modernen LLMs und Vision Transformers.")
    with kc2:
        key_concept("🌊", "Diffusion",
                    "Bilder schrittweise verrauschen, dann das Rückwärts-Modell lernen. "
                    "Basis von Stable Diffusion, DALL·E 3, Midjourney.")
        key_concept("📐", "Lineare Algebra",
                    "Matrizen, Vektoren, Eigenwerte — die Sprache, in der Deep Learning formuliert ist. "
                    "Jedes Forward-Pass ist im Kern Matrixmultiplikation.")
        key_concept("📊", "Loss + Gradient Descent",
                    "Fehler messen → Gradient berechnen → Parameter anpassen. "
                    "Die universelle Lernschleife aller neuronalen Netze.")

    divider()

    # ---------- Modul-Übersicht nach Kategorie ----------
    section_header("Alle Module im Überblick", f"{len(MODULES)} Module, gruppiert nach Themenfeld.")
    grouped = modules_by_category()
    for cat, mods in grouped.items():
        with st.expander(f"**{cat}** — {len(mods)} Module", expanded=(cat == "Übersicht")):
            mcols = st.columns(2)
            for i, mod in enumerate(mods):
                with mcols[i % 2]:
                    st.markdown(
                        card(mod.icon, mod.title, mod.short,
                             [mod.level, mod.duration],
                             ["green" if mod.level == "Anfänger"
                              else "amber" if mod.level == "Fortgeschritten" else "pink",
                              "blue"]),
                        unsafe_allow_html=True,
                    )

    divider()

    info_box(
        "**Tipp:** Du musst nicht der Reihenfolge folgen. Jedes Modul ist eigenständig — "
        "spring rein, wo dich gerade etwas interessiert. Die Sidebar links ist dein Wegweiser. "
        "Nutze die 🔍 Suche, um Themen schnell zu finden.",
        kind="tip",
    )
