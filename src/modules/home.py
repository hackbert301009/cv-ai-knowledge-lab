"""Startseite — Landing Page mit Hero, Stats und Modul-Grid."""
import streamlit as st
from src.components import hero, divider, section_header, stat_tile, card, render_card_grid, info_box
from src.registry import MODULES, modules_by_category


def render():
    hero(
        eyebrow="Open-Source · Master Edition",
        title="Computer Vision &amp; KI Knowledge Lab",
        sub="Ein vollständiges Wissenshub von Mathe-Grundlagen über klassische Bildverarbeitung "
            "bis zu Transformern, Diffusion Models und multimodaler KI — interaktiv, mit Code und Visualisierungen."
    )

    # ---------- Stats ----------
    cols = st.columns(4)
    stats = [
        ("22+", "Module"),
        ("150+", "Kapitel & Themen"),
        ("80+", "Code-Beispiele"),
        ("∞",  "Lernpotenzial"),
    ]
    for col, (num, label) in zip(cols, stats):
        col.markdown(stat_tile(num, label), unsafe_allow_html=True)

    divider()

    # ---------- Was macht dieses Lab besonders ----------
    section_header("Was du hier findest", "Sechs Kernbereiche, die zusammen ein vollständiges Bild ergeben.")

    cards = [
        card("📐", "Mathematische Fundamente", "Lineare Algebra, Analysis, Wahrscheinlichkeit — verständlich erklärt mit Anwendungsbezug.", ["Grundlagen"], ["green"]),
        card("🖼️", "Klassische Bildverarbeitung", "Filter, Faltung, Kantenerkennung, Features, Segmentierung — die zeitlosen Basics.", ["CV-Klassik"], ["blue"]),
        card("🤖", "Deep Learning Tiefe", "Von Perzeptron bis ConvNeXt: alles was du über CNNs und Training wissen musst.", ["Deep Learning"], ["amber"]),
        card("⚡", "Transformer &amp; Attention", "ViT, Swin, DINO — die Architektur, die alles verändert hat.", ["State-of-the-Art"], ["pink"]),
        card("🌊", "Generative &amp; Diffusion KI", "Stable Diffusion, GANs, VAEs, Flow Matching — wie Maschinen kreativ werden.", ["State-of-the-Art"], ["pink"]),
        card("🌐", "Multimodal &amp; VLMs", "CLIP, LLaVA, GPT-4o, Sora — Bild, Text, Video, Audio in einem Modell.", ["Modern"], ["pink"]),
    ]
    render_card_grid(cards, cols=3)

    divider()

    # ---------- Lernpfad-Vorschau ----------
    section_header("Empfohlener Lernpfad", "Vier Phasen, vom Anfänger zum Experten.")

    pcols = st.columns(4)
    phases = [
        ("1️⃣", "Mathematik &amp; Pixel", "Lineare Algebra, Analysis, Wahrscheinlichkeit. Verstehen, was ein Bild eigentlich ist.", ["math", "image_basics"]),
        ("2️⃣", "Klassische CV",          "Faltung, Kanten, Features. Algorithmen, die seit Jahrzehnten funktionieren.",         ["filters", "edges", "features"]),
        ("3️⃣", "Deep Learning",          "Neuronale Netze, CNNs, Training. Die Brücke zur modernen KI.",                          ["nn_basics", "cnn", "training"]),
        ("4️⃣", "State-of-the-Art",       "Transformer, Diffusion, VLMs. Was 2024–2026 gerade passiert.",                          ["transformers", "diffusion", "vlm"]),
    ]
    for col, (num, title, desc, _) in zip(pcols, phases):
        with col:
            st.markdown(card(num, title, desc), unsafe_allow_html=True)

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
                        card(mod.icon, mod.title, mod.short, [mod.level, mod.duration],
                             ["green" if mod.level == "Anfänger" else "amber" if mod.level == "Fortgeschritten" else "pink", "blue"]),
                        unsafe_allow_html=True,
                    )

    divider()

    info_box(
        "Tipp: Du musst nicht der Reihenfolge folgen. Jedes Modul ist eigenständig — "
        "spring rein, wo dich gerade etwas interessiert. Die Sidebar links ist dein Wegweiser.",
        kind="tip",
    )
