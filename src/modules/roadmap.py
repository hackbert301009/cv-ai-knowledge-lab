"""Roadmap & Lernpfad — strukturierter Weg durch alle Module."""
import streamlit as st
from src.components import hero, section_header, divider, info_box, card, render_card_grid


def render():
    hero(
        eyebrow="Roadmap",
        title="Dein Weg zum CV &amp; KI Experten",
        sub="Eine durchdachte Reihenfolge — mit Zeitschätzungen und Meilensteinen. "
            "Egal ob du 4 Wochen oder 6 Monate Zeit hast: hier findest du deinen Pfad."
    )

    # ----- Zeitpläne -----
    section_header("Wähle dein Tempo")
    pcols = st.columns(3)
    plans = [
        ("⚡", "Sprint (4 Wochen)", "Für Eilige mit Vorkenntnissen. Fokus auf Deep Learning + Transformers. Ca. 2h/Tag.", ["intensiv"]),
        ("🎯", "Standard (12 Wochen)", "Empfohlen. Alle Phasen mit Praxis. Ca. 1h/Tag, ein Wochenend-Projekt pro Monat.", ["balanced"]),
        ("🌳", "Tiefgang (6 Monate)", "Vollständig. Auch alle Mathe-Beweise, Paper-Studium, eigene Implementierungen.", ["mastery"]),
    ]
    for col, (icon, title, desc, tags) in zip(pcols, plans):
        with col:
            st.markdown(card(icon, title, desc, tags, ["amber"]), unsafe_allow_html=True)

    divider()

    # ----- Phase-für-Phase -----
    section_header("Phase 1 — Fundamente", "Wochen 1–3 · Basics solide aufbauen")
    st.markdown(
        """
        Hier geht es um das **Warum**. Ohne Mathe verstehst du nicht, wieso Backpropagation funktioniert,
        warum CNNs translation-equivariant sind, oder was eine Verlustfunktion eigentlich misst.
        """
    )
    render_card_grid([
        card("📐", "Mathe-Crashkurs",      "Notation, was du wirklich brauchst.",         ["1 Woche"], ["blue"]),
        card("🧮", "Lineare Algebra",       "Tensoren = Bilder. Matrixmult = NN-Forward.", ["1 Woche"], ["blue"]),
        card("∂",  "Analysis",              "Gradienten, die Sprache der Optimierung.",    ["3 Tage"], ["blue"]),
        card("🎲", "Wahrscheinlichkeit",   "Bayes, Cross-Entropy, KL-Divergenz.",         ["4 Tage"], ["blue"]),
    ], cols=4)

    divider()

    section_header("Phase 2 — Klassische Bildverarbeitung", "Wochen 4–6 · Pixel-Level Verständnis")
    st.markdown(
        """
        Bevor wir neuronale Netze auf Bilder loslassen, sollten wir verstehen, **was Bilder sind**
        und welche Operationen seit den 60er Jahren funktionieren. Diese Algorithmen leben in
        OpenCV, in Smartphone-Kameras, in der Medizinbildgebung — überall.
        """
    )
    render_card_grid([
        card("🖼️", "Bildgrundlagen",     "Pixel, Farbräume, Sampling, Quantisierung.",      ["2 Tage"],   ["blue"]),
        card("🌫️", "Filter & Faltung",   "Gauß, Median — die Mutter aller CNNs.",           ["3 Tage"],   ["blue"]),
        card("📏", "Kantendetektion",    "Sobel, Canny — wie Computer Konturen sehen.",     ["2 Tage"],   ["blue"]),
        card("🔑", "Features",           "SIFT/ORB — wie man Bilder eindeutig beschreibt.", ["3 Tage"],   ["blue"]),
        card("🧱", "Morphologie",        "Erosion, Dilatation — Form-Operationen.",         ["1 Tag"],    ["blue"]),
        card("✂️", "Segmentierung",      "Threshold, Watershed, GrabCut.",                  ["3 Tage"],   ["blue"]),
    ], cols=3)

    divider()

    section_header("Phase 3 — Deep Learning", "Wochen 7–9 · Die Brücke zur modernen KI")
    st.markdown(
        """
        Jetzt wird es interessant. Wir lernen wie ein Netzwerk **lernt**, warum Convolution
        so gut zu Bildern passt, und wie man Modelle trainiert, ohne dass sie zerbrechen.
        """
    )
    render_card_grid([
        card("🧠", "Neuronale Netze",     "Perzeptron → MLP → Backprop.",            ["1 Woche"], ["amber"]),
        card("🔲", "CNNs",                "LeNet, AlexNet, VGG, ResNet.",            ["1 Woche"], ["amber"]),
        card("🎯", "Training & Optimizer","SGD, Adam, Scheduling, Regularisierung.", ["1 Woche"], ["amber"]),
        card("🏛️", "Moderne Archs",       "EfficientNet, ConvNeXt, U-Net.",          ["3 Tage"],  ["amber"]),
    ], cols=4)

    divider()

    section_header("Phase 4 — State of the Art", "Wochen 10–12 · Was 2024–2026 wirklich passiert")
    st.markdown(
        """
        Die spannendste Phase. Transformers haben CV revolutioniert, Diffusion Models haben
        generative KI auf ein neues Level gehoben, und VLMs verbinden Sehen und Sprache.
        """
    )
    render_card_grid([
        card("⚡", "Transformer & ViT",     "Attention is all you need — und in Bildern auch.", ["1 Woche"], ["pink"]),
        card("👁️‍🗨️", "VLMs",               "CLIP, BLIP-2, LLaVA. Bild ↔ Text.",                ["4 Tage"],  ["pink"]),
        card("🌊", "Diffusion",             "DDPM, Stable Diffusion, Flow Matching.",           ["1 Woche"], ["pink"]),
        card("🎨", "Generative KI",          "GANs, VAEs — die Vorgänger.",                      ["3 Tage"],  ["pink"]),
        card("🌐", "Multimodal",             "GPT-4o, Gemini, Sora.",                            ["3 Tage"],  ["pink"]),
    ], cols=3)

    divider()

    section_header("Phase 5 — Praxis", "Ongoing · Machen statt nur lesen")
    render_card_grid([
        card("💻", "Praxisprojekte",  "10+ Hands-on Projekte mit komplettem Code.",     ["varies"], ["green"]),
        card("📦", "Datasets",        "Wo du saubere Daten findest und wie du sie nutzt.", ["2 Tage"], ["green"]),
        card("🚀", "Deployment",      "Vom Notebook zur Produktions-API.",               ["1 Woche"], ["green"]),
    ], cols=3)

    divider()

    info_box(
        "Konsistenz schlägt Intensität. Lieber jeden Tag 30 Minuten als einmal pro Woche 5 Stunden. "
        "Code mit. Erkläre einer anderen Person, was du gelernt hast — das ist der ultimative Test.",
        kind="tip",
    )
