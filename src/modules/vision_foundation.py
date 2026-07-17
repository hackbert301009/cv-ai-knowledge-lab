"""Vision Foundation Models — SAM 2, DINOv3, Depth Anything, CLIP, Grounding."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.components import (
    hero, section_header, divider, info_box, lab_header, key_concept, card, render_card_grid,
    video_embed, render_learning_block, render_quiz_checkpoint,
)

# Spielzeug-Semantikraum: Achsen = [Tier, Fahrzeug, Natur, Gebäude, Essen]
_AXES = ["Tier", "Fahrzeug", "Natur", "Gebäude", "Essen"]
_IMAGES = {
    "🐱 Katze auf Sofa":   [1.0, 0.0, 0.0, 0.15, 0.0],   # Sofa → etwas "Gebäude/Interieur"
    "🚗 Auto auf Straße":  [0.0, 1.0, 0.1, 0.0, 0.0],
    "🏔️ Berglandschaft":   [0.0, 0.0, 1.0, 0.0, 0.0],
    "🍕 Pizza auf Tisch":  [0.0, 0.0, 0.0, 0.15, 1.0],
    "🏛️ Gebäude in Stadt": [0.0, 0.1, 0.0, 1.0, 0.0],
}
_LABELS = {
    "ein Foto einer Katze":   [1.0, 0.0, 0.0, 0.0, 0.0],
    "ein Foto eines Hundes":  [0.9, 0.0, 0.1, 0.0, 0.0],
    "ein Foto eines Autos":   [0.0, 1.0, 0.0, 0.0, 0.0],
    "ein Foto eines Berges":  [0.0, 0.0, 1.0, 0.0, 0.0],
    "ein Foto einer Pizza":   [0.0, 0.0, 0.0, 0.0, 1.0],
    "ein Foto eines Gebäudes":[0.0, 0.0, 0.0, 1.0, 0.0],
}


def _cos(a, b):
    a, b = np.array(a, float), np.array(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def render():
    hero(
        eyebrow="State-of-the-Art · Vision Foundation Models",
        title="Vision Foundation Models",
        sub="Groß vortrainierte Modelle, die ohne aufgabenspezifisches Training funktionieren: "
            "SAM 2, DINOv3, Depth Anything, Grounding DINO — und CLIP als Bindeglied zu Sprache."
    )

    tabs = st.tabs([
        "🧭 Konzept", "✂️ SAM 2", "🦖 DINOv3", "📏 Depth Anything",
        "🎯 Grounding & Open-Vocab", "🧪 CLIP Zero-Shot Lab", "🎬 Lernvideos",
    ])

    with tabs[0]:
        section_header("Was ist ein Foundation Model?")
        st.markdown(r"""
Ein **Vision Foundation Model** wird einmal auf riesigen, oft unlabelierten Datenmengen vortrainiert
und ist danach **breit übertragbar** — häufig **zero-shot**, ohne weiteres Training.

#### Drei Eigenschaften
1. **Skalierung** — Daten + Compute + Parameter im großen Maßstab.
2. **Generalität** — eine Basis für viele Downstream-Tasks.
3. **Prompt-/Transfer-basiert** — anpassbar per Prompt, Linear Probe oder leichtem Fine-Tuning.
        """)
        cards = [
            card("✂️", "SAM 2", "Promptbare Segmentierung für Bild & Video (Meta).", ["Segmentierung"], ["pink"]),
            card("🦖", "DINOv3", "Selbstüberwachte Features ohne Labels (Meta).", ["Features"], ["pink"]),
            card("📏", "Depth Anything", "Monokulare Tiefe, starke Zero-Shot-Generalisierung.", ["Tiefe"], ["pink"]),
            card("🎯", "Grounding DINO", "Objekterkennung per Textprompt (Open-Vocabulary).", ["Detection"], ["pink"]),
            card("🔗", "CLIP", "Gemeinsamer Bild-Text-Raum, Basis für Zero-Shot.", ["VLM"], ["amber"]),
            card("🌊", "Stable Diffusion VAE", "Encoder/Decoder als Baustein generativer Modelle.", ["Generativ"], ["amber"]),
        ]
        render_card_grid(cards, cols=3)

    with tabs[1]:
        section_header("SAM 2 — Segment Anything (Model 2)")
        st.markdown(r"""
**SAM** (2023) und **SAM 2** (2024, Meta) segmentieren **alles** — promptbar über Punkte, Boxen oder Masken.

- **Zero-Shot**: kein Training auf deine Klassen nötig.
- **Promptbar**: Klick auf ein Objekt → Maske.
- **SAM 2**: erweitert auf **Video** mit zeitlich konsistenten Masken (Memory-Mechanismus).
- Anwendungen: Annotation-Beschleunigung, Bild-/Videobearbeitung, Robotik.
        """)
        info_box("SAM ersetzt keinen Klassifikator — es sagt *wo* ein Objekt ist, nicht *was* es ist. "
                 "Kombiniere es mit CLIP/Grounding DINO für benannte Objekte.", kind="tip")

    with tabs[2]:
        section_header("DINOv2 / DINOv3 — selbstüberwachte Features")
        st.markdown(r"""
**DINO** lernt starke Bildfeatures **ohne Labels** (Self-Distillation, siehe Modul Self-Supervised).

- **DINOv2/v3** (Meta) liefern universelle Features für Klassifikation, Segmentierung, Retrieval, Tiefe.
- Oft reicht ein **linearer Probe** obendrauf — kein volles Fine-Tuning.
- Emergente Eigenschaft: Attention-Maps segmentieren Objekte ohne jede Segmentierungs-Supervision.
        """)

    with tabs[3]:
        section_header("Depth Anything — monokulare Tiefe")
        st.markdown(r"""
**Depth Anything** (V1 2024, V2) schätzt relative **Tiefe aus einem einzelnen Bild** mit
beeindruckender Zero-Shot-Generalisierung — trainiert auf riesigen Mengen (auch pseudo-gelabelter) Daten.

- Input: ein RGB-Bild → Output: dichte Tiefenkarte.
- Nützlich für 3D-Effekte, Robotik, Bokeh, AR.
        """)

    with tabs[4]:
        section_header("Grounding & Open-Vocabulary")
        st.markdown(r"""
**Open-Vocabulary**-Modelle erkennen Objekte, die **nicht** in einem festen Klassenkatalog stehen —
gesteuert per **Textprompt**.

- **Grounding DINO**: "finde die *rote Tasse*" → Bounding Box.
- **Grounded-SAM**: Grounding DINO (Box) + SAM (präzise Maske) = benannte Segmentierung.
- Basis ist ein **gemeinsamer Bild-Text-Raum** (CLIP-artig).
        """)
        key_concept("🔓", "Open-Vocabulary",
                    "Nicht auf feste Klassen beschränkt — beliebige Begriffe per Sprache abfragbar.")
        key_concept("🎯", "Grounding",
                    "Sprachbegriff an eine konkrete Bildregion 'erden' (lokalisieren).")

    # ── Tab 5: CLIP Zero-Shot Lab ────────────────────────────────────────────
    with tabs[5]:
        lab_header("CLIP Zero-Shot Klassifikation", "Wähle ein Bild und eine Menge Text-Labels — CLIP ordnet per Kosinusähnlichkeit zu.")
        st.caption("Vereinfachte Simulation in einem 5-dimensionalen Semantikraum "
                   f"({', '.join(_AXES)}) — echtes CLIP nutzt ~512+ Dimensionen.")
        img_name = st.selectbox("Bild", list(_IMAGES.keys()))
        labels = st.multiselect("Kandidaten-Labels (frei wählbar = Open-Vocabulary)",
                                 list(_LABELS.keys()), default=list(_LABELS.keys()))
        if not labels:
            st.info("Wähle mindestens ein Label.")
        else:
            img_vec = _IMAGES[img_name]
            sims = np.array([_cos(img_vec, _LABELS[l]) for l in labels])
            probs = np.exp(sims / 0.1)
            probs = probs / probs.sum()   # Softmax mit Temperatur (wie CLIP)

            order = np.argsort(probs)[::-1]
            fig = go.Figure(go.Bar(
                x=[probs[i] for i in order],
                y=[labels[i] for i in order],
                orientation="h",
                marker_color=["#7C3AED" if k == 0 else "#4B5563" for k in range(len(order))],
                text=[f"{probs[i]*100:.1f}%" for i in order],
                textposition="auto",
            ))
            fig.update_layout(template="plotly_dark", height=340,
                              xaxis_title="Zuordnungswahrscheinlichkeit", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"Vorhersage: **{labels[order[0]]}** — ohne dass das Modell je auf diese "
                       "Label-Menge trainiert wurde (Zero-Shot).")
        info_box(
            "Der Trick von CLIP: Bild und Text landen im **gleichen** Embedding-Raum. Klassifikation wird "
            "zur Ähnlichkeitssuche — die Klassenmenge ist frei wählbar und muss nicht vortrainiert sein.",
            kind="info",
        )

    with tabs[6]:
        section_header("Lernvideos")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**CLIP erklärt**")
            video_embed("KcSXcpluDe4", "CLIP explained")
        with col2:
            st.markdown("**Segment Anything (SAM)**")
            video_embed("eYhvJR4zFUM", "SAM explained")

    divider()
    render_learning_block(
        key_prefix="vision_foundation",
        progression=[
            ("🟢", "Guided", "Erkläre, warum CLIP Zero-Shot-Klassifikation ermöglicht.", "Guided", "green"),
            ("🟠", "Challenge", "Kombiniere Grounding DINO + SAM gedanklich zu 'benannter Segmentierung'.", "Challenge", "amber"),
            ("🔴", "Debug", "SAM findet ein Objekt, kann es aber nicht benennen — warum, und was ergänzt du?", "Debug", "pink"),
            ("🧩", "Mini-Projekt", "Nutze ein CLIP-Modell (open_clip), um 10 Bilder zero-shot zu klassifizieren.", "Projekt", "blue"),
        ],
        mcq_question="Was ist die Kernidee, die CLIPs Zero-Shot-Fähigkeit ermöglicht?",
        mcq_options=[
            "Bild und Text liegen in einem gemeinsamen Embedding-Raum → Klassifikation = Ähnlichkeitssuche",
            "Es wurde auf allen möglichen Klassen trainiert",
            "Es nutzt keine Textinformation",
            "Es segmentiert zuerst und klassifiziert dann",
        ],
        mcq_correct_option="Bild und Text liegen in einem gemeinsamen Embedding-Raum → Klassifikation = Ähnlichkeitssuche",
        open_question="Warum ist ein selbstüberwachtes Modell wie DINO ein gutes Foundation Model, obwohl es keine Labels sieht?",
        cheat_sheet=[
            "Foundation Model: groß vortrainiert, breit übertragbar, oft zero-shot.",
            "SAM 2: promptbare Segmentierung (Bild + Video), sagt 'wo', nicht 'was'.",
            "DINOv2/v3: selbstüberwachte Universalfeatures ohne Labels.",
            "Depth Anything: monokulare Tiefe, zero-shot.",
            "CLIP: gemeinsamer Bild-Text-Raum → Open-Vocabulary & Zero-Shot.",
        ],
        key_takeaways=[
            "Foundation Models verschieben Arbeit von 'pro Task trainieren' zu 'prompten/anpassen'.",
            "Kombinationen (Grounded-SAM) sind oft mächtiger als Einzelmodelle.",
        ],
        common_errors=[
            "SAM als Klassifikator missverstehen.",
            "Zero-Shot mit 'kein Vortraining' verwechseln (es ist viel Vortraining, nur kein Task-Training).",
            "Open-Vocabulary-Modelle wie klassenfeste Detektoren behandeln.",
        ],
    )
    render_quiz_checkpoint(
        key_prefix="vision_foundation",
        module_id="vision_foundation",
        question="Was liefert SAM (Segment Anything) als Ausgabe?",
        options=[
            "Präzise Masken für promptbare Objekte — aber keine Klassennamen",
            "Klassennamen ohne Position",
            "Nur Bounding Boxes mit Labels",
            "Eine Tiefenkarte",
        ],
        correct_option="Präzise Masken für promptbare Objekte — aber keine Klassennamen",
        checklist=[
            "Ich kann Foundation Models von klassischen Task-Modellen abgrenzen.",
            "Ich verstehe Zero-Shot & Open-Vocabulary.",
            "Ich weiß, wie CLIP Bild und Text verbindet.",
        ],
    )
