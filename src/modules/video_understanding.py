"""Video Understanding - Action Recognition und temporale Modelle."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.components import (
    divider,
    hero,
    info_box,
    key_concept,
    lab_header,
    section_header,
    step_list,
    render_quiz_checkpoint,
    video_embed,
)


def _temporal_smooth(signal: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return signal
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(signal, kernel, mode="same")


def render():
    hero(
        eyebrow="Deep Learning · Video",
        title="Video Understanding",
        sub="Action Recognition, temporale Modellierung und Video-Transformer kompakt mit Labs.",
    )

    tabs = st.tabs(
        [
            "🎬 Grundlagen",
            "🏗 Modelle",
            "📏 Benchmarks",
            "🧪 Temporal Lab",
            "💻 Code",
            "🎓 Uebung",
            "✅ Checkpoint",
            "🎬 Videos",
        ]
    )

    with tabs[0]:
        section_header("Was ist Video Understanding?")
        st.markdown(
            """
Im Unterschied zu Bildklassifikation muss ein Modell **Zeitinformation** verstehen:
- Reihenfolge von Bewegungen
- Dauer einer Aktion
- Kontext vor und nach dem Event
            """
        )
        key_concept("⏱️", "Temporal Context", "Nicht nur was im Frame ist, sondern wie es sich ueber Zeit aendert.")
        key_concept("📦", "Clip Sampling", "Frames werden als kurze Clips (z. B. 16/32 Frames) verarbeitet.")
        key_concept("🧭", "Long-Term Dependencies", "Wichtige Hinweise liegen oft viele Frames auseinander.")

    with tabs[1]:
        section_header("Architekturen")
        st.markdown(
            """
| Familie | Beispiel | Pluspunkt | Minuspunkt |
|---|---|---|---|
| 2D CNN + Temporal Head | TSN, TSM | Effizient | Begrenzte Zeitmodellierung |
| 3D CNN | I3D, SlowFast | Starke lokale Dynamik | Hoeherer Compute |
| Transformer | TimeSformer, VideoMAE | Flexible Kontextlaenge | Speicherbedarf |
            """
        )
        step_list(
            [
                ("Frames sampeln", "Clip-Laenge und stride bestimmen."),
                ("Backbone waehlen", "2D+Temporal, 3D oder Transformer."),
                ("Aggregation", "Clip-Logits mitteln oder Attention pooling."),
                ("Evaluation", "Top-1/Top-5, mAP je Datensatz."),
            ]
        )

    with tabs[2]:
        section_header("Typische Datensaetze")
        st.markdown(
            """
| Datensatz | Fokus | Metrik |
|---|---|---|
| Kinetics-400/700 | Allgemeine Actions | Top-1, Top-5 |
| Something-Something V2 | Feine temporale Unterschiede | Top-1 |
| AVA | Spatio-temporale Actions pro Person | mAP |
| Epic-Kitchens | Ego-zentrische Aktionen | Verb/Noun Accuracy |
            """
        )
        info_box("Something-Something bestraft Modelle, die nur statische Appearance lernen.", kind="warn")

    with tabs[3]:
        lab_header("Temporal Smoothing", "Beobachte logits vor und nach zeitlicher Glaettung.")
        n = st.slider("Frames", 32, 256, 96, 8)
        noise = st.slider("Noise-Level", 0.0, 1.0, 0.3, 0.05)
        win = st.slider("Smoothing Window", 1, 25, 7, 2)
        rng = np.random.default_rng(5)
        t = np.linspace(0, 6 * np.pi, n)
        logits = 0.5 + 0.35 * np.sin(t) + noise * rng.normal(0, 0.12, n)
        smooth = _temporal_smooth(logits, win)

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=logits, mode="lines", name="Raw logits", line=dict(color="#F59E0B")))
        fig.add_trace(go.Scatter(y=smooth, mode="lines", name="Smoothed", line=dict(color="#10B981", width=3)))
        fig.update_layout(template="plotly_dark", height=420, xaxis_title="Frame", yaxis_title="Score")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        section_header("Praxis-Code")
        st.code(
            """# VideoMAE Fine-Tuning (Pseudo)
from transformers import VideoMAEForVideoClassification

model = VideoMAEForVideoClassification.from_pretrained(
    "MCG-NJU/videomae-base-finetuned-kinetics"
)
outputs = model(pixel_values=batch_video)  # [B, T, C, H, W]
loss = criterion(outputs.logits, labels)
            """,
            language="python",
        )
        divider()
        st.code(
            """# Torchvision SlowFast (Pseudo)
from torchvision.models.video import slowfast_r50

model = slowfast_r50(weights="DEFAULT")
logits = model(video_clip)
            """,
            language="python",
        )

    with tabs[5]:
        section_header("Mini-Uebung: Clip Benchmark")
        st.markdown(
            """
1. Waehle einen Datensatz (Kinetics mini split oder HMDB51).
2. Trainiere zwei Baselines:
   - 2D CNN + Temporal Average
   - Video Transformer (klein)
3. Vergleiche:
   - Accuracy
   - FPS bei Inference
   - Speicherverbrauch
            """
        )

    with tabs[6]:
        render_quiz_checkpoint(
            key_prefix="video_understanding",
            question="Welcher Faktor ist fuer Video-Modelle zentral, aber bei Bildern weniger kritisch?",
            options=[
                "Temporaler Kontext",
                "RGB-Kanaele",
                "Batch Normalization",
                "Top-1 Accuracy",
            ],
            correct_option="Temporaler Kontext",
            checklist=[
                "Ich kann 2D+Temporal, 3D CNN und Video-Transformer vergleichen.",
                "Ich weiss, warum Clip-Sampling das Ergebnis beeinflusst.",
                "Ich kann ein einfaches Benchmark-Setup fuer zwei Modelle definieren.",
            ],
            capstone_prompt="Entwerfe einen Benchmark fuer Action Recognition mit mindestens zwei "
            "Architekturen und klaren Compute/Latency Grenzen.",
        )

    with tabs[7]:
        section_header("Lernvideos")
        video_embed("QjB6l8YwM2o", "SlowFast", "Wie zwei zeitliche Aufloesungen kombiniert werden.")
        divider()
        video_embed("H5e4gWf7gY8", "TimeSformer", "Attention direkt ueber Raum und Zeit.")
        divider()
        video_embed("AhR8H0i6NWo", "VideoMAE", "Masked pretraining fuer Video Transformer.")
