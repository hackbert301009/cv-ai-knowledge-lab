"""Moderne Architekturen — ResNet, EfficientNet, ConvNeXt, U-Net."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.components import (
    hero, section_header, divider, info_box, lab_header, video_embed,
    render_learning_block, render_quiz_checkpoint,
)


# (Name, Jahr, ImageNet Top-1 %, Params in Mio., Familie)
_ARCHS = [
    ("AlexNet",         2012, 63.3, 60.0,  "Klassik"),
    ("VGG-16",          2014, 71.5, 138.0, "Klassik"),
    ("GoogLeNet",       2014, 69.8, 6.8,   "Klassik"),
    ("ResNet-50",       2015, 76.0, 25.6,  "ResNet"),
    ("ResNet-152",      2015, 78.3, 60.2,  "ResNet"),
    ("DenseNet-201",    2017, 77.4, 20.0,  "ResNet"),
    ("EfficientNet-B0", 2019, 77.1, 5.3,   "EfficientNet"),
    ("EfficientNet-B7", 2019, 84.3, 66.0,  "EfficientNet"),
    ("ViT-B/16",        2020, 84.0, 86.0,  "Transformer"),
    ("Swin-B",          2021, 83.5, 88.0,  "Transformer"),
    ("ConvNeXt-T",      2022, 82.1, 28.0,  "ConvNeXt"),
    ("ConvNeXt-L",      2022, 84.3, 198.0, "ConvNeXt"),
]


def render():
    hero(
        eyebrow="Deep Learning · Moderne Architekturen",
        title="Moderne Architekturen",
        sub="Die wichtigsten Architekturen, die du kennen solltest — und wann du welche benutzt."
    )

    tabs = st.tabs([
        "🏛️ ResNet", "⚡ EfficientNet", "🆕 ConvNeXt", "🎯 U-Net", "🎯 YOLO",
        "🧪 Interaktiv & Vergleich", "🎬 Lernvideos",
    ])

    with tabs[0]:
        section_header("ResNet — Residual Networks (2015)")
        st.markdown(r"""
**Das Problem:** Sehr tiefe Netze ließen sich nicht trainieren. Mit jedem zusätzlichen Layer **stieg der Trainingsfehler**
— nicht durch Overfitting, sondern weil die Gradienten verschwanden oder die Optimierung versagte.

**Die Lösung — Skip Connection:**

$$\mathbf{y} = F(\mathbf{x}) + \mathbf{x}$$

Der Layer lernt nicht $\mathbf{x} \to \mathbf{y}$, sondern $\mathbf{x} \to \mathbf{y} - \mathbf{x}$ (das **Residuum**).
Wenn der Layer "nichts tun" soll, lernt er einfach $F(\mathbf{x}) = 0$ — viel einfacher als die Identität zu lernen.

#### Auswirkung
- Plötzlich sind 50, 101, 152 Layer trainierbar.
- ResNet hat ImageNet 2015 mit 3.57% Top-5-Fehler gewonnen.
- **Skip Connections sind heute überall** — auch in Transformern, Diffusion Models, U-Net.
        """)

    with tabs[1]:
        section_header("EfficientNet — Compound Scaling (2019)")
        st.markdown(r"""
Wenn du ein Netz größer machst, hast du drei Achsen:
- **Tiefe** (mehr Layer)
- **Breite** (mehr Channels)
- **Auflösung** (größere Eingabe)

EfficientNet's Erkenntnis: alle drei zusammen mit einem **Compound Coefficient** $\phi$ skalieren:
$$\text{depth} = \alpha^\phi, \quad \text{width} = \beta^\phi, \quad \text{resolution} = \gamma^\phi$$
mit $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$.

EfficientNet-B0 bis B7 sind dieselbe Architektur, nur an unterschiedlichen $\phi$ skaliert.
**B0** = 5M Params, **B7** = 66M — beide State-of-the-Art für ihre Größe.
        """)

    with tabs[2]:
        section_header("ConvNeXt — CNNs schlagen zurück (2022)")
        st.markdown(r"""
Nach dem ViT-Hype (2020) dachten viele, CNNs seien tot.
**ConvNeXt** (Liu et al., Meta AI) zeigte: nicht ganz.

#### Was sie geändert haben
- Größere Kernel (7×7 statt 3×3)
- Depthwise Convolutions
- LayerNorm statt BatchNorm
- GELU statt ReLU
- Inverted Bottleneck (wie in MobileNet/EfficientNet)
- Weniger Aktivierungen pro Block

Ergebnis: **schlägt ViT-B auf ImageNet** mit demselben Compute. CNNs sind nicht obsolet, sie waren nur unfair gebaut.
        """)
        info_box("Wenn du heute eine schnelle, starke CV-Baseline brauchst: ConvNeXt-Tiny ist top.", kind="success")

    with tabs[3]:
        section_header("U-Net — die Mutter aller Segmentierungsnetze (2015)")
        st.markdown(r"""
**U-Net** (Ronneberger, MICCAI 2015) wurde für medizinische Bildsegmentierung erfunden.
Heute findest du es in:
- Medizinische Bildsegmentierung
- Stable Diffusion (als Denoiser)
- Image-to-Image Tasks aller Art
- Audio-Source-Separation

#### Architektur
- **Encoder** — verkleinert die Auflösung, baut tiefe Features
- **Decoder** — vergrößert wieder auf Originalauflösung
- **Skip Connections** zwischen Encoder- und Decoder-Layern auf gleicher Auflösung

Die Skip Connections sind der Trick: feine Details (Kanten, Lokalisation) gehen sonst beim Downsampling verloren.

#### Modernes U-Net
- Residual-Blöcke
- Attention-Layer (Stable Diffusion)
- Group Norm
- Conditional auf Text (Cross-Attention)
        """)

    with tabs[4]:
        section_header("YOLO — Object Detection in Echtzeit")
        st.markdown(r"""
**YOLO** (You Only Look Once, Redmon 2016) hat Object Detection echtzeitfähig gemacht.

#### Idee
Statt Region-Proposals (R-CNN, Fast R-CNN, Faster R-CNN) macht YOLO **alles in einem Forward-Pass**:
- Bild in Grid teilen
- Pro Grid-Zelle: vorhersagen, ob ein Objekt da ist, wo (Bounding Box) und welche Klasse
- Ein einziges CNN, ein Loss

#### Versionen
| Version | Jahr | Highlight |
|---|---|---|
| YOLOv1 | 2016 | Echtzeit (45 fps) |
| YOLOv3 | 2018 | Multi-Scale, Anker-Boxen |
| YOLOv5 | 2020 | Ultralytics — superpopuläres Repo |
| YOLOv8 | 2023 | Anchor-free, modern |
| YOLO11 | 2024 | Aktuelle Ultralytics-Generation |
        """)
        info_box(
            "Wenn du einfach nur Object Detection brauchst und nicht selbst trainieren willst: "
            "Ultralytics YOLO mit pretrained-Weights. Drei Zeilen Code, fertig.",
            kind="tip",
        )

    # ── Tab 5: Interaktiv & Vergleich ────────────────────────────────────────
    with tabs[5]:
        lab_header("Architektur-Landkarte", "Accuracy vs. Modellgröße über die Jahre — je größer der Punkt, desto mehr Parameter.")

        families = sorted({a[4] for a in _ARCHS})
        chosen = st.multiselect("Familien einblenden", families, default=families)
        min_year = st.slider("Ab Jahr", 2012, 2022, 2012)

        fig = go.Figure()
        for fam in chosen:
            pts = [a for a in _ARCHS if a[4] == fam and a[1] >= min_year]
            if not pts:
                continue
            fig.add_scatter(
                x=[p[1] for p in pts], y=[p[2] for p in pts],
                mode="markers+text", name=fam,
                text=[p[0] for p in pts], textposition="top center",
                marker=dict(size=[max(8, p[3] ** 0.5 * 2.2) for p in pts], opacity=0.8),
            )
        fig.update_layout(
            template="plotly_dark", height=460,
            xaxis_title="Jahr", yaxis_title="ImageNet Top-1 Accuracy (%)",
            xaxis=dict(dtick=1),
        )
        st.plotly_chart(fig, use_container_width=True)
        info_box(
            "Der Trend: bei gleicher Accuracy werden Modelle über die Jahre **kleiner/effizienter** "
            "(GoogLeNet, EfficientNet-B0, ConvNeXt-T) — nicht nur größer.",
            kind="info",
        )

        divider()
        lab_header("EfficientNet Compound Scaling", "Ein Regler φ skaliert Tiefe, Breite und Auflösung gemeinsam.")
        phi = st.slider("Compound Coefficient φ", 0.0, 4.0, 1.0, 0.5)
        alpha, beta, gamma = 1.2, 1.1, 1.15  # aus dem Paper
        depth = alpha ** phi
        width = beta ** phi
        res = gamma ** phi
        flops = (alpha * beta ** 2 * gamma ** 2) ** phi  # ≈ 2^φ
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tiefe ×", f"{depth:.2f}")
        c2.metric("Breite ×", f"{width:.2f}")
        c3.metric("Auflösung ×", f"{res:.2f}")
        c4.metric("FLOPs ×", f"{flops:.2f}")
        st.caption(f"Basis-Auflösung 224 px → **{224 * res:.0f} px**. FLOPs wachsen ~2^φ = {2 ** phi:.2f}× "
                   f"(φ={phi:g}).")

    # ── Tab 6: Lernvideos ────────────────────────────────────────────────────
    with tabs[6]:
        section_header("Lernvideos")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**ResNet — Residual Learning erklärt**")
            video_embed("RYth6EbBUqM", "ResNet explained")
        with col2:
            st.markdown("**U-Net — Semantic Segmentation**")
            video_embed("NhdzGfB1q74", "U-Net explained")

    divider()
    render_learning_block(
        key_prefix="modern_archs",
        progression=[
            ("🟢", "Guided", "Zeichne einen ResNet-Basic-Block mit Skip Connection auf Papier.", "Guided", "green"),
            ("🟠", "Challenge", "Wähle für 3 Szenarien (Edge, Server, Medizin-Segmentierung) die passende Architektur.", "Challenge", "amber"),
            ("🔴", "Debug", "Ein 100-Layer-CNN ohne Skip Connections trainiert nicht — erkläre warum.", "Debug", "pink"),
            ("🧩", "Mini-Projekt", "Lade ConvNeXt-Tiny (timm) und klassifiziere 5 eigene Bilder.", "Projekt", "blue"),
        ],
        mcq_question="Warum lassen sich mit Skip Connections viel tiefere Netze trainieren?",
        mcq_options=[
            "Der Gradient kann über die Identitätsverbindung ungehindert zurückfließen",
            "Skip Connections reduzieren die Parameterzahl",
            "Sie ersetzen die Aktivierungsfunktion",
            "Sie erhöhen die Bildauflösung",
        ],
        mcq_correct_option="Der Gradient kann über die Identitätsverbindung ungehindert zurückfließen",
        open_question="Wann würdest du ConvNeXt einem ViT vorziehen — und wann umgekehrt?",
        cheat_sheet=[
            "ResNet: Skip Connection y = F(x) + x — Standard-Backbone.",
            "EfficientNet: Compound Scaling α·β²·γ² ≈ 2.",
            "ConvNeXt: modernisiertes CNN, schlägt ViT bei gleichem Compute.",
            "U-Net: Encoder-Decoder + Skips — Segmentierung & Diffusion-Denoiser.",
        ],
        key_takeaways=[
            "Skip Connections sind die wichtigste Einzelidee moderner Architekturen.",
            "Effizienz (Accuracy pro FLOP) ist wichtiger als reine Größe.",
        ],
        common_errors=[
            "Sehr tiefe Netze ohne Residual-Verbindungen bauen.",
            "ViT auf kleinen Datensätzen ohne Pretraining einsetzen.",
            "Nur Accuracy vergleichen, FLOPs/Latenz ignorieren.",
        ],
    )
    render_quiz_checkpoint(
        key_prefix="modern_archs",
        module_id="modern_archs",
        question="Was ist die Kernidee des Compound Scaling von EfficientNet?",
        options=[
            "Tiefe, Breite und Auflösung gemeinsam mit einem Koeffizienten skalieren",
            "Nur die Netztiefe erhöhen",
            "Batch Normalization durch LayerNorm ersetzen",
            "Region-Proposals statt Grid nutzen",
        ],
        correct_option="Tiefe, Breite und Auflösung gemeinsam mit einem Koeffizienten skalieren",
        checklist=[
            "Ich kann den ResNet-Residual-Block aufzeichnen.",
            "Ich verstehe Compound Scaling.",
            "Ich weiß, wofür U-Net-Skip-Connections gut sind.",
        ],
    )
