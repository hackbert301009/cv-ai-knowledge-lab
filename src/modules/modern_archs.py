"""Moderne Architekturen — ResNet, EfficientNet, ConvNeXt, U-Net."""
import streamlit as st
from src.components import hero, section_header, divider, info_box, card, render_card_grid


def render():
    hero(
        eyebrow="Deep Learning · Modul 14",
        title="Moderne Architekturen",
        sub="Die wichtigsten Architekturen, die du kennen solltest — und wann du welche benutzt."
    )

    tabs = st.tabs(["🏛️ ResNet", "⚡ EfficientNet", "🆕 ConvNeXt", "🎯 U-Net", "🎯 YOLO"])

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
- ResNet hat ImageNet 2015 mit 3.57% Fehler gewonnen — unterhalb der menschlichen Leistung.
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
mit $\alpha \beta^2 \gamma^2 \approx 2$.

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
| YOLOv9–v11 | 2024 | Aktuelle Generation |
        """)
        info_box(
            "Wenn du einfach nur Object Detection brauchst und nicht selbst trainieren willst: "
            "Ultralytics YOLO mit pretrained-Weights. Drei Zeilen Code, fertig.",
            kind="tip",
        )
