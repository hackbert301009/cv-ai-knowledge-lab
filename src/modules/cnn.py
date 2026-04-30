"""Convolutional Neural Networks — von LeNet bis ResNet."""
import streamlit as st
from src.components import hero, section_header, divider, info_box, card, render_card_grid


def render():
    hero(
        eyebrow="Deep Learning · Modul 12",
        title="Convolutional Neural Networks",
        sub="Die Architektur, die Computer Vision revolutioniert hat. "
            "Von LeNet (1998) bis ResNet (2015) — die Geschichte und die Konzepte."
    )

    tabs = st.tabs(["🎯 Warum CNN?", "🔲 Bausteine", "🏛️ Geschichte", "📐 Output-Größe", "💻 Code"])

    with tabs[0]:
        section_header("Warum nicht einfach ein MLP?")
        st.markdown(r"""
Ein 224×224×3 Bild hat **150 528 Pixel**. Eine Dense-Schicht mit 1000 Neuronen hätte
$150528 \times 1000 = 150{,}5$ Mio Parameter — nur in einer Schicht!

**Drei Probleme:**
1. **Zu viele Parameter** → Overfitting, langsam
2. **Keine räumliche Struktur** → Pixel (1,1) und (1,2) sind benachbart, das MLP weiß das nicht
3. **Keine Translationsinvarianz** → Eine Katze links oben ist für ein MLP eine ganz andere Eingabe als eine Katze rechts unten

#### CNN löst alle drei
- **Weight Sharing**: gleicher Kernel an jeder Position → wenige Parameter
- **Lokal verbunden**: nur Nachbarpixel beeinflussen einen Output → räumliche Struktur bleibt
- **Translationsequivariant**: Kanten werden gefunden, egal wo sie sind
        """)

    with tabs[1]:
        section_header("Die Bausteine eines CNN")
        cards = [
            card("🔲", "Convolution", "Lernbarer Kernel gleitet übers Bild. Ein Layer hat oft 32–512 verschiedene Kernel.", ["Hauptbaustein"], ["amber"]),
            card("⚡", "Activation", "Nichtlinearität nach jeder Conv (meist ReLU). Sonst wäre alles linear.", ["nach Conv"], ["amber"]),
            card("⬇️", "Pooling", "Max- oder Average-Pooling reduziert die Auflösung. Macht Netz robust und schnell.", ["Downsampling"], ["amber"]),
            card("📦", "Batch Norm", "Normalisiert Aktivierungen. Macht Training viel stabiler.", ["Stabilität"], ["amber"]),
            card("🌐", "Global Pooling", "Statt Flatten + Dense: GAP reduziert auf einen Vektor pro Channel.", ["modern"], ["amber"]),
            card("🔗", "Residual Connection", "Skip Connection: Output = F(x) + x. Macht sehr tiefe Netze trainierbar (ResNet).", ["Game-Changer"], ["pink"]),
        ]
        render_card_grid(cards, cols=3)

    with tabs[2]:
        section_header("Die Geschichte der Klassiker")
        st.markdown("""
| Jahr | Architektur | Innovation |
|---|---|---|
| 1998 | **LeNet-5** (LeCun) | Erstes CNN für Handschriften (MNIST). Conv → Pool → Conv → Pool → FC. |
| 2012 | **AlexNet** (Krizhevsky et al.) | Gewinn ImageNet 2012 mit großem Vorsprung. ReLU, Dropout, GPU-Training. **Der Big Bang von Deep Learning.** |
| 2014 | **VGG** (Oxford) | Sehr tief (16/19 Layer), nur 3×3 Conv. Klar, modular. |
| 2014 | **GoogLeNet / Inception** | Inception-Module: parallele Conv mit verschiedenen Kernelgrößen. 1×1 Conv für Bottleneck. |
| 2015 | **ResNet** (He et al.) | **Skip Connections** — ermöglicht 50/101/152 Layer. Aktueller Default-Backbone. |
| 2017 | **DenseNet** | Jeder Layer ist mit jedem nachfolgenden verbunden. Effizienter Parameter-Use. |
| 2019 | **EfficientNet** | Compound Scaling: Tiefe, Breite, Auflösung gemeinsam optimieren. |
| 2022 | **ConvNeXt** | "Modernisiertes" CNN, das mit Transformern mithält. CNNs sind nicht tot. |
""")
        info_box(
            "Wenn du heute eine starke CV-Baseline brauchst: ResNet-50 ist immer noch der Standard. "
            "ConvNeXt-Tiny ist die moderne Alternative — vergleichbar zu ViT-B, aber CNN.",
            kind="tip",
        )

    with tabs[3]:
        section_header("Output-Größe berechnen")
        st.markdown(r"""
Bei einem Conv-Layer mit Input $H_\text{in}$, Kernel $k$, Stride $s$, Padding $p$:

$$H_\text{out} = \left\lfloor \frac{H_\text{in} + 2p - k}{s} \right\rfloor + 1$$

#### Beispiele
| Input | Kernel | Stride | Padding | Output |
|---|---|---|---|---|
| 224 | 3 | 1 | 1 | 224 (same) |
| 224 | 7 | 2 | 3 | 112 (halbiert) |
| 224 | 3 | 2 | 1 | 112 |
| 112 | 1 | 1 | 0 | 112 (1×1 Conv: nur Channels mischen) |

#### Anzahl Parameter eines Conv-Layers
$$\text{params} = (k \times k \times C_\text{in} + 1) \cdot C_\text{out}$$

(das +1 ist der Bias). Bei einem 3×3 Conv mit 64 → 128 Channels: $(9 \cdot 64 + 1) \cdot 128 = 73{,}856$ Parameter.
        """)

    with tabs[4]:
        section_header("Ein einfaches CNN in PyTorch")
        st.code("""
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                        # 32x16x16

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                        # 64x8x8

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                # 128x1x1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = SimpleCNN(num_classes=10)
out = model(torch.randn(8, 3, 32, 32))
print(out.shape)   # torch.Size([8, 10])
        """, language="python")
