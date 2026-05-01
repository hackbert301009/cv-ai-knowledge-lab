"""Convolutional Neural Networks — von LeNet bis ConvNeXt."""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.components import (
    hero, section_header, divider, info_box,
    card, render_card_grid, video_embed, lab_header, key_concept, step_list,
)


def render():
    hero(
        eyebrow="Deep Learning · Modul 12",
        title="Convolutional Neural Networks",
        sub="Die Architektur, die Computer Vision revolutioniert hat. "
            "Von LeNet (1998) bis ConvNeXt (2022) — Geschichte, Intuition und Praxis."
    )

    tabs = st.tabs([
        "🎯 Warum CNN?",
        "🔲 Bausteine",
        "🏛️ Architektur-Geschichte",
        "📐 Output-Größe berechnen",
        "🧪 Interaktiv",
        "💻 Code",
        "🎬 Lernvideos",
    ])

    # ------------------------------------------------------------------ #
    with tabs[0]:
        section_header("Warum nicht einfach ein MLP?",
                       "Das fundamentale Problem mit dichten Netzen für Bilder.")
        st.markdown(r"""
Ein **224×224×3** Bild (Standard ImageNet-Eingabe) hat $224 \times 224 \times 3 = 150\,528$ Pixel.
Eine einzige Dense-Schicht mit 1000 Neuronen hätte:

$$150\,528 \times 1000 = 150{,}5\,\text{Mio. Parameter} \quad \text{(nur in einer Schicht!)}$$

Zum Vergleich: ResNet-50 hat 25 Mio. Parameter **gesamt** und ist viel leistungsfähiger.
        """)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
**Problem 1: Zu viele Parameter**

Dense-Netz für 224×224:
- Layer 1: 150M Parameter
- Layer 2: nochmal 150M
- → Overfitting, riesiger Speicher
""")
        with col2:
            st.markdown("""
**Problem 2: Keine Raumstruktur**

Pixel (1,1) und (1,2) sind Nachbarn,
aber für ein MLP sind sie unabhängige
Eingabe-Dimensionen. Das Netz kann
Lokalität nicht ausnutzen.
""")
        with col3:
            st.markdown("""
**Problem 3: Keine Invarianz**

Eine Katze links oben ist für
ein MLP eine **komplett andere**
Eingabe als eine Katze rechts unten.
Müsste jede Position separat lernen.
""")

        divider()
        section_header("CNNs lösen alle drei Probleme")

        key_concept("⚖️", "Weight Sharing",
                    "Derselbe Kernel wird an jeder Position angewendet. Ein 3×3-Kernel hat 9 Parameter "
                    "— statt 150.528 für einen Dense-Layer. Das spart 99,99% der Parameter.")
        key_concept("📍", "Lokale Konnektivität",
                    "Jeder Output hängt nur von einem kleinen Bildausschnitt ab (dem Receptive Field). "
                    "Benachbarte Pixel beeinflussen sich gegenseitig — wie in der realen Welt.")
        key_concept("🔄", "Translationsäquivarianz",
                    "Kanten werden gefunden, egal wo im Bild sie sind. Wenn die Eingabe sich verschiebt, "
                    "verschiebt sich der Output genauso. (Achtung: Äquivarianz ≠ Invarianz!)")

    # ------------------------------------------------------------------ #
    with tabs[1]:
        section_header("Die Bausteine eines CNN",
                       "Jeder Baustein hat eine klare Aufgabe.")

        cards = [
            card("🔲", "Convolution Layer",
                 "Lernbarer Kernel gleitet übers Bild. $C_{out}$ verschiedene Kernel pro Layer, "
                 "jeder lernt ein anderes Feature (Kante, Textur, Form).",
                 ["Kern-OP"], ["amber"]),
            card("⚡", "Activation (ReLU)",
                 "Nichtlinearität nach jeder Conv. Ohne wäre das gesamte CNN eine einzige lineare "
                 "Transformation. ReLU ist Standard, GELU in modernen Archs.",
                 ["nach Conv"], ["amber"]),
            card("⬇️", "Pooling",
                 "Max- oder Average-Pooling reduziert Auflösung (typisch 2×). "
                 "Macht das Netz robuster gegen kleine Verschiebungen.",
                 ["Downsampling"], ["amber"]),
            card("📦", "Batch Normalization",
                 "Normalisiert Aktivierungen pro Batch. Ermöglicht höhere Lernraten, "
                 "macht Training stabiler. Reihenfolge: Conv → BN → Activation.",
                 ["Stabilität"], ["blue"]),
            card("🌐", "Global Average Pooling",
                 "Statt Flatten+Dense: GAP mittelt jeden Feature-Map auf einen Wert. "
                 "Viel weniger Parameter, besser generalisierbar.",
                 ["modern"], ["blue"]),
            card("🔗", "Residual Connection",
                 "Skip Connection: Output = F(x) + x. Löst Vanishing Gradient in sehr tiefen Netzen. "
                 "Das Netz kann lernen, den Block zu ignorieren (→ Identität).",
                 ["Game-Changer"], ["pink"]),
            card("🔀", "Depthwise Separable Conv",
                 "Aufgeteilt in Depthwise (räumlich) + Pointwise (1×1, kanalweise). "
                 "8-9× weniger Operationen. Basis von MobileNet.",
                 ["Effizienz"], ["green"]),
            card("🔄", "Squeeze-and-Excitation",
                 "Lernt, welche Feature-Maps wichtig sind (Channel Attention). "
                 "Kleiner Overhead, großer Gewinn.",
                 ["Attention"], ["green"]),
            card("📏", "1×1 Convolution",
                 "Mixiert Kanäle ohne räumliche Interaktion. Reduziert/erhöht Channelzahl "
                 "günstig (Bottleneck in ResNet). Sehr mächtig.",
                 ["Bottleneck"], ["blue"]),
        ]
        render_card_grid(cards, cols=3)

        st.markdown("#### Warum BN vor oder nach Activation?")
        st.markdown(r"""
Original ResNet: `Conv → BN → ReLU`
Pre-Activation ResNet: `BN → ReLU → Conv` (oft besser für tiefe Netze)

In der Praxis: folge dem Paper der Architektur, die du benutzt.
        """)

    # ------------------------------------------------------------------ #
    with tabs[2]:
        section_header("Architektur-Geschichte", "Von 1998 bis heute — die Meilensteine.")

        timeline = {
            1998: ("LeNet-5", "LeCun et al.", "Erstes CNN für Handschriften (MNIST). 60K Parameter. Conv→Pool→Conv→Pool→FC. Beweis, dass CNNs funktionieren.", "#3B82F6"),
            2012: ("AlexNet", "Krizhevsky et al.", "ImageNet 2012: 63% → 78% Top-5 Accuracy. ReLU, Dropout, GPU-Training. DER Big Bang von Deep Learning.", "#EC4899"),
            2014: ("VGG-16/19", "Simonyan et al. (Oxford)", "Sehr tief (16-19 Layer), nur 3×3 Conv. Klar, modular. Noch heute als Backbone genutzt.", "#7C3AED"),
            2014: ("GoogLeNet/Inception", "Szegedy et al.", "Inception-Module: parallele Conv (1×1, 3×3, 5×5). 6,8M Parameter bei 22 Layern. Sehr effizient.", "#F59E0B"),
            2015: ("ResNet-50/101/152", "He et al.", "SKIP CONNECTIONS! Gradient fließt direkt durch. Ermöglicht 50-1000+ Layer. Noch heute Standard-Backbone.", "#10B981"),
            2017: ("DenseNet", "Huang et al.", "Jeder Layer verbunden mit jedem folgenden. Feature Reuse, sehr parameter-effizient.", "#06B6D4"),
            2019: ("EfficientNet", "Tan & Le (Google)", "Compound Scaling: Tiefe, Breite, Auflösung gemeinsam optimieren. B0-B7 Varianten. Damals SOTA auf ImageNet.", "#A78BFA"),
            2022: ("ConvNeXt", "Liu et al. (Meta/Facebook)", "'Modernisiertes' CNN: übernimmt Ideen von ViT (Layer Norm, GELU, Patchify Stem). Konkurriert mit Swin Transformer.", "#EC4899"),
        }

        for year, (name, authors, desc, color) in sorted(timeline.items()):
            st.markdown(
                f"""<div style="display:flex;gap:1rem;align-items:flex-start;padding:0.75rem 0;
                    border-left:2px solid {color};padding-left:1rem;margin-left:0.5rem;margin-bottom:0.5rem;">
                    <div style="min-width:3rem;font-weight:800;color:{color};font-size:0.85rem;">{year}</div>
                    <div>
                      <div style="font-weight:700;color:#F3F4F6;">{name} <span style="font-weight:400;color:#9CA3AF;font-size:0.85rem;">— {authors}</span></div>
                      <div style="font-size:0.875rem;color:#9CA3AF;margin-top:0.2rem;">{desc}</div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

        info_box(
            "**Heute 2026:** Transformer-basierte Architekturen (ViT, Swin) dominieren viele Benchmarks, "
            "aber ConvNeXt und EfficientNet sind weiterhin top für praktische Anwendungen auf Edge-Geräten.",
            kind="tip",
        )

        divider()
        section_header("ResNet — das Herzstück verstehen")
        st.markdown(r"""
#### Warum tiefe Netze **ohne** Skip Connections schlechter werden

Wenn du ein 56-Layer-Netz nimmst und ein 20-Layer-Netz, sollte das 56er mindestens genauso gut sein
(es könnte die extra Layer einfach als Identität lernen). Aber in der Praxis ist das 56er **schlechter**.

Das ist kein Overfitting — auch auf den Trainingsdaten ist es schlechter.
Die Gradienten verschwinden einfach in den tiefen Layern.

#### Die ResNet-Lösung: Skip Connections

$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$

Der Residual-Block lernt nur die **Differenz** $\mathcal{F}(\mathbf{x}) = \mathbf{y} - \mathbf{x}$
(das "Residual"). Das ist viel einfacher als das volle Mapping zu lernen.

**Kritischer Effekt:** Der Gradient fließt **ungehindert** durch die $+\mathbf{x}$ Verbindung zurück —
kein Vanishing Gradient mehr, egal wie tief das Netz ist.
        """)

    # ------------------------------------------------------------------ #
    with tabs[3]:
        section_header("Output-Größe berechnen", "Die Formel, die du auswendig kennen solltest.")
        st.markdown(r"""
Bei einem Conv-Layer mit Eingabe $H_\text{in}$, Kernel $k$, Stride $s$, Padding $p$:

$$H_\text{out} = \left\lfloor \frac{H_\text{in} + 2p - k}{s} \right\rfloor + 1$$

Für **Same Padding** (Output = Input): $p = \lfloor k/2 \rfloor$ bei $s=1$.
        """)

        lab_header("Conv Output-Rechner", "Berechne interaktiv Ausgabegröße und Parameter-Anzahl.")

        cc1, cc2, cc3, cc4 = st.columns(4)
        h_in = cc1.number_input("H_in (Input)", min_value=1, max_value=2048, value=224)
        k = cc2.number_input("Kernel k", min_value=1, max_value=31, value=3, step=2)
        s = cc3.number_input("Stride s", min_value=1, max_value=8, value=1)
        p = cc4.number_input("Padding p", min_value=0, max_value=32, value=1)

        cc5, cc6, cc7 = st.columns(3)
        c_in = cc5.number_input("C_in (Eingangs-Channels)", min_value=1, max_value=2048, value=64)
        c_out = cc6.number_input("C_out (Ausgangs-Channels)", min_value=1, max_value=2048, value=128)
        groups = cc7.number_input("Gruppen (1=normal, C=depthwise)", min_value=1, max_value=2048, value=1)

        h_out = int(np.floor((h_in + 2*p - k) / s) + 1)
        params_per_filter = (k * k * (c_in // groups) + 1)
        params_total = params_per_filter * c_out
        flops = 2 * k * k * (c_in // groups) * c_out * h_out * h_out  # multiply-adds

        result_ok = h_out > 0

        if result_ok:
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Ausgabegröße", f"{h_out}×{h_out}")
            r2.metric("Parameter", f"{params_total:,}")
            r3.metric("FLOPs (approx)", f"{flops/1e6:.1f}M")
            r4.metric("Auflösungsfaktor", f"×{h_out/h_in:.2f}")
        else:
            st.error("⚠️ Ungültige Konfiguration — Ausgabegröße ist 0 oder negativ!")

        st.markdown("#### Häufige Konfigurationen")
        common = [
            ("224", "3", "1", "1", "→ 224", "Same Padding — Auflösung bleibt"),
            ("224", "7", "2", "3", "→ 112", "ResNet Stem — halbiert Auflösung"),
            ("224", "3", "2", "1", "→ 112", "Stride 2 — günstiger als Pooling"),
            ("112", "1", "1", "0", "→ 112", "1×1 Conv — nur Channels ändern"),
            ("7",   "7", "1", "0", "→ 1",   "AlexNet FC-Schicht als Conv"),
        ]
        st.markdown("| H_in | k | s | p | H_out | Warum |")
        st.markdown("|---|---|---|---|---|---|")
        for row in common:
            st.markdown(f"| {' | '.join(row)} |")

    # ------------------------------------------------------------------ #
    with tabs[4]:
        lab_header("Interaktive Feature-Map Visualisierung",
                   "Wie Feature-Maps durch das Netz fließen und was ein CNN 'sieht'.")

        st.markdown("#### Was ein trainiertes CNN in verschiedenen Layern sieht")
        st.markdown(r"""
Ein CNN lernt hierarchische Features — jeder Layer baut auf dem vorherigen auf:

| Layer-Tiefe | Was gelernt wird | Beispiele |
|---|---|---|
| Layer 1 (sehr früh) | Einfache Kanten & Farben | Horizontale Linien, Vertikale Linien, Farb-Bänder |
| Layer 2–3 | Muster & Texturen | Gitter, Ecken, Kurven |
| Layer 4–6 | Teile von Objekten | Augen, Räder, Fenster |
| Layer 7+ | Ganze Objekte | Gesichter, Autos, Hunde |
        """)

        # Interaktive Demonstration: Synthetische Kernel
        st.markdown("#### Kernel-Visualisierung (synthetisch)")
        kernel_type = st.selectbox("Kernel-Typ", [
            "Kante links", "Kante rechts", "Kante oben", "Kante unten",
            "Diagonal 45°", "Diagonal 135°", "Zentrum", "Ring",
        ])

        kernel_map = {
            "Kante links":    np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float),
            "Kante rechts":   np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float),
            "Kante oben":     np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float),
            "Kante unten":    np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float),
            "Diagonal 45°":   np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=float),
            "Diagonal 135°":  np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype=float),
            "Zentrum":        np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=float),
            "Ring":           np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=float),
        }
        kern = kernel_map[kernel_type]

        import cv2
        # Synthetisches Bild
        test_img = np.zeros((120, 120), dtype=np.float32)
        cv2.rectangle(test_img, (10, 10), (60, 60), 1.0, -1)
        cv2.circle(test_img, (90, 80), 25, 1.0, -1)
        cv2.line(test_img, (0, 80), (120, 40), 1.0, 2)

        filtered = cv2.filter2D(test_img, -1, kern.astype(np.float32))
        filtered_abs = np.abs(filtered)

        kcol1, kcol2, kcol3 = st.columns(3)
        kcol1.markdown("**Kernel**")
        fig_k = go.Figure(go.Heatmap(
            z=kern, colorscale=[[0, "#EF4444"], [0.5, "#1F2937"], [1, "#3B82F6"]],
            showscale=False, text=kern.astype(str), texttemplate="%{text}",
        ))
        fig_k.update_layout(
            height=180, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )
        kcol1.plotly_chart(fig_k, use_container_width=True)

        kcol2.markdown("**Eingabe**")
        kcol2.image((test_img * 255).astype(np.uint8), clamp=True, use_container_width=True)

        kcol3.markdown("**Feature Map (nach Kernel)**")
        fmap_vis = np.clip(filtered_abs / filtered_abs.max() * 255, 0, 255).astype(np.uint8)
        kcol3.image(fmap_vis, clamp=True, use_container_width=True)

        info_box(
            "Das ist genau was Layer 1 eines CNN tut — nur dass CNN seine Kernel selbst **lernt** "
            "statt sie hardcoded zu haben.",
            kind="info",
        )

    # ------------------------------------------------------------------ #
    with tabs[5]:
        section_header("CNN in PyTorch — von einfach bis ResNet")

        st.markdown("#### Einfaches CNN (CIFAR-10)")
        st.code("""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Block 1: 3→32 Channels, 32×32 → 16×16
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),              # 32×32 → 16×16
        )
        # Block 2: 32→64 Channels, 16×16 → 8×8
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),              # 16×16 → 8×8
        )
        # Block 3: 64→128 Channels, 8×8 → 1×1 via GAP
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),      # Global Average Pooling → 128×1×1
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.flatten(1)                  # [B, 128, 1, 1] → [B, 128]
        return self.classifier(x)

# Modell testen
model = SimpleCNN(num_classes=10)
dummy = torch.randn(8, 3, 32, 32)
out = model(dummy)
print(f"Input: {dummy.shape} → Output: {out.shape}")
# Parameteranzahl
total = sum(p.numel() for p in model.parameters())
print(f"Parameter: {total:,}")
        """, language="python")

        divider()
        st.markdown("#### Residual Block (ResNet-Stil)")
        st.code("""
class ResidualBlock(nn.Module):
    def __init__(self, channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(channels, channels * (2 if downsample else 1),
                               3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels * (2 if downsample else 1))
        self.conv2 = nn.Conv2d(channels * (2 if downsample else 1),
                               channels * (2 if downsample else 1),
                               3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels * (2 if downsample else 1))
        self.relu  = nn.ReLU(inplace=True)
        # Skip Connection: falls Größe ändert, 1×1 Conv zum Anpassen
        self.shortcut = nn.Sequential(
            nn.Conv2d(channels, channels * (2 if downsample else 1),
                      1, stride=stride, bias=False),
            nn.BatchNorm2d(channels * (2 if downsample else 1)),
        ) if downsample else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)       # Skip Connection
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)    # F(x) + x

# Beispiel
block = ResidualBlock(64, downsample=True)   # 64ch → 128ch, /2 Auflösung
out = block(torch.randn(4, 64, 56, 56))
print(out.shape)  # torch.Size([4, 128, 28, 28])
        """, language="python")

        divider()
        st.markdown("#### Transfer Learning mit vortrainiertem ResNet-50")
        st.code("""
import torchvision.models as models
import torch.nn as nn

# Vortrainiertes ResNet-50 (ImageNet)
model = models.resnet50(weights='IMAGENET1K_V2')

# Option 1: Feature Extraction — nur letzten Layer austauschen
for param in model.parameters():
    param.requires_grad = False          # Alle Layer einfrieren

model.fc = nn.Linear(2048, num_classes)  # Nur Classifier wird trainiert

# Option 2: Fine-Tuning — alle Layer mit kleiner LR trainieren
for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW([
    {'params': model.layer4.parameters(), 'lr': 1e-4},   # Tiefe Layer: kleine LR
    {'params': model.fc.parameters(),     'lr': 1e-3},   # Neuer Kopf: größere LR
])

# Transfer Learning ist fast immer besser als From-Scratch auf kleinen Datasets!
        """, language="python")

    # ------------------------------------------------------------------ #
    with tabs[6]:
        section_header("Lernvideos", "Die besten Erklärungen zu CNNs.")

        st.markdown("#### 3Blue1Brown — But what is a convolution?")
        video_embed("KuXjwB4LzSA",
                    "But what is a convolution? — 3Blue1Brown",
                    "Grant Sanderson erklärt Faltung visuell und mathematisch. Essenziell.")

        divider()

        st.markdown("#### How Convolutional Neural Networks work (Brandon Rohrer)")
        video_embed("FmpDIaiMIeA",
                    "How Convolutional Neural Networks work",
                    "Schritt für Schritt: wie ein CNN Bilder verarbeitet. Für Einsteiger perfekt.")

        divider()

        st.markdown("#### Andrej Karpathy — Building micrograd (Autograd von Grund auf)")
        video_embed("VMj-3S1tku0",
                    "The spelled-out intro to neural networks and backpropagation — Andrej Karpathy",
                    "Andrej Karpathy baut ein komplettes Autograd-System und neuronales Netz in ~2h. "
                    "Wer das versteht, versteht PyTorch wirklich.")

        info_box(
            "Karpathy's Tutorial ist der Goldstandard: nach diesem Video weißt du nicht nur, "
            "**was** PyTorch macht, sondern **warum** — weil du es selbst gebaut hast.",
            kind="tip",
        )
