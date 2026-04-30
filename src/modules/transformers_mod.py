"""Transformer & Attention."""
import streamlit as st
from src.components import hero, section_header, divider, info_box


def render():
    hero(
        eyebrow="State-of-the-Art · Modul 15",
        title="Transformer &amp; Attention",
        sub="Die Architektur, die alles verändert hat. Self-Attention, Vision Transformer, "
            "Swin — die neue Lingua Franca von Computer Vision."
    )

    tabs = st.tabs(["⚡ Attention", "🤖 Transformer", "👁️ ViT", "🪟 Swin", "🆚 ViT vs CNN"])

    with tabs[0]:
        section_header("Self-Attention — die zentrale Idee")
        st.markdown(r"""
Self-Attention erlaubt jedem Element einer Sequenz, **direkt** mit jedem anderen zu kommunizieren —
nicht durch lokale Convolutions, sondern global.

#### Drei Vektoren pro Position
- **Query** ($\mathbf{q}$): "Was suche ich?"
- **Key** ($\mathbf{k}$): "Was biete ich an?"
- **Value** ($\mathbf{v}$): "Wenn du mich willst, hier ist mein Inhalt."

#### Die Formel
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

In Worten:
1. Berechne Ähnlichkeit jeder Query mit jedem Key (Skalarprodukt).
2. Skaliere mit $\sqrt{d_k}$ (verhindert zu große Werte).
3. Softmax → Aufmerksamkeitsgewichte.
4. Gewichteter Mittelwert der Values.

#### Multi-Head Attention
Statt einer Attention-Operation laufen **mehrere parallel** ($h$ Heads), jeder mit eigenen $Q, K, V$.
Verschiedene Heads lernen verschiedene Beziehungstypen.
        """)

    with tabs[1]:
        section_header("Der Transformer-Block")
        st.markdown(r"""
Ein **Transformer-Block** besteht aus:

1. **Multi-Head Self-Attention**
2. **Add & LayerNorm** (Skip Connection + Normalisierung)
3. **Feed-Forward Network** (zwei Dense-Layer mit GELU dazwischen)
4. **Add & LayerNorm**

```
x → MHA(LN(x)) + x → FFN(LN(x)) + x → output
```

Diese Struktur wird **N Mal gestapelt** (N=12 für ViT-Base, N=24 für Large).

#### Positional Encoding
Da Attention permutationsinvariant ist, müssen Positionen explizit kodiert werden:
- **Absolute Sinus-Cosinus** (Original Transformer)
- **Lernbare Embeddings** (BERT, ViT)
- **Relative** (T5, Swin)
- **RoPE** (LLaMA, modern)
        """)

    with tabs[2]:
        section_header("Vision Transformer (ViT) — Bilder als Sequenz")
        st.markdown(r"""
**ViT** (Dosovitskiy et al., 2020) hat die Idee gehabt: behandle ein Bild einfach wie einen Text.

#### Pipeline
1. Bild $224 \times 224 \times 3$ in **Patches** $16 \times 16$ teilen → 196 Patches
2. Jeden Patch flach machen → 768-d Vektor (linear projiziert)
3. **CLS-Token** vorne anhängen (wird zur Klassifikation benutzt)
4. **Positional Embeddings** addieren
5. Durch $N$ Transformer-Blöcke schicken
6. CLS-Token am Ende → Klassifikator

#### Was war der Aha-Moment?
- Mit **genug Daten** (JFT-300M, mehr als ImageNet) übertrifft ViT klar CNNs.
- Mit nur ImageNet schneidet ViT schlechter ab — kein induktiver Bias.
- **DeiT** (2021) zeigte: mit guter Augmentation und Distillation reicht auch ImageNet.

#### ViT-Größen
| Modell | Layers | Heads | Hidden | Params |
|---|---|---|---|---|
| ViT-B/16 | 12 | 12 | 768 | 86M |
| ViT-L/16 | 24 | 16 | 1024 | 307M |
| ViT-H/14 | 32 | 16 | 1280 | 632M |
        """)

    with tabs[3]:
        section_header("Swin Transformer — Hierarchie zurück (2021)")
        st.markdown(r"""
ViT hat ein Problem: alle Patches mit allen anderen — **quadratische Komplexität** in der Anzahl Patches.
Bei hochauflösenden Bildern oder Videos wird das untragbar.

#### Swin's Lösung
- **Lokale Attention** in nicht-überlappenden Fenstern (z.B. 7×7)
- **Shifted Windows** zwischen den Layern — so kommunizieren Fenster doch noch
- **Hierarchisch** wie ein CNN: Auflösung halbieren, Channels verdoppeln

Resultat: gleiche Eingabegröße, aber **lineare Komplexität** und besser für Detection/Segmentation.

#### Wo Swin glänzt
- Object Detection (Mask R-CNN mit Swin-Backbone)
- Semantic Segmentation
- Hochauflösende Inputs allgemein
        """)

    with tabs[4]:
        section_header("ViT vs. CNN — wann was?")
        st.markdown(r"""
| Aspekt | CNN | Transformer (ViT) |
|---|---|---|
| **Induktiver Bias** | Stark (Lokalität, Translation) | Schwach |
| **Datenhunger** | Funktioniert mit wenigen Daten | Braucht viele Daten oder pretrain |
| **Long-Range Beziehungen** | Schwach (nur tief im Netz) | Stark (jede Position sieht alle) |
| **Komplexität** | Linear in Pixeln | Quadratisch in Patches |
| **Interpretierbarkeit** | Mittel (Filter-Visualisierung) | Hoch (Attention-Maps) |
| **Geschwindigkeit** | Schnell | Schnell, aber Speicher-hungrig |

#### Die heutige Realität
- **Vortrainiert + Finetuned**: ViT/Swin-basiert dominieren Benchmarks
- **From scratch auf kleinen Datasets**: ConvNeXt oder ResNet
- **Echtzeit auf Edge-Geräten**: MobileNet, EfficientNet
- **Foundation Models (DINO, CLIP, SAM)**: Fast immer ViT-basiert
        """)
        info_box(
            "Die Konvergenz hat begonnen: ConvNeXt nimmt Ideen von Transformern, "
            "Hierarchical Transformers (Swin) nehmen Ideen von CNNs. Das Beste aus beiden Welten.",
            kind="tip",
        )
