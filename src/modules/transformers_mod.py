"""Transformer & Attention — von Attention Is All You Need bis ViT und Swin."""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.components import (
    hero, section_header, divider, info_box,
    video_embed, lab_header, key_concept, step_list, card, render_card_grid, render_learning_block,
)


def _softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def render():
    hero(
        eyebrow="State-of-the-Art · Modul 15",
        title="Transformer &amp; Attention",
        sub="Die Architektur, die alles verändert hat. Self-Attention, Vision Transformer, "
            "Swin — die neue Lingua Franca von Computer Vision und KI."
    )

    tabs = st.tabs([
        "⚡ Attention-Mechanismus",
        "🤖 Transformer-Block",
        "👁️ Vision Transformer (ViT)",
        "🪟 Swin Transformer",
        "🆚 ViT vs CNN",
        "🧪 Attention Lab",
        "🎬 Lernvideos",
        "🧭 Lernpfad & Übungen",
    ])

    # ------------------------------------------------------------------ #
    with tabs[0]:
        section_header("Self-Attention — die zentrale Idee",
                       "'Attention Is All You Need' (Vaswani et al., 2017)")
        st.markdown(r"""
Self-Attention erlaubt jedem Element einer Sequenz, **direkt** mit jedem anderen zu kommunizieren —
nicht durch lokale Convolutions, sondern global und in einem einzigen Schritt.

#### Das Problem, das Attention löst
Bei CNNs müssen Informationen über viele Faltungsschichten "wandern", um globale Beziehungen
zu erfassen. Ein Auge links oben sieht erst nach 20+ Layern das Mund rechts unten.

**Attention**: Beide Positionen kommunizieren **direkt** — in einem einzigen Layer.
        """)

        key_concept("❓", "Query (Q)", "Was sucht diese Position? Der Suchbegriff. "
                    "Wie eine Google-Suche: 'Ich suche nach Kontext zu diesem Wort/Patch.'")
        key_concept("🔑", "Key (K)", "Was biete ich an? Die Schlagwörter meines Inhalts. "
                    "Jede Position beschreibt sich selbst, damit andere sie finden können.")
        key_concept("💎", "Value (V)", "Mein eigentlicher Inhalt. "
                    "Wenn eine Position von mir 'Attention' bekommt, gibt sie ihren Value weiter.")

        st.markdown(r"""
#### Die Formel
$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

**Schritt für Schritt:**
1. $QK^\top$: Berechne Ähnlichkeit (Skalarprodukt) jeder Query mit jedem Key → Attention-Scores
2. $/\sqrt{d_k}$: Skaliere um $\sqrt{d_k}$ (verhindert zu große Werte bei hoher Dimensionalität)
3. $\text{softmax}(\cdot)$: Wandle in Wahrscheinlichkeiten (Attention-Gewichte) um
4. $\cdot V$: Gewichteter Mittelwert der Values = Ausgabe

#### Warum $\sqrt{d_k}$?
Ohne Skalierung: Skalarprodukte zweier $d_k$-dimensionaler Vektoren haben Varianz $\propto d_k$.
Bei $d_k = 64$ werden die Werte zu groß → Softmax sättigt → Gradienten verschwinden.
$\sqrt{d_k}$ normalisiert die Varianz auf 1.

#### Multi-Head Attention (MHA)
Statt einer Attention-Operation laufen $h$ **parallele Heads**, jeder mit eigenen $W^Q, W^K, W^V$:

$$\text{MHA}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

Verschiedene Heads können verschiedene Beziehungstypen lernen:
- Head 1: syntaktische Abhängigkeiten
- Head 2: semantische Ähnlichkeit
- Head 3: räumliche Nähe (in ViT)
        """)

    # ------------------------------------------------------------------ #
    with tabs[1]:
        section_header("Der Transformer-Block", "Die Grundeinheit, die N Mal gestapelt wird.")
        st.markdown(r"""
Ein **Transformer-Block** (Pre-LN Variante, heute Standard):

```
x → LayerNorm → Multi-Head Attention → + x → LayerNorm → FFN → + x → output
```

In Formeln:
$$\mathbf{x}' = \mathbf{x} + \text{MHA}(\text{LN}(\mathbf{x}))$$
$$\mathbf{y} = \mathbf{x}' + \text{FFN}(\text{LN}(\mathbf{x}'))$$

#### Die vier Schlüssel-Komponenten
        """)

        step_list([
            ("Layer Normalization (LN)",
             "Normalisiert entlang der Feature-Dimension (nicht Batch-Dimension wie BN). "
             "Stabiler für variable Sequenzlängen und kleine Batches."),
            ("Multi-Head Self-Attention (MHA)",
             "h parallele Attention-Operationen. Jeder Head lernt andere Beziehungen. "
             "Typisch: d_model=768, h=12 → d_k=d_v=64 pro Head."),
            ("Residual Connection (Add)",
             "x + MHA(...). Gradienten fließen direkt. Netz kann Attention-Block ignorieren. "
             "Essenziell für Training sehr tiefer Transformers (24-32+ Blöcke)."),
            ("Feed-Forward Network (FFN)",
             "Zwei lineare Schichten mit GELU. Dimension: d_model → 4×d_model → d_model. "
             "Speichert 'Fakten' (das wurde gezeigt von Geva et al., 2021)."),
        ])

        st.markdown("#### Positional Encoding — Sequenz-Reihenfolge einfügen")
        st.markdown(r"""
Attention ist **permutationsinvariant** — ob Patch 1 vor oder nach Patch 5 kommt, egal.
Positionen müssen explizit kodiert werden:

| Methode | Beschreibung | Wo |
|---|---|---|
| **Sinus/Cosinus** | $\text{PE}(pos, 2i) = \sin(pos/10000^{2i/d})$ | Original Transformer |
| **Lernbar (APE)** | Jede Position hat gelernten Embedding-Vektor | ViT, BERT |
| **Relativ (RPE)** | Bias abhängig von $i-j$ | T5, Swin |
| **RoPE** | Rotiert Q,K basierend auf Position | LLaMA, GPT-NeoX |
| **ALiBi** | Negativer Bias proportional zu $|i-j|$ | BLOOM |

RoPE ist 2024/25 dominant für LLMs. Swin nutzt Relative Positional Bias.
        """)

    # ------------------------------------------------------------------ #
    with tabs[2]:
        section_header("Vision Transformer (ViT)",
                       "Dosovitskiy et al., Google Brain, 2020 — 'An Image is Worth 16×16 Words'")
        st.markdown(r"""
**ViT** hatte die simple aber revolutionäre Idee: **Behandle ein Bild wie Text-Tokens.**
Teile es in Patches auf und schicke sie durch einen Standard-Transformer.
        """)

        step_list([
            ("Bild in Patches aufteilen",
             "224×224 Bild → 196 Patches je 16×16. Jeder Patch wird flach gemacht: 16×16×3 = 768 Werte."),
            ("Lineare Projektion",
             "Jeder Patch wird auf einen d_model-dimensionalen Vektor projiziert (lernbare lineare Schicht)."),
            ("CLS-Token prependen",
             "Ein spezielles [CLS]-Token wird vorne angehängt (insgesamt 197 Tokens). "
             "Am Ende repräsentiert es das gesamte Bild."),
            ("Positional Embeddings addieren",
             "Lernbare 1D Positional Embeddings — Modell lernt räumliche Beziehungen zwischen Patches."),
            ("N Transformer-Blöcke",
             "ViT-B: 12 Blöcke, 12 Heads, d=768. ViT-L: 24 Blöcke, 16 Heads, d=1024."),
            ("Klassifikation über CLS-Token",
             "Der CLS-Token nach dem letzten Block → Linear → Klasse."),
        ])

        st.markdown(r"""
#### ViT-Größen
| Modell | Blöcke | Heads | d_model | Parameter |
|---|---|---|---|---|
| ViT-T/16 | 12 | 3 | 192 | 5.7M |
| **ViT-S/16** | 12 | 6 | 384 | 22M |
| **ViT-B/16** | 12 | 12 | 768 | 86M |
| ViT-L/16 | 24 | 16 | 1024 | 307M |
| ViT-H/14 | 32 | 16 | 1280 | 632M |

#### Der Aha-Moment: Daten-Hunger
- Mit nur ImageNet (1.28M Bilder): ViT verliert gegen ResNet
- Mit JFT-300M (300M Bilder): ViT gewinnt klar
- Mit guter Augmentation + Distillation (DeiT): reicht auch ImageNet

**Fazit**: CNNs haben starken induktiven Bias (Lokalität, Translation) — gut mit wenig Daten.
ViT hat wenig induktiven Bias — braucht mehr Daten oder pretraining.
        """)

        info_box(
            "Heute: Fast alle Foundation Models nutzen ViT. CLIP, DINO, SAM, DINOv2, MAE — "
            "alle auf ViT-Basis. Du wirst ViT für den Rest deines CV-Lebens sehen.",
            kind="tip",
        )

    # ------------------------------------------------------------------ #
    with tabs[3]:
        section_header("Swin Transformer", "Liu et al., Microsoft Research, 2021")
        st.markdown(r"""
**ViTs Problem**: Attention über alle 196 Patches = quadratische Komplexität $O(N^2)$.
Bei $512 \times 512$ Bild: $1024$ Patches → $1024^2 = 1M$ Operationen pro Layer.
Für hochauflösende Bilder oder Videos ist das untragbar.

#### Swins Lösung: Hierarchische Windowed Attention
        """)

        key_concept("🪟", "Window Attention",
                    "Attention nur innerhalb nicht-überlappender 7×7 Windows. "
                    "Komplexität O(N) statt O(N²) — linear in der Bildgröße.")
        key_concept("🔀", "Shifted Windows",
                    "Zwischen Layern werden die Windows um (W/2, W/2) verschoben. "
                    "Dadurch kommunizieren benachbarte Windows indirekt — globale Information kann fließen.")
        key_concept("📐", "Hierarchie (Patch Merging)",
                    "Wie ResNet: Auflösung halbieren, Channels verdoppeln. "
                    "Ermöglicht Multi-Scale Feature Maps für Detection & Segmentation.")

        st.markdown(r"""
#### Swin vs. ViT

| Aspekt | ViT | Swin |
|---|---|---|
| Attention-Reichweite | Global (alle Patches) | Lokal (Fenster) |
| Komplexität | $O(N^2)$ | $O(N)$ |
| Feature-Skalen | Nur eine | Hierarchisch (wie ResNet) |
| Für Detection | Schwierig | Ideal (FPN kompatibel) |
| Für Segmentation | Schwierig | Ideal |
| Recheneffizienz | Niedrig bei hoher Auflösung | Hoch |

#### Swin Varianten
| Modell | Parameter | ImageNet Top-1 |
|---|---|---|
| Swin-T (Tiny) | 28M | 81.3% |
| Swin-S (Small) | 50M | 83.0% |
| Swin-B (Base) | 88M | 83.5% |
| Swin-L (Large) | 197M | 86.3% |

#### Wo Swin heute eingesetzt wird
- Object Detection (DINO, DETA als Backbone)
- Semantic Segmentation (UperNet + Swin)
- Medical Image Segmentation
- Video Understanding (Video Swin)
        """)

    # ------------------------------------------------------------------ #
    with tabs[4]:
        section_header("ViT vs CNN — wann was?", "Eine ehrliche Gegenüberstellung.")
        st.markdown(r"""
| Aspekt | CNN (ResNet/ConvNeXt) | ViT | Swin |
|---|---|---|---|
| **Induktiver Bias** | Stark (Lokalität, Translation) | Schwach | Mittel |
| **Datenhunger** | Niedrig (gut mit wenig) | Hoch (braucht pretraining) | Mittel |
| **Long-Range** | Schwach (erst in tiefen Layern) | Stark (immer global) | Mittel (via shift) |
| **Komplexität** | O(N) | O(N²) | O(N) |
| **Interpretierbarkeit** | Mittel (CAM, Grad-CAM) | Hoch (Attention Maps) | Hoch |
| **Echtzeit / Edge** | Sehr gut (MobileNet etc.) | Schlecht | Mittel |
| **Backbone für Det/Seg** | Gut (ResNet+FPN) | Mittel | Sehr gut (Hierarchie) |
| **Pretraining verfügbar** | Viel (ImageNet) | Sehr viel (MAE, DINO etc.) | Viel |

#### Praktische Empfehlung 2026
        """)

        cards = [
            card("🚀", "Kleines Dataset (<10K Bilder)",
                 "ResNet-50 oder EfficientNet-B3 mit ImageNet pretrained. Transfer Learning ist Pflicht.",
                 ["Transfer Learning"], ["green"]),
            card("⚡", "Echtzeit / Edge / Mobile",
                 "MobileNetV3, EfficientNet-Lite, oder MobileViT. CNNs sind hier noch unschlagbar.",
                 ["Edge AI"], ["amber"]),
            card("🎯", "Detection / Segmentation",
                 "Swin oder ConvNeXt als Backbone. Bessere Hierarchie als reiner ViT.",
                 ["Detection"], ["blue"]),
            card("🌍", "Large-Scale / Foundation",
                 "ViT-Large oder ViT-Huge, pretrained auf JFT/ImageNet-21K. Dann finetunen.",
                 ["Foundation"], ["pink"]),
            card("🏭", "Produktionssystem",
                 "Was funktioniert + was das Team kennt. Benchmark auf eigenem Dataset. ONNX/TensorRT exportieren.",
                 ["Produktion"], ["amber"]),
            card("🔬", "Research",
                 "Swin-Transformer oder ViT als Baseline. Dann die neueste Architektur vom CVPR testen.",
                 ["Research"], ["pink"]),
        ]
        render_card_grid(cards, cols=3)

        info_box(
            "Die Grenze verschwimmt: ConvNeXt nutzt Ideen aus ViT (Layer Norm, depthwise Conv, GELU). "
            "Swin hat hierarchische Struktur wie CNN. Das Beste aus beiden Welten.",
            kind="tip",
        )

    # ------------------------------------------------------------------ #
    with tabs[5]:
        lab_header("Interaktives Attention Lab",
                   "Verstehe Self-Attention durch direktes Experimentieren.")

        st.markdown("#### Self-Attention auf einer Mini-Sequenz")
        st.markdown("""Stell dir vor, wir haben 6 Bild-Patches. Jeder hat ein zufälliges Embedding.
Wir berechnen, wie stark jeder Patch auf jeden anderen "achtet".""")

        n_tokens = st.slider("Anzahl Patches (Tokens)", 3, 10, 6)
        d_k = st.slider("Key-Dimensionalität d_k", 4, 32, 8)
        temperature = st.slider("Temperatur (1/√d_k · Skala)", 0.1, 5.0, 1.0, 0.1)
        seed = st.number_input("Seed (für Reproduzierbarkeit)", min_value=0, max_value=999, value=42)

        rng = np.random.default_rng(int(seed))
        Q = rng.standard_normal((n_tokens, d_k))
        K = rng.standard_normal((n_tokens, d_k))
        V = rng.standard_normal((n_tokens, d_k))

        scale = temperature / np.sqrt(d_k)
        scores = Q @ K.T * scale
        attn_weights = _softmax(scores, axis=-1)
        output = attn_weights @ V

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Attention-Matrix (Gewichte)**")
            token_labels = [f"P{i+1}" for i in range(n_tokens)]
            fig_attn = go.Figure(go.Heatmap(
                z=attn_weights,
                x=token_labels, y=token_labels,
                colorscale=[
                    [0.0, "#0B0B0F"], [0.4, "#4C1D95"],
                    [0.7, "#7C3AED"], [1.0, "#F59E0B"]
                ],
                showscale=True,
                colorbar=dict(title="Attention", thickness=10, len=0.7),
                text=[[f"{v:.2f}" for v in row] for row in attn_weights],
                texttemplate="%{text}",
                textfont=dict(size=9),
            ))
            fig_attn.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", height=320,
                margin=dict(l=40, r=10, t=10, b=40),
                xaxis=dict(title="Key-Token (Von wo kommt Attention?)"),
                yaxis=dict(title="Query-Token (Wer fragt?)"),
            )
            st.plotly_chart(fig_attn, use_container_width=True)

        with col_b:
            st.markdown("**Attention-Scores (vor Softmax)**")
            fig_scores = go.Figure(go.Heatmap(
                z=scores,
                x=token_labels, y=token_labels,
                colorscale="RdBu_r",
                showscale=True,
                colorbar=dict(title="Score", thickness=10, len=0.7),
                text=[[f"{v:.1f}" for v in row] for row in scores],
                texttemplate="%{text}",
                textfont=dict(size=9),
            ))
            fig_scores.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", height=320,
                margin=dict(l=40, r=10, t=10, b=40),
                xaxis=dict(title="Key"), yaxis=dict(title="Query"),
            )
            st.plotly_chart(fig_scores, use_container_width=True)

        # Attention-Entropie als Maß für "wie fokussiert" die Attention ist
        entropy = -np.sum(attn_weights * np.log(attn_weights + 1e-9), axis=-1)
        max_entropy = np.log(n_tokens)

        st.markdown(f"""
**Attention-Entropie (pro Token):**
- Mittlere Entropie: **{entropy.mean():.2f}** nats (max mögliche: **{max_entropy:.2f}**)
- Verhältnis: **{entropy.mean()/max_entropy*100:.0f}%** — {'diffuse (breit verteilte Attention)' if entropy.mean()/max_entropy > 0.7 else 'fokussierte Attention'}
""")

        info_box(
            "**Temperatur-Effekt:** Höhere Temperatur → gleichmäßigere Attention (wie Softmax gegen 1/N). "
            "Niedrigere Temperatur → schärfere, fokussiertere Attention. "
            "Das Skalieren mit 1/√d_k verhindert, dass Attention zu scharf wird.",
            kind="info",
        )

    # ------------------------------------------------------------------ #
    with tabs[6]:
        section_header("Lernvideos", "3Blue1Brown und Andrej Karpathy erklären Attention.")

        st.markdown("#### Attention in Transformers, visually explained (3Blue1Brown)")
        video_embed("eMlx5fFNoYc",
                    "Attention in transformers, visually explained — 3Blue1Brown",
                    "Kapitel 6 der Neural Networks Serie. Attention ist all you need — visuell erklärt.")

        divider()

        st.markdown("#### But what is a GPT? Visual intro to transformers (3Blue1Brown)")
        video_embed("wjZofJX0v4M",
                    "But what is a GPT? — 3Blue1Brown",
                    "Wie GPT und Transformer-basierte Modelle intern funktionieren. ~27 Minuten.")

        divider()

        st.markdown("#### Let's build GPT from scratch (Andrej Karpathy)")
        video_embed("kCc8FmEb1nY",
                    "Let's build GPT from scratch — Andrej Karpathy",
                    "Karpathy baut GPT komplett from scratch in PyTorch. Das definitive Tutorial. ~2 Stunden.")

        info_box(
            "Karpathys GPT-Tutorial ist lang, aber jede Minute wert. "
            "Nach diesem Video kannst du einen eigenen Transformer implementieren.",
            kind="tip",
        )

    # ------------------------------------------------------------------ #
    with tabs[7]:
        st.markdown("### Visual: Modell-Entscheidung")
        st.graphviz_chart(
            """
            digraph G {
                rankdir=TB;
                node [shape=box, style=rounded];
                A [label="Datensatzgröße?"];
                B [label="<10k: CNN/Transfer"];
                C [label(">50k + Pretraining: ViT/Swin")];
                D [label="Detection/Segmentation: Swin"];
                A -> B;
                A -> C;
                C -> D;
            }
            """
        )

        render_learning_block(
            key_prefix="transformers",
            section_title="Lernpfad für Transformer",
            section_sub="Progression mit Praxisfokus",
            progression=[
                ("🟢", "Guided Lab", "Baue Attention-Matrix für eine Mini-Sequenz und interpretiere Heads.", "Beginner", "green"),
                ("🟠", "Challenge Lab", "Vergleiche ViT und CNN auf kleinem Datensatz mit gleichem Budget.", "Intermediate", "amber"),
                ("🔴", "Debug Lab", "Finde Ursachen für instabile Attention oder schlechte Generalisierung.", "Advanced", "pink"),
                ("🏁", "Mini-Projekt", "Patch-Classifier mit Fehleranalyse und Kurzbericht.", "Abschluss", "blue"),
            ],
            mcq_question="Warum braucht ViT ohne starkes Pretraining oft mehr Daten als CNN?",
            mcq_options=[
                "Weil ViT keine Residual Connections hat",
                "Weil ViT weniger Parameter hat",
                "Weil ViT weniger induktiven Bias (Lokalität/Translation) besitzt",
                "Weil ViT nicht mit AdamW trainiert werden kann",
            ],
            mcq_correct_option="Weil ViT weniger induktiven Bias (Lokalität/Translation) besitzt",
            mcq_success_message="Exakt. Weniger Bias bedeutet oft höheren Datenbedarf.",
            mcq_retry_message="Noch nicht. Schau in den ViT-vs-CNN Vergleich.",
            open_question="Offene Frage: In welchem Produktfall würdest du Swin statt ViT-B einsetzen und warum?",
            code_task="""# Code-Aufgabe: Attention-Map pro Head mitteln
import torch

attn = torch.randn(12, 197, 197)  # 12 Heads
# TODO: gemittelte Attention über alle Heads berechnen
""",
            cheat_sheet=[
                "Kleine Daten: CNN + Transfer Learning.",
                "Große Daten / Foundation: ViT.",
                "Detection/Segmentation mit Hierarchie: Swin.",
            ],
            key_takeaways=[
                "Attention ermöglicht globale Abhängigkeiten in einem Schritt.",
                "Architekturwahl ist immer ein Trade-off aus Daten, Compute und Latenz.",
            ],
            common_errors=[
                "ViT ohne Pretraining auf sehr kleinen Datensätzen.",
                "Positions-Embeddings bei Auflösungswechsel falsch behandeln.",
                "Nur Accuracy vergleichen, keine Laufzeit/Memory-Metriken.",
                "Unterschiedliche Augmentationen zwischen Modellen.",
                "Fehlende Fehleranalyse der Aufmerksamkeitsmuster.",
            ],
        )
