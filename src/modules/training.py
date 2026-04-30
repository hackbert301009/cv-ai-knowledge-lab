"""Training, Loss & Optimizer."""
import streamlit as st
from src.components import hero, section_header, divider, info_box


def render():
    hero(
        eyebrow="Deep Learning · Modul 13",
        title="Training, Loss &amp; Optimizer",
        sub="Wie ein Netz **wirklich** lernt: Loss-Funktionen, Optimizer, Learning-Rate Scheduling, "
            "Regularisierung. Die handwerklichen Tricks, die zwischen 70% und 95% Accuracy entscheiden."
    )

    tabs = st.tabs(["📉 Loss", "🚀 Optimizer", "📊 LR Schedule", "🛡️ Regularisierung", "🐛 Debug"])

    with tabs[0]:
        section_header("Loss-Funktionen")
        st.markdown(r"""
| Aufgabe | Loss | Formel |
|---|---|---|
| Binäre Klassifikation | Binary Cross-Entropy | $-\sum [y \log \hat{y} + (1-y)\log(1-\hat{y})]$ |
| Multi-Class Klassifikation | Cross-Entropy | $-\sum y_c \log \hat{y}_c$ |
| Regression | MSE | $\frac{1}{n}\sum (y - \hat{y})^2$ |
| Robuste Regression | Huber / Smooth L1 | $\begin{cases} 0.5 x^2 & \|x\| < 1 \\ \|x\| - 0.5 & \text{sonst}\end{cases}$ |
| Object Detection | Focal Loss | $-(1-\hat{y})^\gamma \log \hat{y}$ |
| Embedding Learning | Triplet / Contrastive | siehe SimCLR, CLIP |
| Segmentation | Dice / IoU | $1 - \frac{2 |Y \cap \hat{Y}|}{|Y| + |\hat{Y}|}$ |

#### Focal Loss — fürs unbalancierte Detection
$$\text{FL}(p) = -(1-p)^\gamma \log p$$
Wenn das Modell schon richtig liegt ($p \approx 1$), wird der Term winzig.
Das Modell konzentriert sich auf schwierige Beispiele.
        """)

    with tabs[1]:
        section_header("Optimizer")
        st.markdown(r"""
#### SGD (Stochastic Gradient Descent)
$$\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$$

Einfachster Optimizer. Mit **Momentum** wird's interessant:
$$v \leftarrow \mu v + \nabla_\theta \mathcal{L}, \quad \theta \leftarrow \theta - \eta v$$

Momentum hilft, durch flache Bereiche zu rollen und Schwankungen zu glätten.

#### Adam — der Default
Adam kombiniert Momentum mit **adaptiven Lernraten** pro Parameter:
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(Momentum)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(quadratische EMA)}$$
$$\theta \leftarrow \theta - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

#### AdamW
Wie Adam, aber **Weight Decay korrekt entkoppelt** von der Adam-Update-Regel.
Standard für Transformers, ViT, ConvNeXt.

#### Lion (2023)
Neuer, einfacher Optimizer, der manchmal Adam übertrifft. Nur Vorzeichen wird benutzt — sehr speichereffizient.
        """)
        info_box(
            "Faustregel: AdamW mit lr=1e-3 oder 3e-4 als Default. "
            "Bei Vision: SGD + Momentum oft besser für die letzten Prozente Accuracy.",
            kind="tip",
        )

    with tabs[2]:
        section_header("Learning Rate Scheduling")
        st.markdown(r"""
Die wichtigste Hyperparameter — und sie sollte sich über die Zeit ändern.

| Schedule | Beschreibung |
|---|---|
| **Step Decay** | Alle $N$ Epochen LR halbieren |
| **Cosine Annealing** | Glatt von $\eta$ auf 0 sinken |
| **Warmup + Cosine** | Erst hochfahren (warmup), dann cosine. Standard für Transformer. |
| **One-Cycle** | Hoch → runter → noch tiefer. Schnelles Konvergieren (fast.ai) |
| **ReduceLROnPlateau** | LR halbieren, wenn Loss plateaut |

#### Warum Warmup?
Am Anfang sind Gewichte zufällig — große Updates können das Netz destabilisieren.
Linear hochfahren über die ersten paar tausend Steps gibt dem Netz Zeit, sich zu sortieren.
        """)

    with tabs[3]:
        section_header("Regularisierung — gegen Overfitting")
        st.markdown(r"""
| Technik | Was macht sie? |
|---|---|
| **Weight Decay** ($\ell_2$) | Bestraft große Gewichte: $\mathcal{L}_\text{total} = \mathcal{L} + \lambda \|\theta\|^2$ |
| **Dropout** | Zufällig ein Teil der Aktivierungen auf 0 setzen — beim Training |
| **Batch Norm** | Reguliert implizit; dazu Trainingsstabilität |
| **Data Augmentation** | Bilder rotieren, croppen, Farbe verschieben — mehr Varianz |
| **Label Smoothing** | One-Hot $[0,1,0]$ → $[0.05, 0.9, 0.05]$. Modell wird weniger overconfident |
| **MixUp / CutMix** | Bilder linear mischen, Labels auch — sehr starkes regularisierendes Augmentation |
| **Stochastic Depth** | Zufällig ganze Residual-Blöcke skippen |
| **Early Stopping** | Training abbrechen, wenn Validation-Loss steigt |
        """)

    with tabs[4]:
        section_header("Debug-Checkliste, wenn dein Netz nicht lernt")
        st.markdown("""
1. **Daten anschauen** — visualisiere ein Batch nach allen Augmentations. Sind Labels korrekt?
2. **Auf einem Mini-Batch overfitten** — kann das Netz 10 Beispiele perfekt lernen? Wenn nein, ist es zu klein oder Loss ist falsch.
3. **Lernrate prüfen** — meistens das Problem. Halbier sie / verdoppel sie und schau.
4. **Gradients prüfen** — `nn.utils.clip_grad_norm_` vor `optimizer.step()`. Großer Gradient-Norm = explodierende Gradienten.
5. **Verlust-Skalierung** — bei Mixed Precision unterläuft der Loss schnell. `GradScaler` benutzen.
6. **Dataset-Reihenfolge** — shuffle=True vergessen ist häufig.
7. **NaN-Check** — `torch.isnan(loss)` an strategischen Stellen.
8. **Klein anfangen** — kleines Modell, kleines Dataset, mache es zum Laufen, dann skalieren.
""")
        info_box(
            "Karpathy's berühmter Tipp: 'Werde eins mit dem Datensatz.' "
            "90% der Bugs sind Daten-Bugs, nicht Code-Bugs.",
            kind="warn",
        )
