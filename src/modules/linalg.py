"""Lineare Algebra für CV & KI."""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.components import hero, section_header, divider, info_box


def render():
    hero(
        eyebrow="Grundlagen · Modul 2",
        title="Lineare Algebra",
        sub="Vektoren, Matrizen, Tensoren — die DNA von Computer Vision und Deep Learning. "
            "Alles was sich in einem neuronalen Netz bewegt, ist Matrixmultiplikation."
    )

    tabs = st.tabs(["🎯 Warum?", "📐 Vektoren", "🔲 Matrizen", "🌌 Tensoren", "👁️ Eigenwerte", "🧪 Interaktiv"])

    # ---------- Warum? ----------
    with tabs[0]:
        section_header("Warum Lineare Algebra für CV & KI?")
        st.markdown(r"""
Stell dir ein 224×224 Farbbild vor. Das sind $224 \times 224 \times 3 = 150{,}528$ Zahlen.
Jede einzelne Operation in einem neuronalen Netz — jede Faltung, jede Attention,
jede Aktivierung — ist im Kern **Lineare Algebra**.

**Drei Konzepte ändern alles:**

1. **Bilder = Tensoren.** Sobald du das verinnerlicht hast, kannst du jede Operation als Math denken.
2. **Layer = Matrixmultiplikation.** Ein Dense-Layer ist $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$. Mehr nicht.
3. **Embeddings = Vektoren in Bedeutungsraum.** Cosine Similarity zwischen Embeddings ist die ganze Magie hinter CLIP und Suchmaschinen.
        """)
        info_box("Wenn du das hier verstehst, hast du 80% der mathematischen Hürde von Deep Learning genommen.", kind="success")

    # ---------- Vektoren ----------
    with tabs[1]:
        section_header("Vektoren — Punkte im Raum")
        st.markdown(r"""
Ein Vektor $\mathbf{v} \in \mathbb{R}^n$ ist eine Liste von $n$ Zahlen — geometrisch ein Pfeil vom Ursprung zu einem Punkt.

#### Operationen die zählen
- **Addition:** $\mathbf{a} + \mathbf{b}$ — komponentenweise
- **Skalierung:** $\alpha \mathbf{a}$ — alle Komponenten mit $\alpha$ multiplizieren
- **Skalarprodukt:** $\mathbf{a} \cdot \mathbf{b} = \sum_i a_i b_i = \|\mathbf{a}\| \|\mathbf{b}\| \cos\theta$
- **Norm:** $\|\mathbf{a}\|_2 = \sqrt{\mathbf{a} \cdot \mathbf{a}}$

#### Cosine Similarity — der wichtigste Use Case
$$\text{cos\_sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$$

Wert zwischen $-1$ (entgegengesetzt) und $+1$ (parallel). Wenn CLIP entscheidet,
ob ein Bild zu "ein Hund auf einer Wiese" passt, berechnet es **genau das**.
        """)
        st.code("""
import numpy as np

# Zwei Embeddings (z.B. von CLIP)
img_emb  = np.array([0.2, 0.8, -0.3, 0.5])
text_emb = np.array([0.3, 0.7, -0.2, 0.4])

# Cosine Similarity
cos = (img_emb @ text_emb) / (np.linalg.norm(img_emb) * np.linalg.norm(text_emb))
print(f"Similarity: {cos:.3f}")  # ~0.99 — sehr ähnlich
        """, language="python")

    # ---------- Matrizen ----------
    with tabs[2]:
        section_header("Matrizen — Lineare Transformationen")
        st.markdown(r"""
Eine Matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ ist nicht nur eine Tabelle —
sie ist eine **Funktion**, die Vektoren aus $\mathbb{R}^n$ in $\mathbb{R}^m$ transformiert.

#### Geometrische Bedeutung
- **Rotation:** $\mathbf{R}_\theta = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$
- **Skalierung:** $\mathbf{S} = \begin{pmatrix} s_x & 0 \\ 0 & s_y \end{pmatrix}$
- **Scherung, Spiegelung, Projektion** — alles Matrizen.

In CV nennt man diese **affine Transformationen** und benutzt sie für Bildregistrierung,
Augmentation, Homographien.

#### Matrixmultiplikation = Komposition
$\mathbf{A}\mathbf{B}$ heißt: **erst $\mathbf{B}$ anwenden, dann $\mathbf{A}$**.
Genau so funktioniert ein neuronales Netz: Layer für Layer Komposition linearer Transformationen
(plus Nichtlinearitäten dazwischen — sonst wäre alles eine einzige Matrix).
        """)
        st.code("""
import numpy as np

# Bild als Matrix (8x8 Graustufe)
img = np.random.rand(8, 8)

# Forward-Pass eines Dense-Layers
W = np.random.randn(16, 64) * 0.1   # Gewichte
b = np.random.randn(16) * 0.1       # Bias

x = img.flatten()                   # 64-d Vektor
y = W @ x + b                       # 16-d Output
print(y.shape)                      # (16,)
        """, language="python")

    # ---------- Tensoren ----------
    with tabs[3]:
        section_header("Tensoren — höherdimensionale Daten")
        st.markdown(r"""
Tensor ist einfach das Wort für "Array beliebiger Dimension":
- **0D**: Skalar
- **1D**: Vektor
- **2D**: Matrix
- **3D+**: Tensor

#### Typische Shapes in CV
| Daten | Shape (PyTorch) |
|---|---|
| Graustufenbild | $(H, W)$ |
| Farbbild | $(C, H, W)$ |
| Bild-Batch | $(N, C, H, W)$ |
| Video-Batch | $(N, T, C, H, W)$ |
| Embedding-Set | $(N, D)$ |

#### Broadcasting
NumPy/PyTorch erweitern automatisch kleinere Tensoren auf passende Form:
        """)
        st.code("""
import numpy as np

batch = np.random.rand(32, 3, 224, 224)   # (N, C, H, W)
mean  = np.array([0.485, 0.456, 0.406])   # (3,) ImageNet means

# Broadcasting: subtrahiere Mean von jedem Channel
normalized = batch - mean[None, :, None, None]   # mean wird auf (1,3,1,1) erweitert
print(normalized.shape)  # (32, 3, 224, 224)
        """, language="python")

    # ---------- Eigenwerte ----------
    with tabs[4]:
        section_header("Eigenwerte & Eigenvektoren")
        st.markdown(r"""
Ein **Eigenvektor** $\mathbf{v}$ einer Matrix $\mathbf{A}$ ist ein Vektor, der bei der Transformation seine Richtung **nicht** ändert:

$$\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$$

$\lambda$ ist der zugehörige **Eigenwert**.

#### Wofür braucht man das in CV?
- **PCA** (Principal Component Analysis) — Dimensionsreduktion über Eigenvektoren der Kovarianzmatrix
- **SVD** — verallgemeinerte Eigenwertzerlegung, hinter Bildkompression und Empfehlungssystemen
- **Spectral Methods** — Graph-CNNs, Laplace-Eigenmaps
- **Stabilität von Trainings** — Eigenwerte der Hesse-Matrix sagen, ob du in einem Sattelpunkt sitzt

#### PCA in 5 Zeilen
        """)
        st.code("""
import numpy as np

X = np.random.randn(1000, 50)    # 1000 Datenpunkte, 50 Features
X_centered = X - X.mean(0)
cov = np.cov(X_centered.T)
eigvals, eigvecs = np.linalg.eigh(cov)
top_k = eigvecs[:, -10:]           # Top 10 Komponenten
X_reduced = X_centered @ top_k     # 1000 x 10
        """, language="python")

    # ---------- Interaktiv ----------
    with tabs[5]:
        section_header("Interaktiv: 2D-Transformation visualisieren")
        st.markdown("Wähle Parameter für eine 2×2 Transformationsmatrix und sieh, wie sie ein Quadrat verformt.")

        c1, c2, c3, c4 = st.columns(4)
        a = c1.slider("a (oben-links)", -2.0, 2.0, 1.0, 0.1, key="lin_a")
        b = c2.slider("b (oben-rechts)", -2.0, 2.0, 0.5, 0.1, key="lin_b")
        c = c3.slider("c (unten-links)", -2.0, 2.0, 0.0, 0.1, key="lin_c")
        d = c4.slider("d (unten-rechts)", -2.0, 2.0, 1.0, 0.1, key="lin_d")

        M = np.array([[a, b], [c, d]])
        square = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]])
        transformed = M @ square

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=square[0], y=square[1], mode="lines+markers",
                                  name="Original", line=dict(color="#7C3AED", width=3)))
        fig.add_trace(go.Scatter(x=transformed[0], y=transformed[1], mode="lines+markers",
                                  name="Transformiert", line=dict(color="#EC4899", width=3)))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(range=[-3, 3], gridcolor="#222"),
            yaxis=dict(range=[-3, 3], gridcolor="#222", scaleanchor="x"),
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

        det = a * d - b * c
        st.markdown(f"**Determinante**: `{det:.3f}` &nbsp;—&nbsp; "
                    f"{'flächenerhaltend' if abs(abs(det) - 1) < 0.05 else 'streckt' if abs(det) > 1 else 'staucht'} · "
                    f"{'orientierungserhaltend' if det > 0 else 'spiegelnd' if det < 0 else 'singulär (Kollaps)'}")
