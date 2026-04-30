"""Mathe-Crashkurs — die wichtigsten Konzepte kompakt."""
import streamlit as st
from src.components import hero, section_header, divider, info_box


def render():
    hero(
        eyebrow="Grundlagen · Modul 1",
        title="Mathe-Crashkurs",
        sub="Die mathematische Sprache von CV und KI auf einen Blick. "
            "Keine Beweise, keine Trockenheit — nur das, was du wirklich brauchst."
    )

    tab1, tab2, tab3, tab4 = st.tabs(["📝 Notation", "🎯 Skalare → Tensoren", "🔢 Operationen", "📚 Cheatsheet"])

    with tab1:
        section_header("Notation, die du immer wieder sehen wirst")
        st.markdown(r"""
| Symbol | Bedeutung | Beispiel |
|---|---|---|
| $x$ | Skalar | $x = 3.14$ |
| $\mathbf{x}$ | Vektor (klein, fett) | $\mathbf{x} \in \mathbb{R}^{n}$ |
| $\mathbf{X}$ | Matrix (groß, fett) | $\mathbf{X} \in \mathbb{R}^{m \times n}$ |
| $\mathcal{X}$ | Tensor / Menge | Bild-Tensor $\mathcal{X} \in \mathbb{R}^{H \times W \times C}$ |
| $\|\mathbf{x}\|_2$ | L2-Norm | $\sqrt{\sum_i x_i^2}$ |
| $\langle \mathbf{x}, \mathbf{y} \rangle$ | Skalarprodukt | $\sum_i x_i y_i$ |
| $\nabla f$ | Gradient | $(\partial f/\partial x_1, \dots)$ |
| $\mathbb{E}[X]$ | Erwartungswert | $\sum_x x \cdot p(x)$ |
""")

    with tab2:
        section_header("Vom Skalar zum Tensor")
        st.markdown(r"""
**Skalar** — eine Zahl. Pixelhelligkeit eines Graustufenbildes an Position (3, 5): $x = 127$.

**Vektor** — eine Liste. Ein RGB-Pixel: $\mathbf{p} = (210, 87, 34)$.

**Matrix** — eine 2D-Tabelle. Ein 8×8 Graustufenbild ist eine $\mathbf{X} \in \mathbb{R}^{8 \times 8}$.

**Tensor** — beliebige Dimension. Ein Farbbild ist ein 3D-Tensor:
$$\mathcal{X} \in \mathbb{R}^{H \times W \times 3}$$

Ein Batch von Bildern für ein neuronales Netz ist 4D:
$$\mathcal{B} \in \mathbb{R}^{N \times C \times H \times W}$$
        """)
        info_box(
            "Merke: PyTorch nutzt $(N, C, H, W)$ — Batch, Channels, Height, Width. "
            "TensorFlow nutzt $(N, H, W, C)$. Das ist der häufigste Bug-Grund beim Wechsel.",
            kind="warn",
        )

    with tab3:
        section_header("Operationen, die du verstehen musst")
        st.markdown(r"""
### Skalarprodukt
$$\mathbf{a} \cdot \mathbf{b} = \sum_i a_i b_i$$
Misst Ähnlichkeit zweier Vektoren. **Cosine Similarity** ist normalisiertes Skalarprodukt — Basis von CLIP & Co.

### Matrixmultiplikation
$$(\mathbf{A}\mathbf{B})_{ij} = \sum_k A_{ik} B_{kj}$$
**Das** Herzstück jedes neuronalen Netzes. Forward-Pass eines Layers: $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$.

### Transponieren
$\mathbf{A}^\top$ tauscht Zeilen und Spalten. Wichtig für: Self-Attention, Backprop.

### Outer Product
$$\mathbf{a} \otimes \mathbf{b} = \mathbf{a}\mathbf{b}^\top$$
Ergibt eine Matrix. Brauchst du, wenn du Attention-Maps verstehen willst.

### Norm
$$\|\mathbf{x}\|_p = \left(\sum_i |x_i|^p\right)^{1/p}$$
- $p=1$: L1-Norm (Manhattan), erzeugt sparse Lösungen
- $p=2$: L2-Norm (Euklidisch), Standard für Distanz
- $p=\infty$: Maximum

### Gradient
$$\nabla f(\mathbf{x}) = \begin{pmatrix} \partial f / \partial x_1 \\ \vdots \\ \partial f / \partial x_n \end{pmatrix}$$
Zeigt in die Richtung des steilsten Anstiegs. Gradient Descent geht $-\nabla f$ entlang.
        """)

    with tab4:
        section_header("Cheatsheet: Was wofür?")
        st.markdown("""
| Du willst... | Brauchst... | Modul |
|---|---|---|
| Bilder als Daten verstehen | Tensoren, Matrixops | Lineare Algebra |
| NNs trainieren | Gradienten, Kettenregel | Analysis |
| Klassifikation verstehen | Cross-Entropy, Softmax | Wahrscheinlichkeit |
| CNNs verstehen | Faltung als Operation | Lineare Algebra + Filter |
| Attention verstehen | Skalarprodukte + Softmax | Lineare Algebra |
| Diffusion verstehen | SDEs, Wahrscheinlichkeit | Wahrscheinlichkeit (advanced) |
| Backprop selbst rechnen | Jacobi-Matrizen | Analysis (advanced) |
""")

    divider()
    info_box(
        "Lass dich nicht von der Notation einschüchtern. Lies die Formeln laut vor, "
        "übersetze sie in Worte. Mit der Zeit wird Mathe zu einer zweiten Sprache.",
        kind="tip",
    )
