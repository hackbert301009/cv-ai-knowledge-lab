"""Filter & Faltung — die Mutter aller CNNs."""
import streamlit as st
import numpy as np
import cv2
import plotly.graph_objects as go
from src.components import hero, section_header, divider, info_box


def render():
    hero(
        eyebrow="Bildverarbeitung · Modul 6",
        title="Filter &amp; Faltung",
        sub="Die fundamentale Operation: ein kleiner Kernel gleitet über das Bild und berechnet einen gewichteten Durchschnitt. "
            "Aus dieser Idee sind Convolutional Neural Networks entstanden."
    )

    tabs = st.tabs(["🌫️ Was ist Faltung?", "🎛️ Klassische Kernel", "🧪 Interaktiv", "🔗 Brücke zu CNN"])

    with tabs[0]:
        section_header("Faltung — die Formel")
        st.markdown(r"""
$$(I * K)(y, x) = \sum_{i=-k}^{k} \sum_{j=-k}^{k} I(y+i, x+j) \cdot K(i, j)$$

In Worten: Lege den Kernel $K$ (Matrix, oft 3×3 oder 5×5) auf jede Pixelposition,
multipliziere elementweise, summiere — das ist der neue Pixelwert.

#### Eigenschaften
- **Linear** und **shift-invariant** — gleicher Filter überall
- **Lokal** — nur Nachbarn beeinflussen den neuen Wert
- **Bestimmt durch Kernel** — kleine Matrix steuert Verhalten

#### Padding & Stride
- **Padding**: Wie behandeln wir den Rand? (zero, reflect, replicate)
- **Stride**: Wie weit springt der Kernel? (1 = jeder Pixel; 2 = halbiert Auflösung)
        """)

    with tabs[1]:
        section_header("Klassische Kernel — was wofür?")
        st.markdown("""
| Kernel | Effekt | Typische Werte |
|---|---|---|""")
        st.markdown(r"""
| **Identity** | Bild unverändert | $\begin{pmatrix} 0&0&0\\0&1&0\\0&0&0 \end{pmatrix}$ |
| **Box-Blur** | Mittelwert (verschwommen) | $\frac{1}{9}\begin{pmatrix} 1&1&1\\1&1&1\\1&1&1 \end{pmatrix}$ |
| **Gauß-Blur** | Gewichteter Blur — natürlicher | $\frac{1}{16}\begin{pmatrix} 1&2&1\\2&4&2\\1&2&1 \end{pmatrix}$ |
| **Sharpen** | Kanten verstärken | $\begin{pmatrix} 0&-1&0\\-1&5&-1\\0&-1&0 \end{pmatrix}$ |
| **Sobel-X** | Horizontale Kanten | $\begin{pmatrix} -1&0&1\\-2&0&2\\-1&0&1 \end{pmatrix}$ |
| **Sobel-Y** | Vertikale Kanten | $\begin{pmatrix} -1&-2&-1\\0&0&0\\1&2&1 \end{pmatrix}$ |
| **Laplace** | Zweite Ableitung — alle Kanten | $\begin{pmatrix} 0&-1&0\\-1&4&-1\\0&-1&0 \end{pmatrix}$ |
| **Emboss** | 3D-Relief-Effekt | $\begin{pmatrix} -2&-1&0\\-1&1&1\\0&1&2 \end{pmatrix}$ |
        """)

    with tabs[2]:
        section_header("Interaktiv: Filter ausprobieren")
        st.markdown("Lade ein Bild hoch (oder nutze das Demo-Bild) und probiere Filter aus.")

        uploaded = st.file_uploader("Bild hochladen", type=["png", "jpg", "jpeg"])

        if uploaded:
            file_bytes = np.frombuffer(uploaded.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # Demo-Bild generieren
            img = np.zeros((200, 200, 3), dtype=np.uint8)
            cv2.circle(img, (100, 100), 60, (240, 100, 200), -1)
            cv2.rectangle(img, (40, 40), (90, 90), (50, 200, 240), -1)
            cv2.line(img, (10, 180), (190, 180), (255, 255, 255), 3)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        kernels = {
            "Identity":  np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.float32),
            "Box-Blur":  np.ones((3,3), dtype=np.float32) / 9,
            "Gauß-Blur (5×5)": cv2.getGaussianKernel(5, 1.0) @ cv2.getGaussianKernel(5, 1.0).T,
            "Sharpen":   np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32),
            "Sobel-X":   np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32),
            "Sobel-Y":   np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32),
            "Laplace":   np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=np.float32),
            "Emboss":    np.array([[-2,-1,0],[-1,1,1],[0,1,2]], dtype=np.float32),
        }
        choice = st.selectbox("Kernel wählen", list(kernels.keys()), index=2)
        kernel = kernels[choice]

        filtered = cv2.filter2D(gray, -1, kernel)

        c1, c2 = st.columns(2)
        c1.markdown("**Original (Graustufe)**")
        c1.image(gray, clamp=True, use_container_width=True)
        c2.markdown(f"**Nach `{choice}`**")
        c2.image(filtered, clamp=True, use_container_width=True)

        st.markdown(f"**Verwendeter Kernel:**")
        st.code(np.array2string(kernel, precision=2, suppress_small=True), language="text")

    with tabs[3]:
        section_header("Vom klassischen Filter zum CNN")
        st.markdown(r"""
Die klassische Bildverarbeitung hat **handentworfene** Kernel benutzt — Sobel, Gauß, Laplace.
Ein **Convolutional Neural Network** macht genau das gleiche, aber:

#### CNNs **lernen** ihre Kernel
Statt einen Sobel-Kernel von Hand einzusetzen, initialisiert ein CNN die Kernel zufällig
und passt sie über Backprop an, um das gegebene Problem zu lösen.

#### Mehrere Kernel pro Layer
Ein Conv-Layer hat oft 32–512 verschiedene Kernel — jeder lernt ein anderes Feature
(Kanten in einer Richtung, Texturen, später ganze Augen, Gesichter, …).

#### Tiefe Hierarchie
- Layer 1: Lernt einfache Kanten und Farben (oft Sobel-ähnlich!)
- Layer 5: Lernt Texturen
- Layer 10: Lernt Objektteile
- Layer 20: Lernt ganze Kategorien
        """)
        info_box(
            "Wenn du einen Conv-Layer-Kernel eines trainierten CNN visualisierst, siehst du oft Strukturen, "
            "die genau wie Gabor-Filter oder Sobel aussehen. Das Netz hat klassische CV neu entdeckt.",
            kind="success",
        )
