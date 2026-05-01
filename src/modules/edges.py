"""Kantendetektion — Sobel, Canny, Laplace."""
import streamlit as st
import numpy as np
import cv2
from src.components import hero, section_header, divider, info_box


def render():
    hero(
        eyebrow="Bildverarbeitung · Modul 7",
        title="Kantendetektion",
        sub="Wo ändert sich die Helligkeit stark? Genau dort sind Kanten — und dort fängt Bildverstehen an. "
            "Sobel, Canny, Laplace — die Klassiker."
    )

    tabs = st.tabs(["📐 Theorie", "🔵 Sobel", "🎯 Canny", "🌊 Laplace", "🧪 Live-Demo"])

    with tabs[0]:
        section_header("Was ist eine Kante?")
        st.markdown(r"""
Eine **Kante** ist ein Ort, an dem sich die Bildhelligkeit schnell ändert.
Mathematisch: hoher **Gradient** $\nabla I = (\partial I/\partial x, \partial I/\partial y)$.

#### Magnitude und Richtung
$$|\nabla I| = \sqrt{G_x^2 + G_y^2}, \quad \theta = \arctan\frac{G_y}{G_x}$$

- $|\nabla I|$: Wie stark ist die Kante?
- $\theta$: In welche Richtung verläuft die Kante?

#### Strategie aller Edge-Detektoren
1. **Glätten** (Gauß) — Rauschen reduzieren
2. **Gradient berechnen** (Sobel oder ähnlich)
3. **Magnitude** schwellwertieren
4. (optional) **Non-Maximum Suppression** und **Hysterese** (Canny)
        """)

    with tabs[1]:
        section_header("Sobel — der Klassiker")
        st.markdown(r"""
Sobel berechnet $G_x$ und $G_y$ über zwei 3×3 Kernel:

$$G_x = \begin{pmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{pmatrix} \quad
  G_y = \begin{pmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{pmatrix}$$

Beachte das **Smoothing-Element**: die Mitte hat doppeltes Gewicht — das ist Gauß-artig **integriert**.
Daher ist Sobel rauscharmer als ein nackter Differenzfilter.
        """)
        st.code("""
import cv2
gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
mag = cv2.magnitude(gx, gy)
        """, language="python")

    with tabs[2]:
        section_header("Canny — der Goldstandard")
        st.markdown(r"""
**Canny Edge Detector** (1986) ist immer noch der Standard für klassische Kantendetektion.
Fünf Schritte:

1. **Gauß-Smoothing** — Rauschen weg.
2. **Gradient via Sobel** — $G_x, G_y, |\nabla|, \theta$.
3. **Non-Maximum Suppression** — entlang der Gradient-Richtung nur lokale Maxima behalten.
4. **Doppel-Schwellwert** (low, high):
   - $|\nabla| > \text{high}$: definitiv Kante
   - $|\nabla| < \text{low}$: definitiv keine Kante
   - dazwischen: nur, wenn mit starker Kante verbunden
5. **Hysterese**: Verfolgt schwache Kanten, die mit starken zusammenhängen.

Das Ergebnis sind dünne (1-Pixel breite) Kanten — perfekt für weitere Verarbeitung.
        """)
        st.code("""
edges = cv2.Canny(gray, threshold1=50, threshold2=150)
        """, language="python")

    with tabs[3]:
        section_header("Laplace — zweite Ableitung")
        st.markdown(r"""
$$\nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$$

Kanten sind **Nulldurchgänge** der zweiten Ableitung. Empfindlicher gegen Rauschen,
deshalb meist als **Laplacian of Gaussian (LoG)** kombiniert:

$$\text{LoG}(x, y) = -\frac{1}{\pi \sigma^4}\left(1 - \frac{x^2+y^2}{2\sigma^2}\right) e^{-(x^2+y^2)/2\sigma^2}$$

Verwandt: **Difference of Gaussians (DoG)** — Approximation von LoG durch Differenz zweier Gauß-Filter.
DoG ist die Basis von SIFT.
        """)

    with tabs[4]:
        section_header("Live-Demo: alle Detektoren vergleichen")
        uploaded = st.file_uploader("Bild hochladen", type=["png", "jpg", "jpeg"], key="edge_upload")

        if uploaded:
            file_bytes = np.frombuffer(uploaded.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        else:
            # Demo: Gradient + Kreis + Rechteck
            img = np.zeros((300, 300, 3), dtype=np.uint8)
            for y in range(256):
                img[y, :] = (y, y, y)
            cv2.circle(img, (150, 150), 60, (255, 255, 255), -1)
            cv2.rectangle(img, (50, 50), (110, 110), (0, 0, 0), -1)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 1.0)

        c1, c2 = st.columns(2)
        low = c1.slider("Canny low",  0, 255, 50)
        high = c2.slider("Canny high", 0, 255, 150)

        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.uint8(np.clip(cv2.magnitude(gx, gy), 0, 255))
        laplace = np.uint8(np.clip(np.abs(cv2.Laplacian(gray, cv2.CV_64F, ksize=3)), 0, 255))
        canny = cv2.Canny(gray, low, high)

        cols = st.columns(4)
        cols[0].markdown("**Original**");        cols[0].image(gray, clamp=True, use_container_width=True)
        cols[1].markdown("**Sobel Magnitude**"); cols[1].image(sobel, clamp=True, use_container_width=True)
        cols[2].markdown("**Laplace**");          cols[2].image(laplace, clamp=True, use_container_width=True)
        cols[3].markdown(f"**Canny ({low}/{high})**"); cols[3].image(canny, clamp=True, use_container_width=True)

        info_box(
            "Sobel zeigt graue Kanten unterschiedlicher Stärke. Canny zeigt nur dünne, binäre Kanten — "
            "ideal als Eingang für weitere Algorithmen wie Hough-Transformation oder Konturen.",
            kind="info",
        )
