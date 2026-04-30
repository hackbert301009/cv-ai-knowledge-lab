"""Feature Detection & Matching."""
import streamlit as st
import numpy as np
import cv2
from src.components import hero, section_header, divider, info_box


def render():
    hero(
        eyebrow="Bildverarbeitung · Modul 8",
        title="Feature Detection &amp; Matching",
        sub="Wie findet man dieselbe Stelle in zwei Bildern wieder? "
            "Keypoints, Deskriptoren, Matching — die Basis von Panorama-Stitching, SLAM und 3D-Rekonstruktion."
    )

    tabs = st.tabs(["🎯 Was sind Features?", "📍 Harris", "✨ SIFT", "⚡ ORB", "🔗 Matching", "🧪 Live-Demo"])

    with tabs[0]:
        section_header("Was sind gute Features?")
        st.markdown(r"""
Ein **Feature** ist ein **Keypoint** + **Deskriptor**:
- **Keypoint**: Eine Position $(x, y)$ und meist auch Skala und Orientierung.
- **Deskriptor**: Ein Vektor (z.B. 128-d), der die Umgebung des Keypoints beschreibt.

#### Was macht ein Feature gut?
1. **Wiederholbar (repeatable)** — gleicher Punkt wird in verschiedenen Bildern wiedergefunden.
2. **Invariant** — gegen Rotation, Skalierung, Beleuchtung.
3. **Diskriminativ** — Deskriptor ist unterscheidbar von anderen.
4. **Effizient** — schnell zu berechnen.

#### Typische Anwendungen
- **Panorama Stitching** — gleiche Punkte in zwei Fotos finden, zusammensetzen
- **SLAM** (Simultaneous Localization and Mapping) — Roboternavigation
- **Object Recognition** — vor Deep Learning der Standard
- **Bildregistrierung** — z.B. medizinische Bilder ausrichten
        """)

    with tabs[1]:
        section_header("Harris Corner Detector")
        st.markdown(r"""
Harris (1988) findet **Ecken** — Stellen, an denen sich die Intensität in **mehrere Richtungen** ändert.

Idee: Verschiebe ein Fenster um $(\Delta x, \Delta y)$ und berechne die Intensitätsänderung:

$$E(\Delta x, \Delta y) = \sum_{x,y} w(x,y) [I(x+\Delta x, y+\Delta y) - I(x,y)]^2$$

Approximation mit Taylor:
$$E \approx \begin{pmatrix} \Delta x & \Delta y \end{pmatrix} \mathbf{M} \begin{pmatrix} \Delta x \\ \Delta y \end{pmatrix}$$

mit der Strukturmatrix $\mathbf{M}$.
Die **Eigenwerte** $\lambda_1, \lambda_2$ von $\mathbf{M}$ verraten die Region:
- Beide klein: flache Region
- Einer groß, einer klein: Kante
- Beide groß: **Ecke** ✓

Harris-Antwort: $R = \det(\mathbf{M}) - k \cdot \text{tr}(\mathbf{M})^2$.
        """)

    with tabs[2]:
        section_header("SIFT — Scale-Invariant Feature Transform")
        st.markdown(r"""
**SIFT** (Lowe, 1999/2004) ist der historisch wichtigste Feature-Detektor. Skalen- und rotationsinvariant.

#### Pipeline
1. **Skalenraum** mit Difference-of-Gaussians (DoG)
2. **Keypoint-Lokalisierung** als Extrema in Position und Skala
3. **Orientierungszuweisung** anhand dominantes Gradient-Histogramm
4. **Deskriptor** — 128-d Vektor aus 4×4 Gradient-Histogrammen á 8 Bins

#### Eigenschaften
- ✅ Sehr robust, hochwertig
- ❌ Patentiert (bis 2020), langsam
- 🔄 Heute oft durch ORB oder Deep Learning ersetzt
        """)
        st.code("""
import cv2
sift = cv2.SIFT_create()
kp, desc = sift.detectAndCompute(gray, None)
print(len(kp), desc.shape)   # z.B. 500, (500, 128)
        """, language="python")

    with tabs[3]:
        section_header("ORB — Oriented FAST and Rotated BRIEF")
        st.markdown(r"""
**ORB** (2011) ist eine **freie und schnelle** Alternative zu SIFT.

#### Komponenten
- **FAST** für Keypoint-Detektion (super schnell)
- **BRIEF** für Deskriptor (binär — sehr klein)
- **Orientierung** wird zusätzlich berechnet (BRIEF hat das nicht)

#### Vorteile
- ✅ Open Source (frei nutzbar)
- ✅ ~10× schneller als SIFT
- ✅ Binärer Deskriptor → Matching mit Hamming-Distanz, sehr schnell
- ⚠️ Etwas weniger robust als SIFT
        """)
        st.code("""
import cv2
orb = cv2.ORB_create(nfeatures=500)
kp, desc = orb.detectAndCompute(gray, None)
print(desc.shape, desc.dtype)   # (500, 32) uint8 — binär
        """, language="python")

    with tabs[4]:
        section_header("Feature Matching")
        st.markdown(r"""
Nachdem du Features in zwei Bildern hast, willst du sie **zuordnen**:
"Das ist derselbe Punkt".

#### Brute-Force Matching
Für jeden Deskriptor in Bild A: finde den ähnlichsten in Bild B.
- Float-Deskriptoren (SIFT): L2-Distanz
- Binär-Deskriptoren (ORB): Hamming-Distanz

#### Lowe's Ratio Test
Statt nur den besten Match zu nehmen, vergleiche die zwei besten:
$$\frac{d_1}{d_2} < 0.75$$
nur dann ist Match wahrscheinlich korrekt.

#### RANSAC
Outliers entfernen, indem du wiederholt eine Hypothese (z.B. Homographie)
aus zufälligen Sample-Matches schätzt und Inliers zählst.
        """)

    with tabs[5]:
        section_header("Live-Demo: Features in deinem Bild")
        uploaded = st.file_uploader("Bild hochladen (oder Demo nutzen)", type=["png", "jpg", "jpeg"], key="feat_upload")

        if uploaded:
            file_bytes = np.frombuffer(uploaded.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        else:
            img = np.zeros((300, 400, 3), dtype=np.uint8)
            cv2.rectangle(img, (50, 50), (150, 150), (180, 180, 180), -1)
            cv2.circle(img, (280, 100), 50, (200, 100, 200), -1)
            cv2.line(img, (50, 220), (350, 220), (255, 255, 255), 3)
            for x in range(20, 380, 30):
                cv2.circle(img, (x, 260), 5, (100, 200, 100), -1)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        n_features = st.slider("Max. Features", 50, 1000, 300)
        orb = cv2.ORB_create(nfeatures=n_features)
        kp, _ = orb.detectAndCompute(gray, None)
        out = cv2.drawKeypoints(img, kp, None,
                                color=(124, 58, 237),
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        st.image(out_rgb, caption=f"{len(kp)} ORB-Keypoints gefunden", use_container_width=True)

        info_box(
            "Die Größe der Kreise entspricht der Skala, an der das Feature detektiert wurde. "
            "Linien zeigen die Orientierung.",
            kind="tip",
        )
