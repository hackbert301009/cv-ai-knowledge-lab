"""Klassische Segmentierung: Threshold, Watershed, GrabCut, K-Means."""
import streamlit as st
import numpy as np
import cv2
from src.components import hero, section_header, divider, info_box


def render():
    hero(
        eyebrow="Bildverarbeitung · Modul 10",
        title="Klassische Segmentierung",
        sub="Bevor Deep Learning Segmentierung übernahm, gab es eine Reihe genialer Ideen — "
            "und sie sind immer noch oft die schnellste, beste Wahl."
    )

    tabs = st.tabs(["🌗 Threshold", "🌊 Watershed", "✂️ GrabCut", "🎨 K-Means", "🧪 Live"])

    with tabs[0]:
        section_header("Schwellwertverfahren")
        st.markdown(r"""
Das Einfachste: Pixel über einem Schwellwert sind Vordergrund, der Rest Hintergrund.

#### Globaler Schwellwert
$$\text{out}(x, y) = \begin{cases} 255 & I(x,y) > T \\ 0 & \text{sonst} \end{cases}$$

#### Otsu — automatischer Schwellwert
Otsu (1979) findet den Schwellwert, der die **Varianz zwischen Klassen** maximiert.
Funktioniert perfekt, wenn das Histogramm bimodal ist (zwei klare Peaks).

#### Adaptiver Schwellwert
Schwellwert pro Region — sinnvoll bei ungleichmäßiger Beleuchtung.
        """)
        st.code("""
import cv2
# Binary
_, b1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Otsu (T wird automatisch gewählt)
_, b2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Adaptiv
b3 = cv2.adaptiveThreshold(gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        """, language="python")

    with tabs[1]:
        section_header("Watershed — Wasserscheiden-Algorithmus")
        st.markdown(r"""
Stell dir das Bild als Landschaft vor (helle Pixel = Berge, dunkle = Täler).
**Lass Wasser aufsteigen** — wo zwei Wasserquellen aufeinandertreffen, ist eine Grenze.

#### Pipeline (typisch)
1. Distanztransformation
2. Marker setzen (was sicher Vordergrund / Hintergrund ist)
3. Watershed füllt den Rest auf

Funktioniert sehr gut, um sich **berührende Objekte** zu trennen — z.B. Zellen unter dem Mikroskop.
        """)

    with tabs[2]:
        section_header("GrabCut — interaktive Segmentierung")
        st.markdown(r"""
**GrabCut** (2004) ist Photoshop's "Smart Selection". Du gibst eine grobe Bounding Box vor,
der Algorithmus iteriert mit Graph Cut + GMM (Gaussian Mixture Model) auf Vordergrund- und Hintergrundfarben.

Sehr stark, wenn der Hintergrund einigermaßen einheitlich ist.
        """)
        st.code("""
import cv2, numpy as np
mask = np.zeros(img.shape[:2], np.uint8)
bgd, fgd = np.zeros((1,65), np.float64), np.zeros((1,65), np.float64)
rect = (50, 50, 400, 300)   # bounding box
cv2.grabCut(img, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')
out = img * mask2[:, :, np.newaxis]
        """, language="python")

    with tabs[3]:
        section_header("K-Means Color Clustering")
        st.markdown(r"""
Alle Pixel als Punkte im Farbraum betrachten und in $k$ Cluster gruppieren.
Jeder Pixel bekommt die Farbe seines Cluster-Zentrums.

Ergibt oft sehr stilisierte, "poster-artige" Bilder und kann als
preisgünstige Vor-Segmentierung dienen.
        """)

    with tabs[4]:
        section_header("Live-Demo")
        uploaded = st.file_uploader("Bild hochladen", type=["png", "jpg", "jpeg"], key="seg_upload")

        if uploaded:
            file_bytes = np.frombuffer(uploaded.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        else:
            img = np.zeros((250, 350, 3), dtype=np.uint8)
            cv2.circle(img, (100, 130), 60, (180, 90, 200), -1)
            cv2.rectangle(img, (180, 60), (320, 200), (60, 200, 180), -1)
            for _ in range(2000):
                y, x = np.random.randint(0, 250), np.random.randint(0, 350)
                img[y, x] = np.clip(img[y, x] + np.random.randint(-20, 20, 3), 0, 255)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        method = st.radio("Methode", ["Otsu Threshold", "Adaptive Threshold", "K-Means (k=4)"], horizontal=True)

        if method == "Otsu Threshold":
            _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            display = out
        elif method == "Adaptive Threshold":
            display = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
        else:
            data = img.reshape(-1, 3).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            out = centers[labels.flatten()].reshape(img.shape).astype(np.uint8)
            display = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

        c1, c2 = st.columns(2)
        c1.markdown("**Original**")
        c1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
        c2.markdown(f"**{method}**")
        c2.image(display, clamp=True, use_container_width=True)
