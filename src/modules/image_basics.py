"""Bildgrundlagen — was ist ein Bild eigentlich?"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.components import hero, section_header, divider, info_box


def render():
    hero(
        eyebrow="Bildverarbeitung · Modul 5",
        title="Bildgrundlagen &amp; Pixel",
        sub="Bevor wir Bilder verarbeiten können, müssen wir verstehen, was sie sind. "
            "Pixel, Farbräume, Sampling, Quantisierung — die Anatomie digitaler Bilder."
    )

    tabs = st.tabs(["📸 Was ist ein Bild?", "🎨 Farbräume", "📐 Sampling", "🔢 Quantisierung", "💾 Formate"])

    with tabs[0]:
        section_header("Ein Bild = Funktion auf Pixeln")
        st.markdown(r"""
Ein digitales Bild ist eine Funktion:
$$I: \{1, \dots, H\} \times \{1, \dots, W\} \to \{0, 1, \dots, 255\}^C$$

Sie nimmt eine Pixelposition $(y, x)$ und gibt einen **Intensitätswert** zurück:
- **Graustufen** ($C=1$): Eine Zahl pro Pixel.
- **RGB** ($C=3$): Drei Zahlen — Rot, Grün, Blau.
- **RGBA** ($C=4$): Plus Alpha (Transparenz).

#### Pixel = Picture Element
Pixel sind diskret. Die reale Welt ist kontinuierlich. Ein Kamerasensor **abtastet** (sampelt)
die Welt an einem Raster und **quantisiert** die Helligkeit zu diskreten Werten.

#### Auflösung vs. Bittiefe
- **Auflösung**: Wie viele Pixel? (z.B. 1920×1080)
- **Bittiefe**: Wie viele Werte pro Pixel? (8 bit = 256 Werte; 16 bit = 65536; HDR-Formate haben Float)
        """)

    with tabs[1]:
        section_header("Farbräume — RGB ist nicht alles")
        st.markdown(r"""
| Farbraum | Achsen | Wofür gut? |
|---|---|---|
| **RGB** | Rot, Grün, Blau | Standard für Displays, Kameras |
| **HSV** | Hue, Saturation, Value | Farbbasierte Segmentierung — "alle roten Pixel" wird einfach |
| **LAB** | Helligkeit, a (grün-rot), b (blau-gelb) | Wahrnehmungsbasiert — Distanzen entsprechen Farbunterschieden |
| **YCbCr** | Helligkeit + Farb-Differenzen | JPEG-Kompression, TV-Übertragung |
| **Graustufen** | Eine Helligkeit | Wenn Farbe egal ist (oft in CV) |

#### Warum HSV zum Segmentieren so praktisch ist
Im RGB sind "alle Rot-Töne" über alle drei Kanäle verteilt. In HSV ist Rot ein **Bereich auf einer Achse** —
viel einfacher zu filtern.
        """)
        st.code("""
import cv2

img_bgr  = cv2.imread('image.jpg')
img_hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
img_lab  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Alle roten Pixel finden (HSV)
lower_red = (0, 100, 100)
upper_red = (10, 255, 255)
mask = cv2.inRange(img_hsv, lower_red, upper_red)
        """, language="python")
        info_box("OpenCV liest Bilder als **BGR**, nicht RGB. Klassischer Bug-Quell.", kind="warn")

    with tabs[2]:
        section_header("Sampling — wie aus der Welt ein Bild wird")
        st.markdown(r"""
Ein Kamerasensor sampelt das kontinuierliche Lichtsignal an einem diskreten Raster.
Die **Sampling-Rate** bestimmt, welche Detail-Frequenzen du noch auflösen kannst.

#### Nyquist-Shannon Theorem
Um ein Signal mit Frequenzen bis $f_{\max}$ verlustfrei zu rekonstruieren,
musst du mit mindestens $2 f_{\max}$ samplen.

In Bildern heißt das: feine Muster, die feiner sind als 2 Pixel,
gehen **nicht nur verloren** — sie tauchen als **Aliasing** wieder auf (Moiré-Muster).

#### Anti-Aliasing
Bevor du runtersampelst, **glätte** zuerst (typisch mit Gauß-Filter), um hohe Frequenzen zu entfernen.
        """)

        # Mini Aliasing Demo
        st.markdown("**Demo:** Gleicher Kreis, verschiedene Auflösungen.")
        cols = st.columns(4)
        for i, size in enumerate([8, 16, 32, 128]):
            xs, ys = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
            r = np.sqrt(xs**2 + ys**2)
            img = (np.sin(20 * r) > 0).astype(float)
            cols[i].markdown(f"**{size}×{size}**")
            fig = go.Figure(go.Heatmap(z=img, colorscale="gray", showscale=False))
            fig.update_layout(height=180, margin=dict(l=0, r=0, t=0, b=0),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              xaxis=dict(visible=False, scaleanchor="y"),
                              yaxis=dict(visible=False, autorange="reversed"))
            cols[i].plotly_chart(fig, use_container_width=True)
        info_box("Bei niedriger Auflösung siehst du Aliasing-Artefakte — die Ringe werden falsch dargestellt.", kind="info")

    with tabs[3]:
        section_header("Quantisierung — kontinuierliche Helligkeit zu diskreten Werten")
        st.markdown(r"""
Ein analoger Helligkeitswert wird auf einen von $2^b$ diskreten Werten gerundet, wo $b$ die Bittiefe ist.

| Bittiefe | Werte | Verwendung |
|---|---|---|
| 1 bit  | 2     | Strichzeichnungen, Faxe |
| 8 bit  | 256   | Standard JPEG/PNG |
| 12 bit | 4096  | RAW-Fotografie |
| 16 bit | 65536 | Medizin, Wissenschaft |
| Float (32 bit) | ~∞ | HDR, Deep Learning Inputs |

#### Quantisierungsrauschen
Je weniger Bits, desto mehr Rauschen entsteht durch Rundung. Sichtbar als **Posterisation** —
glatte Verläufe werden zu Stufen.
        """)

    with tabs[4]:
        section_header("Bildformate — was, wann, warum")
        st.markdown("""
| Format | Komprimierung | Verlust? | Wann nutzen? |
|---|---|---|---|
| **PNG** | Lossless (deflate) | Nein | Logos, Screenshots, Grafiken mit harten Kanten |
| **JPEG** | Lossy (DCT-basiert) | Ja | Fotos — wo Kompressionsartefakte ok sind |
| **WebP** | Lossy/Lossless | Beides | Web — bessere Kompression als JPEG |
| **TIFF** | Lossless | Nein | Wissenschaft, Druck, Mehrere Layer |
| **HEIF/HEIC** | Lossy | Ja | Apple-Fotos — sehr effizient |
| **RAW** | Keine | Nein | Profi-Fotografie — alle Sensor-Daten |
| **AVIF** | Lossy/Lossless | Beides | Modern, beste Kompression — noch nicht überall |

#### JPEG verstehen
JPEG zerlegt das Bild in 8×8-Blöcke, transformiert per DCT in den Frequenzraum,
und wirft hohe Frequenzen weg. Daher die typischen Block-Artefakte bei starker Kompression.
""")
        info_box(
            "Für Deep Learning: niemals JPEG-Bilder mehrfach speichern — Generationsverlust. "
            "Arbeite mit PNG oder Float-Tensoren, solange möglich.",
            kind="tip",
        )
