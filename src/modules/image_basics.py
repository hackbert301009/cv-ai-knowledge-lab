"""Bildgrundlagen — was ist ein Bild eigentlich?"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from src.components import (
    hero, section_header, divider, info_box,
    video_embed, lab_header, key_concept, step_list, render_learning_block,
)


def render():
    hero(
        eyebrow="Bildverarbeitung · Modul 5",
        title="Bildgrundlagen &amp; Pixel",
        sub="Bevor wir Bilder verarbeiten können, müssen wir verstehen, was sie sind. "
            "Pixel, Farbräume, Sampling, Quantisierung — die Anatomie digitaler Bilder."
    )

    tabs = st.tabs([
        "📸 Was ist ein Bild?",
        "🎨 Farbräume",
        "🧪 Pixel-Lab",
        "📐 Sampling & Nyquist",
        "🔢 Quantisierung",
        "💾 Formate & Kompression",
        "🎬 Lernvideos",
        "🧭 Lernpfad & Übungen",
    ])

    # ------------------------------------------------------------------ #
    with tabs[0]:
        section_header("Ein Bild = Funktion auf Pixeln",
                       "Das mathematische Fundament, auf dem alles aufbaut.")
        st.markdown(r"""
Ein digitales Bild ist eine **diskrete Funktion**:

$$I: \{1, \dots, H\} \times \{1, \dots, W\} \to \{0, 1, \dots, 255\}^C$$

Sie nimmt eine Pixelposition $(y, x)$ und gibt einen **Intensitätswert** zurück:

| Typ | Kanäle $C$ | Beschreibung |
|---|---|---|
| **Graustufen** | 1 | Ein Helligkeitswert pro Pixel |
| **RGB** | 3 | Rot, Grün, Blau — additives Farbmodell |
| **RGBA** | 4 | RGB + Alpha (Transparenz 0–255) |
| **Multispektral** | N | Satellitenbilder, Medizin (z.B. 12 Bänder) |
| **HDR Float** | 3 | Werte außerhalb [0,1] — für Hochdynamikbilder |

#### Pixel = Picture Element
Pixel sind **diskret**. Die reale Welt ist **kontinuierlich**. Ein Kamerasensor **sampelt** (tastet ab)
die Welt an einem Raster und **quantisiert** die Helligkeit zu diskreten Werten.

#### Wie ein Kamerasensor funktioniert
1. Licht fällt auf einen **CMOS/CCD-Sensor**
2. Jeder Sensor-Pixel (Photodiode) akkumuliert Elektronen proportional zur Lichtintensität
3. Elektrische Ladung wird in einen digitalen Wert umgewandelt (**A/D-Konverter**)
4. Ein **Bayer-Filter** (RGGB-Mosaik) stellt sicher, dass je Pixel nur ein Farbkanal gemessen wird
5. **Demosaicing** interpoliert die fehlenden Kanäle

#### Auflösung vs. Bittiefe
| Eigenschaft | Was es bestimmt | Typische Werte |
|---|---|---|
| **Auflösung** (H×W) | Räumliche Detail-Feinheit | 4K: 3840×2160 |
| **Bittiefe** (b) | Helligkeits-Abstufungen ($2^b$) | 8 bit = 256 Stufen |
| **Kanalzahl** (C) | Spektrale Information | 1 (gray), 3 (RGB) |

> **Merke:** Eine 4K-RGB-Aufnahme mit 8 bit benötigt $3840 \times 2160 \times 3 = 24{,}9\,\text{MB}$ unkomprimiert.
        """)

        key_concept("🎯", "Spatial Resolution",
                    "Wie viele Pixel das Bild hat — bestimmt feinste Details. Mehr ≠ immer besser, weil Speicher und Rechenzeit wachsen.")
        key_concept("🌈", "Bit Depth",
                    "Wie viele Helligkeitsstufen pro Pixel. 8 bit = 256 Stufen. 16 bit = 65.536. Float = ∞ (für Gradientenrechnung essenziell).")
        key_concept("🔬", "Sensor Noise",
                    "Jede Aufnahme enthält Rauschen (Shot noise, Read noise, Dark current). Deep Learning hat gelernt, dies zu kompensieren.")

    # ------------------------------------------------------------------ #
    with tabs[1]:
        section_header("Farbräume — RGB ist nicht alles",
                       "Verschiedene Farbraummodelle für verschiedene Aufgaben.")
        st.markdown(r"""
#### Das RGB-Modell (additiv)
RGB ist das Standardmodell für **Displays** und **Kameras**. Drei Primärfarben werden addiert.
Weißes Licht = R+G+B bei voller Intensität.

#### Probleme mit RGB für Computer Vision
- "Alle roten Pixel finden" → R, G, B sind alle involviert — schwer zu filtern
- Helligkeit und Farbe sind **vermischt** — eine Änderung der Beleuchtung verändert alle drei Kanäle
- **Nicht wahrnehmungslinear** — der Unterschied (0,0,0)→(1,1,1) wirkt größer als (200,200,200)→(201,201,201)
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
| Farbraum | Achsen | Ideal für |
|---|---|---|
| **RGB** | Rot, Grün, Blau | Displays, Kameras, Standard |
| **HSV** | Hue, Saturation, Value | Farbbasierte Segmentierung |
| **HSL** | Hue, Saturation, Lightness | Grafik/UI-Design |
| **LAB** | L*, a*, b* | Wahrnehmungsbasierte Distanzen |
| **YCbCr** | Luma + Chroma-Differenzen | JPEG, Video-Kompression |
| **XYZ** | Geräteunabhängig (CIE 1931) | Farb-Kalibration, ICC-Profile |
| **Graustufen** | Helligkeit | Wenn Farbe irrelevant ist |
""")
        with col2:
            # HSV-Farbkreis Demo
            theta = np.linspace(0, 2*np.pi, 360)
            r = np.ones(360)
            colors_hsv = [f"hsl({h},100%,50%)" for h in range(360)]
            fig = go.Figure(go.Barpolar(
                r=r, theta=np.degrees(theta),
                marker_color=colors_hsv,
                marker_line_width=0,
                width=1,
            ))
            fig.update_layout(
                title="HSV-Farbkreis (Hue 0–360°)",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=False),
                    angularaxis=dict(showticklabels=True, tickfont_size=10),
                ),
                height=320,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.code("""
import cv2
import numpy as np

img_bgr  = cv2.imread('image.jpg')         # OpenCV liest BGR, nicht RGB!
img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
img_lab  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Alle roten Pixel finden (HSV macht das trivial)
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
mask = cv2.inRange(img_hsv, lower_red, upper_red)

# Rot auf der anderen Seite des Hue-Kreises
mask2 = cv2.inRange(img_hsv, np.array([160, 100, 100]),
                              np.array([180, 255, 255]))
red_mask = cv2.bitwise_or(mask, mask2)
        """, language="python")
        info_box("OpenCV liest Bilder als **BGR**, nicht RGB. Das ist ein klassischer Bug-Quell. "
                 "Immer `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` nach `cv2.imread()` wenn du matplotlib oder Streamlit nutzt.", kind="warn")

        st.markdown("#### LAB — der wahrnehmungslineare Farbraum")
        st.markdown(r"""
LAB (auch CIELAB) ist so konstruiert, dass **gleiche Abstände = gleiche wahrgenommene Unterschiede**.
- **L*** (0–100): Helligkeit (unabhängig von Farbe!)
- **a*** (−128 bis +127): Grün (−) bis Rot (+)
- **b*** (−128 bis +127): Blau (−) bis Gelb (+)

Warum das wichtig ist: Wenn du Bilder auf **Farbähnlichkeit** vergleichen willst,
ist der euklidische Abstand im LAB-Raum viel sinnvoller als im RGB-Raum.

$$\Delta E = \sqrt{(L_1^*-L_2^*)^2 + (a_1^*-a_2^*)^2 + (b_1^*-b_2^*)^2}$$
        """)

    # ------------------------------------------------------------------ #
    with tabs[2]:
        lab_header("Pixel-Explorer",
                   "Erzeuge ein Bild, untersuche einzelne Pixel-Werte und konvertiere Farbräume in Echtzeit.")

        # Synthetisches Testbild generieren
        cols_ctrl = st.columns(3)
        pattern = cols_ctrl[0].selectbox("Muster", ["Farbverlauf", "Farbkreise", "Schachbrett", "Rauschen"])
        size = cols_ctrl[1].slider("Bildgröße", 50, 300, 128, step=10)
        noise_level = cols_ctrl[2].slider("Rausch-Intensität", 0, 60, 0)

        if pattern == "Farbverlauf":
            y, x = np.meshgrid(np.linspace(0, 255, size), np.linspace(0, 255, size), indexing="ij")
            r = x.astype(np.uint8)
            g = y.astype(np.uint8)
            b = (255 - x / 2 - y / 2).clip(0, 255).astype(np.uint8)
            img = np.stack([r, g, b], axis=-1)
        elif pattern == "Farbkreise":
            img = np.zeros((size, size, 3), dtype=np.uint8)
            cx, cy = size // 2, size // 2
            for yi in range(size):
                for xi in range(size):
                    dist = np.sqrt((xi - cx)**2 + (yi - cy)**2)
                    angle = np.degrees(np.arctan2(yi - cy, xi - cx)) % 360
                    if dist < size * 0.45:
                        h = angle / 360.0
                        import colorsys
                        r, g, b = colorsys.hsv_to_rgb(h, min(1, dist / (size * 0.45)), 1)
                        img[yi, xi] = [int(r * 255), int(g * 255), int(b * 255)]
        elif pattern == "Schachbrett":
            sq = max(1, size // 8)
            board = np.indices((size, size)).sum(axis=0) // sq % 2
            img = np.zeros((size, size, 3), dtype=np.uint8)
            img[board == 0] = [240, 220, 180]
            img[board == 1] = [80, 60, 120]
        else:
            img = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)

        if noise_level > 0:
            noise = np.random.randint(-noise_level, noise_level + 1, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        import cv2
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown("**RGB**");       c1.image(img, use_container_width=True)
        c2.markdown("**Graustufen**");c2.image(img_gray, use_container_width=True, clamp=True)
        c3.markdown("**HSV (H-Kanal)**"); c3.image(img_hsv[:, :, 0] * 2, use_container_width=True, clamp=True)
        c4.markdown("**LAB (L-Kanal)**"); c4.image(img_lab[:, :, 0], use_container_width=True, clamp=True)

        st.markdown("#### Pixel-Histogramme")
        fig_hist = go.Figure()
        for ch, col_name, color in zip(range(3), ["Rot", "Grün", "Blau"], ["#EF4444", "#22C55E", "#3B82F6"]):
            vals, bins = np.histogram(img[:, :, ch].flatten(), bins=64, range=(0, 255))
            fig_hist.add_trace(go.Bar(x=bins[:-1], y=vals, name=col_name,
                                      marker_color=color, opacity=0.7))
        fig_hist.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", barmode="overlay", height=250,
            margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("#### Einzelner Pixel-Wert")
        pc1, pc2 = st.columns(2)
        px_y = pc1.slider("Pixel-Zeile (y)", 0, size - 1, size // 2)
        px_x = pc2.slider("Pixel-Spalte (x)", 0, size - 1, size // 2)
        pix = img[px_y, px_x]
        pix_hsv = img_hsv[px_y, px_x]
        pix_lab = img_lab[px_y, px_x]
        hex_col = "#{:02x}{:02x}{:02x}".format(*pix)
        st.markdown(
            f"""<div style="display:flex;gap:1.5rem;align-items:center;padding:1rem;
                background:rgba(255,255,255,0.04);border-radius:10px;border:1px solid rgba(255,255,255,0.08);">
                <div style="width:60px;height:60px;border-radius:8px;background:{hex_col};
                            box-shadow:0 0 16px {hex_col}88;flex-shrink:0;"></div>
                <div>
                  <div style="font-size:1.1rem;font-weight:700;">{hex_col.upper()}</div>
                  <div style="color:#9CA3AF;font-size:0.85rem;">
                    RGB: ({pix[0]}, {pix[1]}, {pix[2]}) &nbsp;|&nbsp;
                    HSV: ({pix_hsv[0]}°, {pix_hsv[1]/255*100:.0f}%, {pix_hsv[2]/255*100:.0f}%) &nbsp;|&nbsp;
                    LAB: L={pix_lab[0]/2.55:.0f}, a={pix_lab[1]-128}, b={pix_lab[2]-128}
                  </div>
                  <div style="color:#9CA3AF;font-size:0.85rem;">Position: ({px_x}, {px_y}) im {size}×{size} Bild</div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------ #
    with tabs[3]:
        section_header("Sampling & Nyquist-Theorem",
                       "Warum Aliasing entsteht und wie Anti-Aliasing es verhindert.")
        st.markdown(r"""
Ein Kamerasensor **sampelt** das kontinuierliche Lichtsignal an einem diskreten Raster.
Die **Sampling-Rate** bestimmt, welche Detail-Frequenzen du noch auflösen kannst.

#### Das Nyquist-Shannon-Abtasttheorem
> Um ein Signal mit Frequenzen bis $f_{\max}$ verlustfrei zu rekonstruieren,
> muss man mit mindestens $f_s \geq 2 f_{\max}$ samplen.

In Bildern bedeutet das: feine Muster, die feiner sind als **2 Pixel pro Periode**,
gehen nicht nur verloren — sie tauchen als **Aliasing** wieder auf (Moiré-Muster).

#### Warum Aliasing entsteht
Ein Signal mit Frequenz $f > f_s/2$ wird nach dem Sampling als Frequenz $f_s - f$ interpretiert.
Das Spektrum wird quasi "umgeklappt" — hence **alias**.

#### Anti-Aliasing beim Downsampling
1. **Glätten** mit Gauß-Filter ($\sigma$ proportional zum Downsample-Faktor)
2. **Dann** runtersampeln

Wenn du zuerst runtersampelst, sind die hohen Frequenzen schon als Aliasing eingebacken.
        """)

        lab_header("Aliasing Demo", "Gleicher Kreis bei verschiedenen Auflösungen.")
        alias_cols = st.columns(5)
        for i, size_a in enumerate([8, 16, 32, 64, 256]):
            xs, ys = np.meshgrid(np.linspace(-1, 1, size_a), np.linspace(-1, 1, size_a))
            r_grid = np.sqrt(xs**2 + ys**2)
            img_a = (np.sin(18 * r_grid) > 0).astype(float)
            alias_cols[i].markdown(f"**{size_a}×{size_a}**")
            fig_a = go.Figure(go.Heatmap(z=img_a, colorscale="gray", showscale=False))
            fig_a.update_layout(
                height=150, margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(visible=False, scaleanchor="y"),
                yaxis=dict(visible=False, autorange="reversed"),
            )
            alias_cols[i].plotly_chart(fig_a, use_container_width=True)
        info_box("Bei 8×8 siehst du Aliasing-Artefakte — Ringe werden falsch dargestellt. "
                 "Bei 256×256 ist das Muster klar erkennbar.", kind="info")

        st.markdown("#### Anti-Aliasing in der Praxis")
        st.code("""
import cv2
import numpy as np

# FALSCH: direkt runtersampeln (erzeugt Aliasing)
img_wrong = img[::4, ::4]  # Nimmt nur jeden 4. Pixel

# RICHTIG: zuerst glätten, dann sampeln
sigma = 4 * 0.5  # sigma proportional zum Downsample-Faktor
img_blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
img_correct = cv2.resize(img_blur, (img.shape[1]//4, img.shape[0]//4))

# Oder direkt mit cv2.resize (benutzt intern Anti-Aliasing)
img_resize = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4),
                        interpolation=cv2.INTER_AREA)  # INTER_AREA = beste Wahl beim Downsampling
        """, language="python")

    # ------------------------------------------------------------------ #
    with tabs[4]:
        section_header("Quantisierung", "Von analogen Helligkeitswerten zu diskreten Zahlen.")
        st.markdown(r"""
Ein analoger Helligkeitswert $v \in [0, 1]$ wird auf einen von $2^b$ diskreten Werten gerundet,
wobei $b$ die **Bittiefe** ist.

$$q(v) = \text{round}(v \cdot (2^b - 1))$$

#### Bittiefe im Vergleich
| Bittiefe | Werte | Größe (3-Kanal, 1 MP) | Verwendung |
|---|---|---|---|
| **1 bit**   | 2          | 0.4 MB | Binärbilder, Strichzeichnungen |
| **8 bit**   | 256        | 3.0 MB | Standard JPEG/PNG, Kameras, Deep Learning |
| **12 bit**  | 4.096      | 4.5 MB | RAW-Fotografie |
| **14 bit**  | 16.384     | 5.3 MB | Hochwertige Kameras |
| **16 bit**  | 65.536     | 6.0 MB | Medizinische Bildgebung, Astronomie |
| **Float32** | ~16 Mio.  | 12.0 MB | HDR, Deep-Learning-Zwischenwerte |

#### Quantisierungsrauschen
Je weniger Bits, desto mehr Rauschen durch Rundung.
Sichtbar als **Posterisation** — glatte Verläufe werden zu Stufen.

Quantisierungsrauschen hat Amplitude $q = \frac{1}{2^b}$, also Rauschen $\approx \frac{q}{\sqrt{12}}$.
        """)

        lab_header("Quantisierung Live", "Wähle die Bittiefe und sieh, wie das Bild aussieht.")
        bits = st.slider("Bittiefe (Bits)", 1, 8, 8)
        levels = 2 ** bits

        # Einfaches Gradientenbild
        grad = np.tile(np.linspace(0, 255, 300, dtype=np.uint8), (100, 1))
        grad_q = (grad // (256 // levels) * (256 // levels)).astype(np.uint8)

        q1, q2 = st.columns(2)
        q1.markdown("**Original (8 bit)**")
        q1.image(grad, use_container_width=True, clamp=True)
        q2.markdown(f"**Quantisiert ({bits} bit = {levels} Stufen)**")
        q2.image(grad_q, use_container_width=True, clamp=True)
        info_box(
            f"Bei {bits} bit siehst du {levels} Helligkeitsstufen. "
            "Unterhalb von 4 bit ist Posterisation deutlich sichtbar.",
            kind="info",
        )

    # ------------------------------------------------------------------ #
    with tabs[5]:
        section_header("Bildformate & Kompression", "Was, wann, warum — und wie JPEG wirklich funktioniert.")
        st.markdown("""
| Format | Komprimierung | Verlust? | Alpha? | Wann nutzen? |
|---|---|---|---|---|
| **PNG** | Lossless (DEFLATE) | Nein | ✅ | Logos, Screenshots, CV-Datensätze |
| **JPEG** | Lossy (DCT-basiert) | Ja | ❌ | Fotos — wenn Dateigröße wichtig |
| **WebP** | Lossy/Lossless | Beides | ✅ | Web — 25–34% kleiner als JPEG |
| **AVIF** | Lossy/Lossless | Beides | ✅ | Modern, beste Kompression verfügbar |
| **TIFF** | Lossless | Nein | ✅ | Wissenschaft, Druck, Multi-Layer |
| **HEIF/HEIC** | Lossy | Ja | ✅ | Apple-Kameras — sehr effizient |
| **RAW** | Keine | Nein | ❌ | Profi-Fotografie — rohe Sensor-Daten |
| **HDF5 / .npy** | Keine | Nein | — | Deep Learning — Tensoren direkt speichern |
""")

        st.markdown("#### Wie JPEG wirklich funktioniert")
        step_list([
            ("RGB → YCbCr konvertieren",
             "Helligkeit (Y) und Farbe (Cb, Cr) trennen. Farbkanäle werden 2× runtersampelt (Chroma Subsampling), weil das Auge Helligkeitsdetails besser sieht."),
            ("Bild in 8×8 Blöcke aufteilen",
             "Jeder Block wird unabhängig verarbeitet. Daher die typischen 8×8-Block-Artefakte bei starker Kompression."),
            ("Diskrete Kosinus-Transformation (DCT)",
             "Jeden Block in Frequenzkomponenten transformieren. Niedrige Frequenzen = grobe Struktur, hohe = feine Details."),
            ("Quantisierung der Frequenzkoeffizienten",
             "Hohe Frequenzen werden stärker gerundet (oder ganz auf 0 gesetzt). Quality-Faktor steuert, wie aggressiv."),
            ("Entropie-Kodierung (Huffman/Arithmetic)",
             "Die quantisierten Koeffizienten werden verlustfrei komprimiert. Viele Nullen → sehr kompakt."),
        ])

        info_box(
            "**Deep-Learning-Tipp:** Niemals JPEG-Bilder mehrfach neu abspeichern — jedes Speichern erhöht "
            "die Artefakte (Generationsverlust). Arbeite intern mit PNG oder Float-Tensoren.",
            kind="tip",
        )

        st.code("""
import cv2
from PIL import Image
import numpy as np

# PNG lesen — verlustfrei
img = cv2.imread('image.png', cv2.IMREAD_UNCHANGED)  # UNCHANGED = Alphakanal behalten

# JPEG mit verschiedenen Qualitätsstufen speichern
for q in [10, 50, 90, 100]:
    cv2.imwrite(f'output_q{q}.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])

# JPEG in Memory komprimieren und dekodieren (für Augmentation)
encode_param = [cv2.IMWRITE_JPEG_QUALITY, 70]
_, encoded = cv2.imencode('.jpg', img, encode_param)
decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

# Numpy Array direkt speichern (für Datasets)
np.save('tensor.npy', img.astype(np.float32) / 255.0)
tensor = np.load('tensor.npy')
        """, language="python")

    # ------------------------------------------------------------------ #
    with tabs[6]:
        section_header("Lernvideos", "Die besten kostenlosen Erklärvideos zum Thema.")

        st.markdown("#### Computerphile — Image Basics (University of Nottingham)")
        video_embed("LZNva7Kf9IM",
                    "How Images Work — Computerphile",
                    "Mike Pound (Univ. Nottingham) erklärt, wie digitale Bilder intern aufgebaut sind.")

        divider()

        st.markdown("#### Bildverarbeitung: Gauß-Filter und Smoothing")
        video_embed("C_zFhWdM4ic",
                    "Gaussian Blur — Computerphile",
                    "Wie der Gauß-Filter mathematisch funktioniert und warum er so gut ist.")

        divider()

        info_box(
            "**Tipp:** Schau die Videos in 1,5× Geschwindigkeit und pausiere bei den Formeln. "
            "Dann öffne Jupyter und probiere die Beispiele selbst nach — aktives Lernen ist 4× effektiver.",
            kind="tip",
        )

    # ------------------------------------------------------------------ #
    with tabs[7]:
        render_learning_block(
            key_prefix="image_basics",
            section_title="Lernpfad für Bildgrundlagen",
            progression=[
                ("🟢", "Guided Lab", "RGB/HSV/LAB auf einem Bild vergleichen und Unterschiede begründen.", "Beginner", "green"),
                ("🟠", "Challenge Lab", "Robuste Farbbereichs-Segmentierung bei wechselnder Beleuchtung bauen.", "Intermediate", "amber"),
                ("🔴", "Debug Lab", "BGR/RGB-Verwechslung und Aliasing-Artefakte gezielt finden und beheben.", "Advanced", "pink"),
                ("🏁", "Mini-Projekt", "Ein kleiner Pixel-Inspector mit Histogrammen und Farbraum-Ansicht.", "Abschluss", "blue"),
            ],
            mcq_question="Welcher Farbraum ist für farbbasierte Segmentierung oft am praktischsten?",
            mcq_options=["RGB", "HSV", "RAW Bayer", "Graustufen"],
            mcq_correct_option="HSV",
            mcq_success_message="Richtig. HSV trennt Farbton und Helligkeit für viele Tasks besser.",
            mcq_retry_message="Noch nicht korrekt. Prüfe den Farbraum-Vergleich.",
            open_question="Offene Frage: Wann würdest du 16-bit statt 8-bit Bilddaten verwenden?",
            code_task="""# Code-Aufgabe: BGR -> RGB korrekt umwandeln
import cv2
img_bgr = cv2.imread("input.jpg")
# TODO: richtige Umwandlung ergänzen und anschließend HSV berechnen
""",
            community_rows=[
                {"Format": "Diskussion", "Thema": "Welche Fehler entstehen durch falschen Farbraum?", "Output": "Kurzbeispiel"},
                {"Format": "Peer-Feedback", "Thema": "Sind Histogramm und Interpretation stimmig?", "Output": "2 Pluspunkte + 1 Verbesserung"},
                {"Format": "Challenge", "Thema": "Bestes Segmentierungsresultat bei wechselndem Licht", "Output": "Bildvergleich"},
            ],
            cheat_sheet=[
                "OpenCV liest BGR, viele Libraries erwarten RGB.",
                "Downsampling mit Anti-Aliasing durchführen.",
                "Bit-Tiefe bewusst nach Aufgabe wählen.",
            ],
            key_takeaways=[
                "Pixel, Sampling und Farbraumwahl bestimmen viele spätere Modellgrenzen.",
                "Saubere Bildvorverarbeitung spart später viel Debug-Zeit.",
            ],
            common_errors=[
                "BGR/RGB verwechselt.",
                "Falsche Interpolation beim Resize.",
                "Zu aggressive JPEG-Kompression im Datensatz.",
                "Bit-Tiefe ignoriert.",
                "Histogramme nicht geprüft.",
            ],
        )
