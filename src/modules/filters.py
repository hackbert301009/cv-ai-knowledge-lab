"""Filter & Faltung — die Mutter aller CNNs."""
import streamlit as st
import numpy as np
import cv2
import plotly.graph_objects as go
from src.components import (
    hero, section_header, divider, info_box,
    video_embed, lab_header, key_concept, step_list,
)


def render():
    hero(
        eyebrow="Bildverarbeitung · Modul 6",
        title="Filter &amp; Faltung",
        sub="Die fundamentale Operation: ein kleiner Kernel gleitet über das Bild und berechnet einen gewichteten Durchschnitt. "
            "Aus dieser einen Idee sind Convolutional Neural Networks entstanden."
    )

    tabs = st.tabs([
        "🌫️ Was ist Faltung?",
        "🎛️ Klassische Kernel",
        "🧪 Interaktiv: Filter",
        "✍️ Eigener Kernel",
        "🔗 Brücke zu CNN",
        "🎬 Lernvideos",
    ])

    # ------------------------------------------------------------------ #
    with tabs[0]:
        section_header("Faltung — die zentrale Operation",
                       "Dieselbe Idee, die klassische Bildverarbeitung und CNNs verbindet.")
        st.markdown(r"""
Die **diskrete 2D-Faltung** (in CV häufig: Korrelation) ist:

$$(I \star K)(y, x) = \sum_{i=-k}^{k} \sum_{j=-k}^{k} I(y+i,\, x+j) \cdot K(i,\, j)$$

**In Worten:** Lege den Kernel $K$ auf jede Pixelposition, multipliziere elementweise, summiere.
Das Ergebnis ist der neue Pixelwert an dieser Position.
        """)

        key_concept("🔲", "Kernel (Filter)",
                    "Eine kleine Matrix (z.B. 3×3), die bestimmt, welche Bildeigenschaft herausgehoben wird. "
                    "Nur 9 Parameter — aber riesige Wirkung.")
        key_concept("↔️", "Stride",
                    "Wie weit der Kernel pro Schritt springt. Stride=1: jeder Pixel. "
                    "Stride=2: halbiert die Ausgabe-Auflösung. CNNs nutzen Stride statt Pooling.")
        key_concept("🔲", "Padding",
                    "Was passiert am Bildrand? Zero-Padding (mit Nullen auffüllen) hält die Auflösung konstant. "
                    "Reflect-Padding ist oft besser an Rändern.")

        st.markdown(r"""
#### Mathematische Eigenschaften
| Eigenschaft | Was bedeutet das? | Warum wichtig? |
|---|---|---|
| **Linearität** | Skalierung + Addition bleiben erhalten | Erlaubt Superposition von Effekten |
| **Shift-Invarianz** | Gleicher Kernel überall im Bild | Merkmale gefunden, egal wo |
| **Lokalität** | Nur Nachbarpixel beeinflussen Output | Räumliche Struktur bleibt erhalten |
| **Effizienz** | $O(k^2 \cdot H \cdot W)$ statt $O(N^2)$ | Praktisch realisierbar |

#### Wichtige Begriffe
- **Valid Convolution**: kein Padding → Output kleiner als Input
- **Same Convolution**: Padding so, dass Output = Input (bei Stride=1)
- **Full Convolution**: viel Padding → Output größer als Input
- **Separable Kernel**: $K = k_x \cdot k_y^\top$ → Faktor-$k$ schneller (Gauß ist separierbar!)
        """)

    # ------------------------------------------------------------------ #
    with tabs[1]:
        section_header("Klassische Kernel — was wofür?",
                       "Handentworfene Filter, die Jahrzehnte CV-Forschung definiert haben.")

        st.markdown("""
| Kernel | Effekt | Mathematik | Typischer Einsatz |
|---|---|---|---|
| **Identity** | Bild unverändert | Einheitsmatrix | Test, Baseline |
| **Box-Blur** | Mittelwert (verschwommen) | $\\frac{1}{9}\\mathbf{1}$ | Schnelles Blur, Downsampling-Vorbereitung |
| **Gauß-Blur** | Gewichteter Blur — natürlicher | Gauß-Funktion | Anti-Aliasing, Rauschreduktion |
| **Sharpen** | Kanten verstärken | $5 \\cdot \\delta - \\text{Laplace}$ | Vorverarbeitung für OCR etc. |
| **Sobel-X** | Horizontale Kanten | Diskrete x-Ableitung | Kantendetektion, Gradient |
| **Sobel-Y** | Vertikale Kanten | Diskrete y-Ableitung | Kantendetektion, Gradient |
| **Laplace** | Alle Kanten | Zweite Ableitung | Blob Detection, LoG |
| **Emboss** | 3D-Relief-Effekt | Asymmetrischer Differenzfilter | Kunstfilter, Texturen |
| **Motion-Blur** | Bewegungsunschärfe | Diagonale Linie | Simulation, Bildrekonstruktion |
| **Unsharp Mask** | Schärfen via Differenz | Original − Blur | Foto-Nachbearbeitung |
""")

        # Kernel-Visualisierungen
        kernels_display = {
            "Gauß 3×3": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float) / 16,
            "Sobel-X": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float),
            "Sobel-Y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float),
            "Laplace": np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=float),
        }
        kd_cols = st.columns(4)
        for col, (name, kern) in zip(kd_cols, kernels_display.items()):
            col.markdown(f"**{name}**")
            fig_k = go.Figure(go.Heatmap(
                z=kern,
                colorscale=[[0, "#EF4444"], [0.5, "#1F2937"], [1, "#3B82F6"]],
                showscale=False,
                text=[[f"{v:.2f}" for v in row] for row in kern],
                texttemplate="%{text}",
                textfont=dict(size=11),
            ))
            fig_k.update_layout(
                height=120, margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(visible=False), yaxis=dict(visible=False),
            )
            col.plotly_chart(fig_k, use_container_width=True)

        info_box(
            "**Merkhilfe Gauß:** Die Mitte ist heißer als die Ränder — proportional zu $e^{-r^2/(2\\sigma^2)}$. "
            "Der Gauß-Kernel ist der **einzige** Blur-Kernel, der das Nyquist-Theorem respektiert.",
            kind="info",
        )

    # ------------------------------------------------------------------ #
    with tabs[2]:
        lab_header("Interaktiv: Filter ausprobieren",
                   "Bild hochladen (oder Demo-Bild) und alle Kernel live vergleichen.")

        uploaded = st.file_uploader("Bild hochladen (PNG, JPG)", type=["png", "jpg", "jpeg"], key="filter_upload")

        if uploaded:
            file_bytes = np.frombuffer(uploaded.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.zeros((220, 220, 3), dtype=np.uint8)
            # Synthetisches Demo-Bild
            cv2.circle(img, (110, 110), 70, (240, 100, 200), -1)
            cv2.rectangle(img, (30, 30), (90, 90), (50, 200, 240), -1)
            cv2.line(img, (10, 190), (210, 190), (255, 255, 255), 3)
            cv2.putText(img, "Demo", (80, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 0), 2)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        kernels = {
            "Identity":          np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32),
            "Box-Blur 3×3":      np.ones((3, 3), dtype=np.float32) / 9,
            "Box-Blur 7×7":      np.ones((7, 7), dtype=np.float32) / 49,
            "Gauß-Blur 3×3":     (cv2.getGaussianKernel(3, 0.8) @ cv2.getGaussianKernel(3, 0.8).T),
            "Gauß-Blur 5×5":     (cv2.getGaussianKernel(5, 1.4) @ cv2.getGaussianKernel(5, 1.4).T),
            "Gauß-Blur 9×9":     (cv2.getGaussianKernel(9, 2.0) @ cv2.getGaussianKernel(9, 2.0).T),
            "Sharpen":           np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
            "Unsharp Mask":      np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32) / 1.0,
            "Sobel-X":           np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
            "Sobel-Y":           np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
            "Laplace":           np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32),
            "Laplace 8-conn":    np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32),
            "Emboss":            np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32),
            "Motion-Blur":       np.eye(7, dtype=np.float32) / 7,
        }

        fc1, fc2 = st.columns(2)
        choice = fc1.selectbox("Kernel wählen", list(kernels.keys()), index=4)
        show_abs = fc2.checkbox("Absolutwert anzeigen (für Kanten-Kernel)", value=True)
        kernel = kernels[choice]

        filtered_raw = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        if show_abs:
            filtered = np.clip(np.abs(filtered_raw), 0, 255).astype(np.uint8)
        else:
            filtered = np.clip(filtered_raw, 0, 255).astype(np.uint8)

        c1, c2 = st.columns(2)
        c1.markdown("**Original (Graustufen)**")
        c1.image(gray, clamp=True, use_container_width=True)
        c2.markdown(f"**Nach `{choice}` {'|Absolut|' if show_abs else ''}**")
        c2.image(filtered, clamp=True, use_container_width=True)

        st.markdown(f"**Kernel ({kernel.shape[0]}×{kernel.shape[1]}):**")
        st.code(np.array2string(kernel, precision=3, suppress_small=True), language="text")

        # Histogram der gefilterten Werte
        vals_raw = filtered_raw.flatten()
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=vals_raw, nbinsx=64, name="Pixel-Werte",
                                         marker_color="#7C3AED", opacity=0.8))
        fig_hist.add_vline(x=0, line_color="#EC4899", line_width=1.5)
        fig_hist.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=180,
            margin=dict(l=20, r=10, t=10, b=30),
            xaxis=dict(title="Pixel-Wert nach Filter"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        info_box(
            f"Kernel `{choice}`: Min={vals_raw.min():.1f}, Max={vals_raw.max():.1f}, "
            f"Mean={vals_raw.mean():.1f}, Std={vals_raw.std():.1f}",
            kind="info",
        )

    # ------------------------------------------------------------------ #
    with tabs[3]:
        lab_header("Eigener Kernel Editor",
                   "Definiere deinen eigenen 3×3 Kernel und sieh sofort das Ergebnis.")

        st.markdown("Ändere die Werte des Kernels und sieh, was passiert.")
        default_kernel = [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
        cols_k = st.columns(3)
        user_kernel = np.zeros((3, 3), dtype=np.float32)
        for row in range(3):
            for col in range(3):
                with cols_k[col]:
                    val = st.number_input(
                        f"K[{row},{col}]",
                        value=float(default_kernel[row][col]),
                        step=0.5,
                        format="%.1f",
                        key=f"kernel_{row}_{col}",
                        label_visibility="collapsed" if row > 0 else "visible",
                    )
                    user_kernel[row, col] = val

        norm_kernel = st.checkbox("Kernel normalisieren (Summe auf 1)", value=False)
        if norm_kernel and user_kernel.sum() != 0:
            display_kernel = user_kernel / user_kernel.sum()
        else:
            display_kernel = user_kernel

        st.markdown(f"**Kernel-Summe:** {user_kernel.sum():.2f} | "
                    f"**Kernel-Norm (L2):** {np.linalg.norm(user_kernel):.2f}")

        # Demo-Bild neu erzeugen
        demo_img_custom = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(demo_img_custom, (100, 100), 60, 200, -1)
        cv2.rectangle(demo_img_custom, (20, 20), (80, 80), 150, -1)
        cv2.line(demo_img_custom, (0, 160), (200, 160), 255, 2)

        custom_filtered = cv2.filter2D(demo_img_custom.astype(np.float32), -1, display_kernel)
        custom_vis = np.clip(np.abs(custom_filtered), 0, 255).astype(np.uint8)

        ck1, ck2, ck3 = st.columns(3)
        ck1.markdown("**Eingabe**")
        ck1.image(demo_img_custom, use_container_width=True, clamp=True)

        ck2.markdown("**Kernel (Heatmap)**")
        fig_uk = go.Figure(go.Heatmap(
            z=user_kernel,
            colorscale=[[0, "#EF4444"], [0.5, "#1F2937"], [1, "#3B82F6"]],
            showscale=True,
            colorbar=dict(thickness=8, len=0.8),
            text=[[f"{v:.1f}" for v in row] for row in user_kernel],
            texttemplate="%{text}", textfont=dict(size=13, color="white"),
        ))
        fig_uk.update_layout(
            height=180, margin=dict(l=0, r=40, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )
        ck2.plotly_chart(fig_uk, use_container_width=True)

        ck3.markdown("**Ausgabe |Absolut|**")
        ck3.image(custom_vis, use_container_width=True, clamp=True)

        info_box(
            "Tipp: Probiere `[[1,1,1],[1,1,1],[1,1,1]]` (Box-Blur) und dann `[[0,-1,0],[-1,4,-1],[0,-1,0]]` (Laplace). "
            "Dann versuche `[[1,0,-1],[0,0,0],[-1,0,1]]` — was siehst du?",
            kind="tip",
        )

    # ------------------------------------------------------------------ #
    with tabs[4]:
        section_header("Vom klassischen Filter zum CNN",
                       "Wie CNNs dieselbe Idee nehmen und sie lernen lassen.")
        st.markdown(r"""
Klassische Bildverarbeitung benutzt **handentworfene** Kernel — Sobel, Gauß, Laplace.
Ein **Convolutional Neural Network** macht genau dasselbe, aber mit einem entscheidenden Unterschied:

#### CNNs **lernen** ihre Kernel
Statt einen Sobel-Kernel von Hand einzusetzen, werden die Kernel **zufällig initialisiert**
und über Backpropagation angepasst, um das gegebene Problem zu lösen.

#### Was in den Layern passiert
        """)

        step_list([
            ("Layer 1: Einfache Kanten & Farben",
             "Die ersten Kernel ähneln oft Gabor-Filtern oder Sobel — das Netz entdeckt klassische CV neu."),
            ("Layer 2–4: Muster & Texturen",
             "Kombination von Kanten → Kurven, Gitter, Texturen. Komplexere Strukturen entstehen."),
            ("Layer 5–10: Objektteile",
             "Augen, Räder, Fenster — semantisch bedeutsame Teile werden erkannt."),
            ("Layer 10+: Objekte & Kategorien",
             "Das Netz 'denkt' in Begriffen wie 'Hund' oder 'Auto'. Sehr abstrakte Repräsentationen."),
        ])

        st.code("""
import torch
import torch.nn as nn

# Ein Conv-Layer hat (C_in × k × k + 1) × C_out Parameter
# Beispiel: 1 Input-Kanal (Graustufen), 32 Kernel à 3×3
conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
print(f"Parameter: {conv.weight.numel() + conv.bias.numel()}")  # (1×3×3+1)×32 = 320

# Die Kernel nach Training visualisieren
import matplotlib.pyplot as plt
kernels = conv.weight.detach().cpu().numpy()  # Shape: [32, 1, 3, 3]
fig, axes = plt.subplots(4, 8, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(kernels[i, 0], cmap='RdBu_r')
    ax.axis('off')
plt.suptitle('32 gelernte Kernel (nach Training)')
plt.show()

# Aktivierungen (Feature Maps) visualisieren
with torch.no_grad():
    # Beispielbild durch den Layer schicken
    x = torch.randn(1, 1, 64, 64)
    features = conv(x)  # [1, 32, 64, 64]
    # Jeder der 32 Output-Maps zeigt ein anderes Merkmal
        """, language="python")

        info_box(
            "**Experiment:** Visualisiere die Kernel-Gewichte von VGG-16 Layer 1. "
            "Du wirst Strukturen sehen, die aussehen wie Gabor-Filter, Sobel-Operatoren und Farbdetektoren — "
            "genau die Kernel, die Bildverarbeitungs-Forscher jahrzehntelang von Hand entworfen haben.",
            kind="success",
        )

    # ------------------------------------------------------------------ #
    with tabs[5]:
        section_header("Lernvideos", "Faltung und Filter visuell verstehen.")

        st.markdown("#### Image Kernels — Computerphile")
        video_embed("uihBwtPIBxM",
                    "Finding the Edges (Sobel Operator) — Computerphile",
                    "Mike Pound erklärt Sobel, Prewitt und Kantendetektion. ~10 Minuten.")

        divider()

        st.markdown("#### Gaussian Blur erklärt — Computerphile")
        video_embed("C_zFhWdM4ic",
                    "Gaussian Blur — Computerphile",
                    "Wie der Gauß-Filter funktioniert und warum er die beste Blur-Methode ist.")

        divider()

        st.markdown("#### But what is a convolution? — 3Blue1Brown")
        video_embed("KuXjwB4LzSA",
                    "But what is a convolution? — 3Blue1Brown",
                    "3Blue1Brown erklärt Faltung von den mathematischen Grundlagen bis zur Anwendung. ~23 Minuten.")

        info_box(
            "Nachdem du diese Videos geschaut hast, öffne das 'Interaktiv'-Tab und probiere alle Kernel durch. "
            "Aktives Experimentieren ist 5× effektiver als nur zuschauen.",
            kind="tip",
        )
