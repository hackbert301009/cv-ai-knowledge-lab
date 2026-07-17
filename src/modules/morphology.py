"""Morphologische Operationen."""
import streamlit as st
import numpy as np
import cv2
from src.components import (
    hero, section_header, divider, info_box,
    render_learning_block, render_quiz_checkpoint,
)


def render():
    hero(
        eyebrow="Bildverarbeitung · Morphologie",
        title="Morphologie",
        sub="Erosion, Dilatation, Opening, Closing — Form-Operationen, mit denen man "
            "binäre Bilder zurechtschnipselt. Klein, mächtig, überall im Einsatz."
    )

    tabs = st.tabs(["📚 Grundlagen", "🔧 Operationen", "🧪 Live-Demo", "🎯 Anwendungen"])

    with tabs[0]:
        section_header("Strukturierendes Element")
        st.markdown(r"""
Morphologische Operationen brauchen ein **Structuring Element (SE)** — eine kleine Maske,
oft 3×3 oder 5×5, mit Form Quadrat, Kreis oder Kreuz.

Sie arbeiten typischerweise auf **Binärbildern** (0/255), funktionieren aber auch auf Graustufen.

#### Die Idee
Das SE wird über das Bild geschoben, und für jeden Pixel wird gefragt:
- Bei **Erosion**: Liegt das SE komplett im Vordergrund?
- Bei **Dilatation**: Berührt das SE den Vordergrund irgendwo?
        """)

    with tabs[1]:
        section_header("Die vier Grundoperationen")
        st.markdown(r"""
| Operation | Effekt | Wofür? |
|---|---|---|
| **Erosion** ($A \ominus B$) | Schrumpft Vordergrundbereiche | Kleine Störungen entfernen |
| **Dilatation** ($A \oplus B$) | Wächst Vordergrund | Lücken schließen |
| **Opening** ($A \circ B = (A \ominus B) \oplus B$) | Erst Erosion, dann Dilatation | Kleine Vordergrund-Objekte entfernen, große bleiben |
| **Closing** ($A \bullet B = (A \oplus B) \ominus B$) | Erst Dilatation, dann Erosion | Kleine Hintergrund-Löcher schließen |

#### Weitere wichtige
- **Morphologischer Gradient**: $(A \oplus B) - (A \ominus B)$ — Konturen
- **Top-Hat**: $A - (A \circ B)$ — helle Details auf dunklem Hintergrund
- **Black-Hat**: $(A \bullet B) - A$ — dunkle Details auf hellem Hintergrund
        """)

    with tabs[2]:
        section_header("Live-Demo")
        uploaded = st.file_uploader("Binärbild hochladen (oder Demo nutzen)", type=["png", "jpg", "jpeg"], key="morph_upload")

        binary = None
        if uploaded:
            file_bytes = np.frombuffer(uploaded.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            if img is None:
                st.warning("Bild konnte nicht dekodiert werden — verwende Demo-Bild.")
            else:
                _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        if binary is None:
            binary = np.zeros((200, 300), dtype=np.uint8)
            cv2.rectangle(binary, (50, 50), (200, 150), 255, -1)
            cv2.rectangle(binary, (210, 80), (260, 130), 255, -1)
            # Rauschen
            np.random.seed(42)
            noise = np.random.rand(200, 300) > 0.97
            binary[noise] = 255
            # kleines Loch
            cv2.circle(binary, (120, 100), 8, 0, -1)

        c1, c2, c3 = st.columns(3)
        op_name = c1.selectbox("Operation",
                               ["Erosion", "Dilatation", "Opening", "Closing", "Gradient", "Top-Hat", "Black-Hat"])
        ksize = c2.slider("Kernel-Größe", 1, 15, 3, step=2)
        se_shape = c3.selectbox("SE-Form", ["Rechteck", "Ellipse", "Kreuz"])
        iterations = st.slider("Iterationen", 1, 5, 1)

        shape_map = {"Rechteck": cv2.MORPH_RECT, "Ellipse": cv2.MORPH_ELLIPSE, "Kreuz": cv2.MORPH_CROSS}
        kernel = cv2.getStructuringElement(shape_map[se_shape], (ksize, ksize))
        op_map = {
            "Erosion":   lambda: cv2.erode(binary, kernel, iterations=iterations),
            "Dilatation":lambda: cv2.dilate(binary, kernel, iterations=iterations),
            "Opening":   lambda: cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel),
            "Closing":   lambda: cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel),
            "Gradient":  lambda: cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel),
            "Top-Hat":   lambda: cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, kernel),
            "Black-Hat": lambda: cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel),
        }
        result = op_map[op_name]()

        cc1, cc2 = st.columns(2)
        cc1.markdown("**Original**");  cc1.image(binary, clamp=True, use_container_width=True)
        cc2.markdown(f"**{op_name}**"); cc2.image(result, clamp=True, use_container_width=True)

    with tabs[3]:
        section_header("Wo wird das benutzt?")
        st.markdown("""
- **Texterkennung**: Buchstaben isolieren, kleine Punkte entfernen
- **Medizinische Bildgebung**: Zellen segmentieren
- **Industrielle Inspektion**: Fehler auf Bauteilen finden
- **Vorverarbeitung für Konturanalyse**: Glatte Konturen brauchen oft Closing davor
- **Skelettierung**: Form auf 1-Pixel-breites Skelett reduzieren
""")

    divider()
    render_learning_block(
        key_prefix="morphology",
        mcq_question="Du willst kleine weiße Störpixel (Salt-Noise) im Vordergrund entfernen, ohne große Objekte zu verkleinern. Welche Operation?",
        mcq_options=["Opening", "Closing", "Dilatation", "Top-Hat"],
        mcq_correct_option="Opening",
        mcq_success_message="Richtig — Opening (Erosion, dann Dilatation) entfernt kleine Objekte, große bleiben.",
        open_question="Warum ist Closing = Dilatation gefolgt von Erosion gut geeignet, kleine Löcher zu schließen?",
        cheat_sheet=[
            "Erosion schrumpft, Dilatation wächst den Vordergrund.",
            "Opening = Erosion → Dilatation: entfernt kleine Vordergrund-Störungen.",
            "Closing = Dilatation → Erosion: schließt kleine Hintergrund-Löcher.",
            "Gradient = Dilatation − Erosion: Kontur/Umriss.",
        ],
        key_takeaways=[
            "Reihenfolge der Grundoperationen bestimmt den Effekt (Opening vs. Closing).",
            "SE-Form und -Größe steuern, welche Strukturen erhalten bleiben.",
        ],
        common_errors=[
            "Opening und Closing verwechseln.",
            "Zu großes SE zerstört gewünschte Strukturen.",
            "Morphologie auf nicht-binärisiertem Bild ohne klaren Effekt anwenden.",
        ],
    )
    render_quiz_checkpoint(
        key_prefix="morphology",
        module_id="morphology",
        question="Was bewirkt Erosion mit einem größeren strukturierenden Element?",
        options=[
            "Der Vordergrund schrumpft stärker",
            "Der Vordergrund wächst stärker",
            "Löcher im Hintergrund werden geschlossen",
            "Es passiert nichts",
        ],
        correct_option="Der Vordergrund schrumpft stärker",
        checklist=[
            "Ich kann Erosion, Dilatation, Opening und Closing unterscheiden.",
            "Ich weiß, wann ich Opening vs. Closing einsetze.",
        ],
    )
