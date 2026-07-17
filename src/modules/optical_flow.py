"""Optical Flow & Motion — Lucas-Kanade, Farnebäck, RAFT."""
import cv2
import numpy as np
import streamlit as st

from src.components import (
    hero, section_header, divider, info_box, lab_header, key_concept, video_embed, video_search,
    render_learning_block, render_quiz_checkpoint,
)


def _scene(dx: int, dy: int, size: int = 240) -> np.ndarray:
    """Graustufen-Szene mit Rechteck + Kreis, um (dx, dy) verschoben."""
    img = np.full((size, size), 40, dtype=np.uint8)
    cv2.rectangle(img, (60 + dx, 70 + dy), (120 + dx, 130 + dy), 220, -1)
    cv2.circle(img, (170 + dx, 160 + dy), 30, 160, -1)
    # etwas Textur, damit Flow gut definiert ist
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 15, (size, size), dtype=np.uint8)
    return cv2.add(img, noise)


def _flow_to_color(flow: np.ndarray) -> np.ndarray:
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)   # Richtung → Farbton
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def render():
    hero(
        eyebrow="Bildverarbeitung · Optical Flow & Motion",
        title="Optical Flow & Motion",
        sub="Wohin bewegt sich jeder Pixel zwischen zwei Frames? Von Lucas-Kanade über "
            "Farnebäck bis RAFT — die Grundlage von Video-Analyse, Tracking und Stabilisierung."
    )

    tabs = st.tabs([
        "🎯 Grundidee", "🧮 Lucas-Kanade", "🌫️ Dense Flow", "🤖 RAFT & Deep Flow",
        "🧪 Live-Demo", "🎬 Lernvideos",
    ])

    with tabs[0]:
        section_header("Was ist Optical Flow?")
        st.markdown(r"""
**Optical Flow** ist das scheinbare Bewegungsfeld von Helligkeitsmustern zwischen zwei Frames.
Für jeden Pixel schätzt man einen Verschiebungsvektor $(u, v)$.

#### Brightness-Constancy-Annahme
Ein Bildpunkt behält seine Helligkeit über kurze Zeit:
$$I(x, y, t) = I(x + u, y + v, t + 1)$$

Taylor-Entwicklung führt zur **Optical-Flow-Constraint-Gleichung**:
$$I_x u + I_y v + I_t = 0$$

Das ist **eine** Gleichung mit **zwei** Unbekannten → das *Aperture-Problem*.
Man braucht eine zusätzliche Annahme (lokal konstanter Fluss, Glattheit …), um es zu lösen.
        """)
        key_concept("🕳️", "Aperture-Problem",
                    "Durch ein kleines Fenster betrachtet, ist nur die Bewegung senkrecht zur Kante sichtbar. "
                    "Deshalb brauchen alle Verfahren eine Zusatzannahme.")
        key_concept("✨", "Sparse vs. Dense",
                    "Sparse Flow verfolgt nur ausgewählte Keypoints (Lucas-Kanade), Dense Flow schätzt einen "
                    "Vektor für jeden Pixel (Farnebäck, RAFT).")

    with tabs[1]:
        section_header("Lucas-Kanade — sparser Fluss")
        st.markdown(r"""
**Lucas-Kanade** (1981) nimmt an, dass der Fluss in einer kleinen Nachbarschaft **konstant** ist.
Damit hat man mehr Gleichungen als Unbekannte und löst per **Least Squares**:

$$\begin{pmatrix} u \\ v \end{pmatrix} = (A^T A)^{-1} A^T b$$

wobei $A$ die Gradienten $(I_x, I_y)$ der Fenster-Pixel enthält und $b = -I_t$.

- Wird meist auf **guten Keypoints** angewandt (Shi-Tomasi/Harris).
- **Pyramidal** (grob → fein), um auch große Bewegungen zu erfassen.
- Sehr schnell, ideal für Tracking (z.B. `cv2.calcOpticalFlowPyrLK`).
        """)

    with tabs[2]:
        section_header("Dense Flow — Farnebäck")
        st.markdown(r"""
**Farnebäck** (2003) approximiert die Nachbarschaft jedes Pixels durch ein **quadratisches Polynom**
und schätzt daraus die Verschiebung — für **jeden** Pixel.

- Ausgabe: ein $H \times W \times 2$ Feld $(u, v)$.
- Robuster gegen texturarme Flächen als naive Verfahren.
- In OpenCV: `cv2.calcOpticalFlowFarneback(...)`.

Dichter Fluss wird oft als **Farbbild** visualisiert: Farbton = Richtung, Sättigung/Helligkeit = Betrag.
        """)

    with tabs[3]:
        section_header("RAFT & Deep-Learning-Flow")
        st.markdown(r"""
**RAFT** (Recurrent All-Pairs Field Transforms, 2020) ist der Deep-Learning-Durchbruch für dichten Flow.

#### Kernideen
- **All-Pairs Correlation Volume**: Ähnlichkeit aller Pixelpaare zwischen beiden Frames.
- **Iterative GRU-Updates**: das Flussfeld wird schrittweise verfeinert.
- **Sehr genau**, auch bei großen Bewegungen und Verdeckungen.

| Verfahren | Typ | Stärke |
|---|---|---|
| Lucas-Kanade | sparse, klassisch | schnell, Tracking |
| Farnebäck | dense, klassisch | kein Training nötig |
| RAFT | dense, Deep Learning | SOTA-Genauigkeit |
| GMFlow / FlowFormer | dense, Transformer | noch genauer, teurer |
        """)
        info_box("Für Produktion mit GPU: RAFT (torchvision hat vortrainierte Gewichte). "
                 "Ohne GPU/Training: Farnebäck aus OpenCV.", kind="tip")

    # ── Tab 4: Live-Demo ─────────────────────────────────────────────────────
    with tabs[4]:
        lab_header("Dense Optical Flow live", "Verschiebe die Objekte und sieh das geschätzte Flussfeld (Farnebäck).")
        c1, c2 = st.columns(2)
        dx = c1.slider("Verschiebung Δx (px)", -20, 20, 8)
        dy = c2.slider("Verschiebung Δy (px)", -20, 20, 4)

        f1 = _scene(0, 0)
        f2 = _scene(dx, dy)
        flow = cv2.calcOpticalFlowFarneback(f1, f2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_rgb = _flow_to_color(flow)

        i1, i2, i3 = st.columns(3)
        i1.image(f1, caption="Frame 1", use_container_width=True, clamp=True)
        i2.image(f2, caption="Frame 2", use_container_width=True, clamp=True)
        i3.image(flow_rgb, caption="Optical Flow (Farbe = Richtung)", use_container_width=True)

        # Nur bewegte Regionen auswerten (Hintergrund ist statisch → Fluss ~0)
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        moving = mag > max(0.5, 0.3 * mag.max())
        if moving.any():
            u_est = float(np.median(flow[..., 0][moving]))
            v_est = float(np.median(flow[..., 1][moving]))
        else:
            u_est = v_est = 0.0
        m1, m2 = st.columns(2)
        m1.metric("Δx an bewegten Pixeln", f"{u_est:+.1f} px", help=f"Wahrer Wert: {dx:+d}")
        m2.metric("Δy an bewegten Pixeln", f"{v_est:+.1f} px", help=f"Wahrer Wert: {dy:+d}")
        info_box(
            "Ausgewertet wird nur, wo tatsächlich Bewegung erkannt wird — der statische Hintergrund "
            "hat Fluss ~0. In texturarmen Regionen ist der Fluss unsicher (Aperture-Problem).",
            kind="info",
        )

    with tabs[5]:
        section_header("Lernvideos")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Optical Flow — First Principles**")
            video_embed("5AUypv5BNbI", "Optical Flow")
        with col2:
            st.markdown("**RAFT erklärt**")
            video_search("RAFT optical flow deep learning explained", "RAFT erklärt", "")

    divider()
    render_learning_block(
        key_prefix="optical_flow",
        progression=[
            ("🟢", "Guided", "Leite die Optical-Flow-Constraint-Gleichung aus der Brightness-Constancy her.", "Guided", "green"),
            ("🟠", "Challenge", "Erkläre, warum Lucas-Kanade eine Nachbarschaftsannahme braucht.", "Challenge", "amber"),
            ("🔴", "Debug", "Farnebäck liefert im Himmel wirren Fluss — nenne die Ursache.", "Debug", "pink"),
            ("🧩", "Mini-Projekt", "Berechne mit OpenCV den Flow zwischen zwei Frames eines eigenen Videos.", "Projekt", "blue"),
        ],
        mcq_question="Warum ist die Optical-Flow-Constraint-Gleichung allein nicht lösbar?",
        mcq_options=[
            "Eine Gleichung, aber zwei Unbekannte (u, v) — das Aperture-Problem",
            "Die Helligkeit ändert sich nie",
            "Sie ist nichtlinear",
            "Es fehlen die Farbkanäle",
        ],
        mcq_correct_option="Eine Gleichung, aber zwei Unbekannte (u, v) — das Aperture-Problem",
        open_question="Wann würdest du sparsen (Lucas-Kanade) statt dichten (Farnebäck/RAFT) Flow einsetzen?",
        cheat_sheet=[
            "Brightness Constancy: I(x,y,t) = I(x+u, y+v, t+1).",
            "Constraint: Iₓu + I_y v + I_t = 0 (unterbestimmt).",
            "Lucas-Kanade: lokal konstanter Fluss, sparse, schnell.",
            "Farnebäck: dichtes Feld, klassisch, kein Training.",
            "RAFT: Correlation Volume + iterative GRU-Updates, SOTA.",
        ],
        key_takeaways=[
            "Jedes Verfahren löst das Aperture-Problem mit einer anderen Zusatzannahme.",
            "Dichter Flow wird als Farbfeld visualisiert (Farbton = Richtung, Helligkeit = Betrag).",
        ],
        common_errors=[
            "Flow ohne Pyramide bei großen Bewegungen berechnen.",
            "Flow in texturlosen Regionen für zuverlässig halten.",
            "Sparse und dense Verfahren verwechseln.",
        ],
    )
    render_quiz_checkpoint(
        key_prefix="optical_flow",
        module_id="optical_flow",
        question="Was ist die zentrale Neuerung von RAFT gegenüber klassischen Verfahren?",
        options=[
            "All-Pairs Correlation Volume + iterative GRU-Verfeinerung des Flussfelds",
            "Es nutzt nur die Brightness Constancy",
            "Es verzichtet komplett auf Gradienten",
            "Es arbeitet ausschließlich sparse",
        ],
        correct_option="All-Pairs Correlation Volume + iterative GRU-Verfeinerung des Flussfelds",
        checklist=[
            "Ich verstehe die Brightness-Constancy-Annahme.",
            "Ich kann sparse und dense Flow unterscheiden.",
            "Ich kenne das Aperture-Problem.",
        ],
    )
