"""Objekterkennung und Tracking - YOLO, DETR, mAP, NMS und MOT."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.components import (
    divider,
    hero,
    info_box,
    key_concept,
    lab_header,
    section_header,
    step_list,
    render_quiz_checkpoint,
    video_embed,
)


def _compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter + 1e-9
    return inter / union


def _run_nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        rest = order[1:]
        filtered = []
        for idx in rest:
            if _compute_iou(boxes[current], boxes[int(idx)]) < iou_thr:
                filtered.append(int(idx))
        order = np.array(filtered, dtype=int)
    return keep


def _mot_simulation(frames: int, miss_prob: float, fp_prob: float) -> tuple[int, int, int]:
    rng = np.random.default_rng(7)
    gt = frames
    misses = int(np.sum(rng.random(frames) < miss_prob))
    fps = int(np.sum(rng.random(frames) < fp_prob))
    id_switches = int(np.sum(rng.random(frames) < (0.12 + 0.4 * miss_prob)))
    mota = 1.0 - (misses + fps + id_switches) / max(1, gt)
    return misses, fps, max(0, int(round(100 * mota)))


def render():
    hero(
        eyebrow="Bildverarbeitung · Detection & Tracking",
        title="Objekterkennung und Tracking",
        sub=(
            "Von YOLO und DETR ueber mAP/NMS bis Multi-Object Tracking mit "
            "ByteTrack und DeepSORT."
        ),
    )

    tabs = st.tabs(
        [
            "🎯 Grundlagen",
            "🏗 Modelle",
            "📏 Metriken",
            "🧪 NMS Lab",
            "🚶 MOT Lab",
            "💻 Code",
            "✅ Checkpoint",
            "🎬 Videos",
        ]
    )

    with tabs[0]:
        section_header("Was ist Detection und was ist Tracking?")
        st.markdown(
            """
- **Objekterkennung (Detection):** In einem Einzelbild Klassen + Bounding Boxes finden.
- **Tracking:** Erkannte Objekte ueber Zeit konsistent mit IDs verfolgen.
- **MOT Pipeline:** Detection -> Association -> Track-Management.
            """
        )
        key_concept("📦", "Bounding Box", "Rechteck mit (x1, y1, x2, y2) um ein Objekt.")
        key_concept("🧷", "Association", "Zuordnung aktueller Detektionen zu existierenden Tracks.")
        key_concept("🆔", "ID Switch", "Tracker verliert die Identitaet eines Objekts.")

        divider()
        st.markdown(
            """
#### Typische Einsatzfaelle
- Retail Analytics (Personenfluss, Shelf-Tracking)
- Verkehrsanalyse (Fahrzeuge, Zaehlung, Verhalten)
- Industrie (Defekterkennung + Teileverfolgung)
- Sport-Analyse (Spielertracking)
            """
        )

    with tabs[1]:
        section_header("YOLO vs DETR")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
**YOLO-Familie**
- One-stage, sehr schnell
- Anchor-free Varianten in neueren Versionen
- Sehr stark fuer Echtzeit-Use-Cases
                """
            )
        with c2:
            st.markdown(
                """
**DETR-Familie**
- Transformer-basierte Set-Prediction
- Kein klassisches NMS erforderlich
- Sehr sauberer End-to-End Ansatz
                """
            )

        divider()
        st.markdown(
            """
| Modell | Staerke | Schwaeche | Typisches Setting |
|---|---|---|---|
| YOLOv8/11 | Sehr niedrige Latenz | Kleine Objekte teils schwieriger | Edge, Live-Cams |
| RT-DETR | Gute Balance aus Speed/Qualitaet | Hoehere Komplexitaet | Server Inference |
| D-FINE / DINO DETR | Hohe AP | Mehr Compute | Offline-Batch |
            """
        )

    with tabs[2]:
        section_header("mAP, IoU, MOTA kompakt")
        st.latex(r"IoU = \frac{|B_{pred} \cap B_{gt}|}{|B_{pred} \cup B_{gt}|}")
        st.latex(r"mAP = \frac{1}{C} \sum_{c=1}^{C} AP_c")
        st.latex(r"MOTA = 1 - \frac{FN + FP + IDSW}{GT}")
        info_box(
            "mAP misst Detection-Qualitaet pro Klasse. MOTA fokussiert auf Tracking-Fehler ueber Zeit.",
            kind="tip",
        )
        step_list(
            [
                ("Detektionen sortieren", "Nach Confidence absteigend."),
                ("TP/FP zuordnen", "Per IoU-Schwelle gegen Ground Truth."),
                ("Precision-Recall Kurve", "Aus TP/FP Verlauf berechnen."),
                ("AP und mAP", "Integral je Klasse und danach Mittelwert."),
            ]
        )

    with tabs[3]:
        lab_header("NMS interaktiv", "Sieh, welche Boxen NMS verwirft.")
        boxes = np.array(
            [
                [10, 12, 58, 60],
                [13, 15, 60, 62],
                [62, 16, 95, 55],
                [15, 65, 56, 95],
                [64, 62, 95, 95],
            ],
            dtype=float,
        )
        scores = np.array([0.93, 0.87, 0.72, 0.66, 0.61], dtype=float)
        thr = st.slider("IoU-Schwelle", 0.1, 0.9, 0.5, 0.05)
        kept = _run_nms(boxes, scores, thr)
        st.write(f"Behaltene Box-Indizes: `{kept}`")

        fig = go.Figure()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            is_kept = i in kept
            color = "#10B981" if is_kept else "#EF4444"
            fig.add_shape(type="rect", x0=x1, y0=y1, x1=x2, y1=y2, line=dict(color=color, width=3))
            fig.add_annotation(x=x1, y=y1, text=f"{i}: {scores[i]:.2f}", showarrow=False, font=dict(color=color))
        fig.update_layout(
            template="plotly_dark",
            height=420,
            xaxis=dict(range=[0, 100], showgrid=False),
            yaxis=dict(range=[100, 0], showgrid=False),
            margin=dict(l=8, r=8, t=10, b=8),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        lab_header("MOT Fehlermodell", "Wie Misses und False Positives MOTA beeinflussen.")
        frames = st.slider("Frames", 50, 800, 300, 10)
        miss_prob = st.slider("Miss-Wahrscheinlichkeit", 0.0, 0.6, 0.15, 0.01)
        fp_prob = st.slider("False-Positive-Wahrscheinlichkeit", 0.0, 0.6, 0.10, 0.01)
        misses, fps, mota_pct = _mot_simulation(frames, miss_prob, fp_prob)
        st.metric("Misses (FN)", misses)
        st.metric("False Positives (FP)", fps)
        st.metric("Approx. MOTA", f"{mota_pct}%")
        info_box(
            "ByteTrack reduziert oft FNs durch geschicktes Matchen auch low-score Detektionen.",
            kind="success",
        )

    with tabs[5]:
        section_header("Praxis-Code")
        st.code(
            """from ultralytics import YOLO

detector = YOLO("yolo11s.pt")
tracker  = YOLO("yolo11s.pt")

# Detection
det_res = detector("frame.jpg")

# MOT with ByteTrack
mot_res = tracker.track(
    source="video.mp4",
    tracker="bytetrack.yaml",
    conf=0.25,
    iou=0.5,
    persist=True,
)
            """,
            language="python",
        )
        divider()
        st.code(
            """# MOT Metriken mit motmetrics
import motmetrics as mm

acc = mm.MOTAccumulator(auto_id=True)
# ... update acc per frame ...
mh = mm.metrics.create()
summary = mh.compute(acc, metrics=["mota", "idf1", "num_switches"], name="run")
print(summary)
            """,
            language="python",
        )

    with tabs[6]:
        render_quiz_checkpoint(
            key_prefix="object_tracking",
            question="Welche Metrik gehoert primaer zum Multi-Object Tracking?",
            options=["mAP", "MOTA", "IoU", "PSNR"],
            correct_option="MOTA",
            checklist=[
                "Ich kann IoU und mAP kurz erklaeren.",
                "Ich verstehe, warum NMS notwendig ist.",
                "Ich kenne den Einfluss von FN/FP/IDSW auf MOTA.",
            ],
            capstone_prompt="Baue eine Mini-Pipeline: YOLO-Detection + ByteTrack auf einem 30s Video. "
            "Welche Fehler beobachtest du bei Occlusions?",
        )

    with tabs[7]:
        section_header("Lernvideos")
        video_embed("ag3DLKsl2vk", "YOLO Erklaerung", "Intuitiver Einstieg in YOLO Detection.")
        divider()
        video_embed("zQeNQQ6L2j8", "DETR in 5 Minuten", "Set Prediction und Hungarian Matching.")
        divider()
        video_embed("0P4vI4c9N4Q", "ByteTrack", "Warum low-score Boxen bei Tracking helfen.")
