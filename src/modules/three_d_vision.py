"""3D Computer Vision - Geometrie, Epipolar, SfM, SLAM und NeRF."""
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
    render_quiz_checkpoint,
    video_embed,
)


def _epipolar_line(F: np.ndarray, x: np.ndarray) -> np.ndarray:
    return F @ x


def _line_points(line: np.ndarray, x_min: float, x_max: float) -> tuple[float, float, float, float]:
    a, b, c = line
    y1 = -(a * x_min + c) / (b + 1e-9)
    y2 = -(a * x_max + c) / (b + 1e-9)
    return x_min, y1, x_max, y2


def render():
    hero(
        eyebrow="State-of-the-Art · 3D Vision",
        title="3D Computer Vision Essentials",
        sub="Kamera-Geometrie, Epipolar-Constraints, SfM/SLAM und NeRF/Gaussian Splatting.",
    )

    tabs = st.tabs(
        [
            "📷 Kamera",
            "📐 Epipolar",
            "🗺 SfM & SLAM",
            "🧠 NeRF & 3DGS",
            "🧪 Epipolar Lab",
            "💻 Code",
            "✅ Checkpoint",
            "🎬 Videos",
        ]
    )

    with tabs[0]:
        section_header("Pinhole Kamera-Modell")
        st.latex(r"\mathbf{p} \sim K [R|t] \mathbf{P}")
        st.markdown(
            """
- **Intrinsics K:** Brennweite und principal point.
- **Extrinsics [R|t]:** Pose der Kamera im Weltkoordinatensystem.
- **Projection:** 3D Punkt -> 2D Pixel.
            """
        )
        key_concept("🎯", "Kalibrierung", "Ermittelt K und Verzeichnung per Schachbrett oder AprilTags.")
        key_concept("🧭", "Pose", "R und t beschreiben Blickrichtung und Position der Kamera.")

    with tabs[1]:
        section_header("Epipolar-Geometrie")
        st.latex(r"\mathbf{x}'^T F \mathbf{x} = 0")
        st.markdown(
            """
Die Fundamental-Matrix koppelt korrespondierende Punkte zweier Bilder.
Ein Punkt in Bild A erzeugt in Bild B eine **Epipolarlinie** - die Suche wird 1D statt 2D.
            """
        )
        info_box("RANSAC ist in der Praxis Pflicht, um Ausreisser bei Korrespondenzen zu filtern.", kind="tip")

    with tabs[2]:
        section_header("SfM und SLAM")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
**Structure from Motion (SfM)**
- Offline Rekonstruktion aus Bildsammlungen
- Bundle Adjustment als Kernoptimierung
- Ergebnis: Sparse/ Dense 3D Punktwolke
                """
            )
        with c2:
            st.markdown(
                """
**SLAM**
- Simultaneous Localization and Mapping
- Online, oft in Echtzeit (Robotics/AR)
- Loop Closure reduziert Drift
                """
            )

        divider()
        st.markdown(
            """
| System | Typ | Staerke |
|---|---|---|
| COLMAP | SfM | Sehr robust fuer Offline-Reconstruction |
| ORB-SLAM3 | SLAM | Starker Klassiker fuer Echtzeit |
| VINS-Fusion | Visual-Inertial | Nutzt Kamera + IMU |
            """
        )

    with tabs[3]:
        section_header("NeRF und Gaussian Splatting")
        st.markdown(
            """
- **NeRF:** MLP approximiert Volumendichte + Farbe entlang Rays.
- **3D Gaussian Splatting:** Szene als anisotrope Gaussians, sehr schnelle Renderzeiten.
- **Trade-off:** NeRF oft detailtreu, 3DGS meist schneller fuer interaktive Darstellung.
            """
        )
        key_concept("🌫️", "Volume Rendering", "Farbe entsteht durch Integration entlang eines Rays.")
        key_concept("⚡", "Real-Time Rendering", "3DGS bringt viele Szenen in den Echtzeitbereich.")

    with tabs[4]:
        lab_header("Epipolar-Live-Demo", "Verschiebe einen Punkt und beobachte die Linie in Kamera B.")
        f_scale = st.slider("Fundamental Matrix Skalierung", 0.4, 2.0, 1.0, 0.1)
        x = st.slider("x in Bild A", 20, 320, 140, 5)
        y = st.slider("y in Bild A", 20, 240, 120, 5)
        F = f_scale * np.array(
            [
                [0.0, -0.0008, 0.12],
                [0.0008, 0.0, -0.18],
                [-0.10, 0.16, 1.0],
            ]
        )
        point = np.array([float(x), float(y), 1.0])
        line = _epipolar_line(F, point)
        x1, y1, x2, y2 = _line_points(line, 0.0, 360.0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", marker=dict(size=12, color="#10B981"), name="Point A"))
        fig.add_trace(
            go.Scatter(
                x=[x1, x2],
                y=[y1, y2],
                mode="lines",
                line=dict(color="#F59E0B", width=3),
                name="Epipolar line in B",
            )
        )
        fig.update_layout(
            template="plotly_dark",
            height=420,
            xaxis=dict(range=[0, 360], title="x"),
            yaxis=dict(range=[260, 0], title="y"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.code(f"Line coefficients (a,b,c): {line.round(4).tolist()}")

    with tabs[5]:
        section_header("Praxis-Code")
        st.code(
            """import cv2
import numpy as np

# Korrespondenzen
pts1 = np.array([[120, 90], [200, 85], [160, 140], [90, 160]], dtype=np.float32)
pts2 = np.array([[126, 95], [208, 92], [166, 146], [98, 165]], dtype=np.float32)

# Fundamental Matrix mit RANSAC
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
print(F)
            """,
            language="python",
        )
        divider()
        st.code(
            """# COLMAP command-line (Beispiel)
colmap feature_extractor --database_path db.db --image_path images/
colmap exhaustive_matcher --database_path db.db
colmap mapper --database_path db.db --image_path images/ --output_path sparse/
            """,
            language="bash",
        )

    with tabs[6]:
        render_quiz_checkpoint(
            key_prefix="three_d_vision",
            question="Welche Gleichung beschreibt die Epipolar-Constraint?",
            options=[
                "x'^T F x = 0",
                "x = K R t",
                "IoU = A/B",
                "MOTA = 1 - FP",
            ],
            correct_option="x'^T F x = 0",
            checklist=[
                "Ich kann Intrinsics und Extrinsics unterscheiden.",
                "Ich verstehe den Nutzen der Epipolarlinien fuer Matching.",
                "Ich kenne den Unterschied zwischen SfM und SLAM.",
            ],
            capstone_prompt="Skizziere ein Mini-Projekt fuer 3D-Rekonstruktion mit 2 Handy-Kameras "
            "(Datenerfassung, Kalibrierung, Rekonstruktion, Validierung).",
        )

    with tabs[7]:
        section_header("Lernvideos")
        video_embed("3xH8NQfV5LQ", "Epipolar Geometrie", "Anschauliche Erklaerung der Fundamental-Matrix.")
        divider()
        video_embed("Vj6s7qX4xg4", "COLMAP Tutorial", "SfM Pipeline von Bildern zur 3D Punktwolke.")
        divider()
        video_embed("JuH79E8rdKc", "NeRF Intro", "Warum Neural Radiance Fields die 3D Welt veraendert haben.")
