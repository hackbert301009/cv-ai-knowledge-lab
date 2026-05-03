"""Pose Estimation — Human Pose, 6DoF, interaktiver Skeleton-Visualizer."""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.components import hero, section_header, divider, info_box, video_embed, lab_header, key_concept, step_list


# COCO 17 Keypoints
COCO_KEYPOINTS = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle",
]

# COCO Skeleton-Verbindungen (Keypoint-Index-Paare)
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),             # Gesicht
    (5, 6),                                        # Schultern
    (5, 7), (7, 9), (6, 8), (8, 10),              # Arme
    (5, 11), (6, 12), (11, 12),                    # Torso
    (11, 13), (13, 15), (12, 14), (14, 16),        # Beine
]

# Farben pro Körpergruppe
_BONE_COLORS = {
    "face":       "#F59E0B",
    "shoulders":  "#7C3AED",
    "arms_left":  "#10B981",
    "arms_right": "#3B82F6",
    "torso":      "#EC4899",
    "legs_left":  "#EF4444",
    "legs_right": "#F97316",
}

_BONE_GROUP = [
    "face", "face", "face", "face",
    "shoulders",
    "arms_left", "arms_left", "arms_right", "arms_right",
    "torso", "torso", "torso",
    "legs_left", "legs_left", "legs_right", "legs_right",
]


def _default_pose() -> np.ndarray:
    """COCO 17 Keypoints — normalisierte Koordinaten [0, 1]."""
    kps = np.array([
        [0.50, 0.07],   # 0  Nose
        [0.46, 0.05],   # 1  Left Eye
        [0.54, 0.05],   # 2  Right Eye
        [0.42, 0.08],   # 3  Left Ear
        [0.58, 0.08],   # 4  Right Ear
        [0.37, 0.22],   # 5  Left Shoulder
        [0.63, 0.22],   # 6  Right Shoulder
        [0.28, 0.40],   # 7  Left Elbow
        [0.72, 0.40],   # 8  Right Elbow
        [0.22, 0.56],   # 9  Left Wrist
        [0.78, 0.56],   # 10 Right Wrist
        [0.40, 0.55],   # 11 Left Hip
        [0.60, 0.55],   # 12 Right Hip
        [0.37, 0.72],   # 13 Left Knee
        [0.63, 0.72],   # 14 Right Knee
        [0.35, 0.90],   # 15 Left Ankle
        [0.65, 0.90],   # 16 Right Ankle
    ], dtype=float)
    return kps


def _pose_t_pose() -> np.ndarray:
    kps = _default_pose().copy()
    kps[7] = [0.15, 0.22]   # Left Elbow — gerade aus
    kps[9] = [0.05, 0.22]   # Left Wrist
    kps[8] = [0.85, 0.22]   # Right Elbow
    kps[10] = [0.95, 0.22]  # Right Wrist
    return kps


def _pose_wave() -> np.ndarray:
    kps = _default_pose().copy()
    kps[6] = [0.67, 0.18]   # Right Shoulder
    kps[8] = [0.80, 0.10]   # Right Elbow (raised)
    kps[10] = [0.90, 0.03]  # Right Wrist (wave up)
    return kps


def _pose_squat() -> np.ndarray:
    kps = _default_pose().copy()
    kps[11] = [0.40, 0.62]  # Left Hip
    kps[12] = [0.60, 0.62]  # Right Hip
    kps[13] = [0.33, 0.75]  # Left Knee (bent)
    kps[14] = [0.67, 0.75]  # Right Knee
    kps[15] = [0.38, 0.85]  # Left Ankle
    kps[16] = [0.62, 0.85]  # Right Ankle
    kps[0] = [0.50, 0.15]   # Nose (torso bent)
    kps[5] = [0.38, 0.30]   # Left Shoulder
    kps[6] = [0.62, 0.30]   # Right Shoulder
    return kps


_POSES = {
    "Standard (Stehen)": _default_pose,
    "T-Pose": _pose_t_pose,
    "Winken": _pose_wave,
    "Kniebeuge": _pose_squat,
}


def _draw_skeleton(kps: np.ndarray, confidence: np.ndarray | None = None) -> go.Figure:
    """Plotly-Figur mit Skeleton."""
    fig = go.Figure()

    for i, (a, b) in enumerate(COCO_SKELETON):
        color = _BONE_COLORS[_BONE_GROUP[i]]
        xa, ya = kps[a]
        xb, yb = kps[b]
        conf_mean = 1.0 if confidence is None else (confidence[a] + confidence[b]) / 2
        opacity = max(0.2, conf_mean)
        fig.add_trace(go.Scatter(
            x=[xa, xb], y=[1 - ya, 1 - yb],
            mode="lines",
            line=dict(color=color, width=4),
            opacity=opacity,
            showlegend=False,
        ))

    # Keypoints
    confs = np.ones(17) if confidence is None else confidence
    kp_colors = ["#FACC15" if c > 0.5 else "#6B7280" for c in confs]
    fig.add_trace(go.Scatter(
        x=kps[:, 0],
        y=1 - kps[:, 1],
        mode="markers+text",
        marker=dict(size=10, color=kp_colors, line=dict(width=2, color="white")),
        text=[f"{i}" for i in range(17)],
        textposition="top right",
        textfont=dict(size=9, color="white"),
        hovertext=COCO_KEYPOINTS,
        showlegend=False,
    ))

    fig.update_layout(
        template="plotly_dark",
        height=500,
        xaxis=dict(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[0, 1.1], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"),
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
    )
    return fig


def render():
    hero(
        eyebrow="State-of-the-Art · Pose Estimation",
        title="Pose Estimation",
        sub="Menschliche Körperhaltung erkennen — von OpenPose bis ViTPose. "
            "6DoF Objektpose für Robotik. Interaktiver Skeleton-Visualizer inklusive.",
    )

    tabs = st.tabs([
        "🧍 Überblick",
        "📜 Geschichte & Modelle",
        "🦴 COCO Keypoints",
        "🤖 6DoF Objektpose",
        "💻 Code",
        "🧪 Skeleton-Visualizer",
        "🎬 Lernvideos",
    ])

    # ── Tab 0: Überblick ──────────────────────────────────────────────────────
    with tabs[0]:
        section_header("Was ist Pose Estimation?")
        st.markdown("""
**Pose Estimation** erkennt die Position und Orientierung von Körperteilen (oder Objekten)
in einem Bild oder Video.

**Zwei Hauptaufgaben:**
1. **Human Pose Estimation (HPE)**: Keypoints des menschlichen Körpers finden (Schultern, Knie etc.)
2. **6DoF Objektpose**: Position (x,y,z) + Rotation (roll, pitch, yaw) eines Objekts im 3D-Raum

**Anwendungen:**
- 🏃 Sport-Analyse und Bewegungskorrektur
- 🤖 Roboter-Interaktion (Hand-Greifer-Kalibrierung)
- 🎮 Motion Capture für Animation
- 🏥 Physio-Therapie und Sturzprävention
- 🚗 Fahrerüberwachung in Autos
- 💃 Tanzen-Lernen-Apps (Beat Saber-Style)
        """)

        key_concept("🗺️", "Heatmap-basiert",
                    "Für jeden Keypoint wird eine 2D-Wahrscheinlichkeitskarte (Gaussian Blob) vorhergesagt. "
                    "Dann wird das Maximum gesucht.")
        key_concept("📊", "Regression-basiert",
                    "Direkte Vorhersage der (x, y)-Koordinaten via Regression. Schneller, aber weniger präzise.")
        key_concept("🔝 → 🔽", "Top-Down",
                    "Erst Personen-Detection (Bounding Box), dann Keypoints für jede Person. Genauer, langsamer.")
        key_concept("🔽 → 🔝", "Bottom-Up",
                    "Erst alle Keypoints im Bild finden, dann zu Personen gruppieren. Skaliert bei vielen Personen.")

        divider()
        st.markdown("""
#### Metriken

| Metrik | Beschreibung |
|--------|-------------|
| **OKS** | Object Keypoint Similarity — wie IoU, aber für Keypoints |
| **PCK** | Percentage of Correct Keypoints — Anteil korrekt erkannter Joints (Schwellwert %-Körpergröße) |
| **mAP@OKS** | Mean AP über verschiedene OKS-Schwellen (0.5, 0.75, 0.5:0.95) |
| **MPJPE** | Mean Per-Joint Position Error — 3D-Fehler in Millimetern |
        """)

    # ── Tab 1: Geschichte & Modelle ──────────────────────────────────────────
    with tabs[1]:
        section_header("Von OpenPose bis ViTPose — eine Geschichte")

        timeline = [
            ("2017", "OpenPose", "CMU",
             "Erster Echtzeit-Multi-Person-Pose-Estimator. Bottom-Up: "
             "Part Affinity Fields (PAFs) verbinden Keypoints zu Skeletten. "
             "Läuft auf einer Consumer-GPU in Echtzeit.",
             "#7C3AED"),
            ("2019", "HRNet", "Microsoft Research",
             "High-Resolution Net: Parallele Auflösungszweige statt sequentiellem Down-/Upsampling. "
             "Hält hochauflösende Repräsentationen während der ganzen Verarbeitung. "
             "State-of-the-Art auf COCO für Jahre.",
             "#3B82F6"),
            ("2019", "SimpleBaseline", "Microsoft Research",
             "Überraschend: Einfaches ResNet + Deconvolutional Head. Beweist, dass "
             "simple Architekturen sehr weit kommen. Top-Down.",
             "#10B981"),
            ("2020", "EfficientPose", "Community",
             "EfficientNet-Backbone für Pose — kleiner und schneller als HRNet, "
             "ähnliche Accuracy. Ideal für Edge-Deployment.",
             "#F59E0B"),
            ("2022", "ViTPose", "UT Austin",
             "Vision Transformer als Backbone. Pre-trained ViT + lightweight Head. "
             "Schlägt alle CNNs auf COCO. Zeigt: Transformer > CNN auch bei Pose.",
             "#EC4899"),
            ("2023", "RTMPose", "OpenMMLab",
             "Real-Time Multi-Person Pose. Optimiert für Deployment: "
             "ONNX, TensorRT, schneller als OpenPose auf CPU.",
             "#EF4444"),
            ("2024", "DWPose / UniPose", "Community",
             "Foundation Models für Pose: Ein Modell für alle Körperteile "
             "(Ganzkörper, Gesicht, Hände). ControlNet-Integration.",
             "#F97316"),
        ]

        for year, name, org, desc, color in timeline:
            st.markdown(
                f"""<div style="display:flex;gap:1rem;margin-bottom:1rem;padding:1rem;
                    background:rgba(255,255,255,0.03);border-radius:12px;
                    border-left:4px solid {color}">
                    <div style="min-width:60px;font-size:1.1rem;font-weight:700;color:{color}">{year}</div>
                    <div>
                        <div style="font-weight:700;font-size:1rem">{name}
                            <span style="font-size:0.75rem;color:#9CA3AF;margin-left:0.5rem">{org}</span>
                        </div>
                        <div style="font-size:0.85rem;color:#D1D5DB;margin-top:0.25rem">{desc}</div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

        divider()
        st.markdown("""
#### Benchmark (COCO val2017, single-scale)

| Modell | Backbone | AP | Params | GFLOPs |
|--------|----------|-----|--------|--------|
| SimpleBaseline | ResNet-50 | 70.4 | 34M | 8.9 |
| HRNet-W32 | HRNet | 74.4 | 28.5M | 7.1 |
| HRNet-W48 | HRNet | 75.5 | 63.6M | 14.6 |
| ViTPose-S | ViT-S | 73.8 | 25M | 5.6 |
| ViTPose-B | ViT-B | 75.8 | 86M | 18.0 |
| ViTPose-H | ViT-H | **79.1** | 632M | 213 |
| RTMPose-m | CSPNeXt | 74.2 | 13M | 4.0 |
        """)

    # ── Tab 2: COCO Keypoints ────────────────────────────────────────────────
    with tabs[2]:
        section_header("COCO 17 Keypoints — das Standard-Schema")
        st.markdown("""
Das **COCO Pose Dataset** definiert 17 Keypoints pro Person — der Industrie-Standard:
        """)

        kp_data = {
            "ID": list(range(17)),
            "Name": COCO_KEYPOINTS,
            "Gruppe": ["Gesicht"] * 5 + ["Oberkörper"] * 6 + ["Unterkörper"] * 6,
        }
        st.dataframe(kp_data, use_container_width=True, height=400)

        divider()
        section_header("Skeleton-Verbindungen")
        st.markdown(f"**{len(COCO_SKELETON)} Verbindungen** verbinden die 17 Keypoints zu einem Skelett:")
        conn_data = {
            "Von": [COCO_KEYPOINTS[a] for a, b in COCO_SKELETON],
            "Nach": [COCO_KEYPOINTS[b] for a, b in COCO_SKELETON],
            "Farbe": [_BONE_GROUP[i] for i in range(len(COCO_SKELETON))],
        }
        st.dataframe(conn_data, use_container_width=True)

        divider()
        section_header("Erweiterungen: Ganzkörper")
        st.markdown("""
**133 Keypoints** (Ganzkörper) = 17 Body + 68 Gesicht + 42 Hände + 6 Füße:
- **MediaPipe Holistic**: 468 Gesichts-Landmarks + 21 Hand + 33 Body
- **SMPL**: Parametrisches 3D-Body-Modell mit 6890 Vertices
- **DWPose**: Für ControlNet-gesteuerte Bildgenerierung
        """)

    # ── Tab 3: 6DoF Objektpose ───────────────────────────────────────────────
    with tabs[3]:
        section_header("6DoF Objektpose — Position + Rotation im 3D-Raum")
        st.markdown(r"""
**6 Degrees of Freedom (6DoF)** = 3 Translation (x, y, z) + 3 Rotation (roll, pitch, yaw).

Gegeben: Bild + Kamera-Intrinsics $K$, bekannte 3D-Objektpunkte $\mathbf{P}_i$.
Gesucht: Rotationsmatrix $R$ und Translationsvektor $t$.

$$\mathbf{p}_i = K [R | t] \mathbf{P}_i$$

Dies wird mit **PnP (Perspective-n-Point)** gelöst — z.B. EPnP oder iterativer Levenberg-Marquardt.
        """)

        key_concept("📐", "Kamera-Intrinsics",
                    "Brennweite (fx, fy) und Bildmittelpunkt (cx, cy) der Kamera. Nötig für 3D↔2D Projektion.")
        key_concept("🎯", "PnP Problem",
                    "Gegeben n 2D-3D-Korrespondenzpaare, finde R und t. OpenCV: cv2.solvePnP()")
        key_concept("📦", "Symmetrie-Problem",
                    "Symmetrische Objekte (Würfel, Zylinder) sind schwer — ADD-S Metrik summiert minimalen Abstand.")

        divider()
        st.markdown("""
#### 6DoF Methoden im Vergleich

| Methode | Ansatz | Stärken | Schwächen |
|---------|--------|---------|-----------|
| **PoseCNN** | Direkte Rotation + Translation Regression | Einfach | Symmetrie-Probleme |
| **PVNet** | Keypoints im Bild → PnP | Robust bei Verdeckung | Training-aufwendig |
| **FoundPose** | Foundation Model, Zero-Shot | Kein 3D-Modell nötig | Langsam |
| **GDR-Net** | Geometry-guided Direkt-Regression | SOTA auf YCB-V | Komplex |
| **FoundPose** | ViT + Template Matching | Zero-Shot möglich | – |
        """)

        st.code("""
# OpenCV PnP — klassische Lösung
import cv2
import numpy as np

# 3D-Modellpunkte (bekannt aus CAD-Modell)
object_points = np.array([
    [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
], dtype=np.float32)

# 2D-Bildpunkte (z.B. von Keypoint-Detektor)
image_points = np.array([
    [100, 200], [250, 195],
    [248, 350], [98, 355],
], dtype=np.float32)

# Kamera-Intrinsics
K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
dist = np.zeros(5)  # keine Verzeichnung

success, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist)
R, _ = cv2.Rodrigues(rvec)  # Rotation-Vektor → Matrix

print(f"Translation: {tvec.flatten()}")
print(f"Rotation:\\n{R}")
        """, language="python")

    # ── Tab 4: Code ──────────────────────────────────────────────────────────
    with tabs[4]:
        section_header("Praxis-Code — MMPose & Ultralytics")
        st.code("""
# ── MMPose: HRNet auf COCO ─────────────────────────────────────────────────
from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules

register_all_modules()

config = "configs/body_2d_keypoint/topdown_heatmap/coco/hrnet_w48_coco_256x192.py"
checkpoint = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth"

model = init_model(config, checkpoint, device="cuda:0")
results = inference_topdown(model, "image.jpg")

keypoints = results[0].pred_instances.keypoints   # (17, 2)
scores    = results[0].pred_instances.keypoint_scores  # (17,)
        """, language="python")

        divider()
        st.code("""
# ── Ultralytics YOLOv8-Pose (einfachste API) ──────────────────────────────
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n-pose.pt")   # n=nano, s/m/l/x auch verfügbar
results = model("image.jpg")

for r in results:
    kps = r.keypoints.xy.cpu().numpy()   # (N_personen, 17, 2)
    confs = r.keypoints.conf.cpu().numpy()  # (N_personen, 17)
    # Annotiertes Bild
    annotated = r.plot()
    cv2.imshow("Pose", annotated)
        """, language="python")

        divider()
        st.code("""
# ── MediaPipe Pose (Cross-Platform, auch Mobile) ──────────────────────────
import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        # z.B. rechte Schulter:
        print(f"Schulter: ({lm[12].x:.3f}, {lm[12].y:.3f}, {lm[12].z:.3f})")

cap.release()
        """, language="python")

        info_box(
            "Für Echtzeit auf CPU: MediaPipe (30+ FPS). Für höchste Accuracy: ViTPose-H. "
            "Für gutes Mittelmaß: Ultralytics YOLOv8-Pose (einfache API, schnell).",
            kind="info",
        )

    # ── Tab 5: Skeleton-Visualizer ───────────────────────────────────────────
    with tabs[5]:
        lab_header("Interaktiver Skeleton-Visualizer", "COCO 17 Keypoints — 2D-Ansicht")

        col_ctrl, col_vis = st.columns([1, 2])

        with col_ctrl:
            pose_name = st.selectbox("Pose auswählen", list(_POSES.keys()))
            kps = _POSES[pose_name]().copy()

            st.markdown("**Confidence je Keypoint:**")
            conf_preset = st.select_slider(
                "Konfidenz-Niveau",
                options=["Niedrig (0.3)", "Mittel (0.6)", "Hoch (0.9)", "Perfekt (1.0)"],
                value="Hoch (0.9)",
            )
            conf_val = {"Niedrig (0.3)": 0.3, "Mittel (0.6)": 0.6,
                        "Hoch (0.9)": 0.9, "Perfekt (1.0)": 1.0}[conf_preset]

            # Manche Keypoints zufällig ausblenden (simuliert Verdeckung)
            rng = np.random.default_rng(42)
            if conf_val < 1.0:
                confidence = rng.uniform(max(0, conf_val - 0.3), min(1, conf_val + 0.15), size=17)
            else:
                confidence = np.ones(17)

            st.markdown("---")
            st.markdown("**Winkel berechnen:**")
            joint_a = st.selectbox("Gelenk A", COCO_KEYPOINTS, index=7)   # Left Elbow
            joint_b = st.selectbox("Gelenk B (Mitte)", COCO_KEYPOINTS, index=5)  # Left Shoulder
            joint_c = st.selectbox("Gelenk C", COCO_KEYPOINTS, index=11)  # Left Hip

            idx_a = COCO_KEYPOINTS.index(joint_a)
            idx_b = COCO_KEYPOINTS.index(joint_b)
            idx_c = COCO_KEYPOINTS.index(joint_c)

            def angle_3pts(p1, p2, p3):
                v1 = p1 - p2
                v2 = p3 - p2
                cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
                return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

            angle = angle_3pts(kps[idx_a], kps[idx_b], kps[idx_c])
            st.metric(f"Winkel {joint_b}", f"{angle:.1f}°")

        with col_vis:
            fig = _draw_skeleton(kps, confidence)
            st.plotly_chart(fig, use_container_width=True)

        divider()
        lab_header("Heatmap-Visualisierung", "Wie Keypoint-Detektoren intern arbeiten")
        st.markdown("""
Heatmap-basierte Pose-Estimatoren (HRNet, ViTPose) sagen für jeden der 17 Keypoints
eine **Gaussian-Heatmap** voraus. Der Peak ist die Keypoint-Position.
        """)
        kp_idx = st.select_slider("Keypoint anzeigen", options=list(range(17)),
                                   format_func=lambda i: f"{i}: {COCO_KEYPOINTS[i]}", value=5)

        # Synthetische Heatmap für ausgewählten Keypoint
        H, W = 64, 48
        center_y = int(kps[kp_idx][1] * H)
        center_x = int(kps[kp_idx][0] * W)
        sigma = 3.0
        yy, xx = np.mgrid[:H, :W]
        heatmap = np.exp(-((xx - center_x)**2 + (yy - center_y)**2) / (2 * sigma**2))
        heatmap *= confidence[kp_idx]  # Confidence beeinflusst Peak-Stärke

        import plotly.express as px
        fig2 = px.imshow(
            heatmap, color_continuous_scale="Turbo",
            title=f"Heatmap: {COCO_KEYPOINTS[kp_idx]} (Confidence={confidence[kp_idx]:.2f})",
            labels=dict(color="Conf"),
        )
        fig2.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 6: Videos ─────────────────────────────────────────────────────────
    with tabs[6]:
        section_header("Lernvideos")
        video_embed("bEVGMMBhBhU", "OpenPose erklärt",
                    "CMU OpenPose: Part Affinity Fields und Bottom-Up Multi-Person Pose")
        divider()
        video_embed("c6PYHj0IbEg", "Human Pose Estimation — Two Minute Papers",
                    "Überblick über moderne Pose-Estimation Methoden")
        divider()
        video_embed("5MXMY2b5b0I", "ViTPose — Vision Transformer für Pose",
                    "Wie ViT CNNs bei Pose Estimation übertrifft")
        divider()
        video_embed("MQp5BMN6N9M", "MediaPipe Pose Tutorial",
                    "Einstieg in MediaPipe Pose mit Python — Echtzeit-Demo")
