"""Roadmap & Lernpfad — strukturierter Weg durch alle Module."""
import streamlit as st
from src.components import hero, section_header, divider, info_box, card, render_card_grid


def render():
    hero(
        eyebrow="Roadmap",
        title="Dein Weg zum CV &amp; KI Experten",
        sub="Eine durchdachte Reihenfolge — mit Zeitschätzungen und Meilensteinen. "
            "Egal ob du 4 Wochen oder 6 Monate Zeit hast: hier findest du deinen Pfad."
    )

    # ----- Zeitplaene -----
    section_header("Waehle dein Tempo")
    pcols = st.columns(3)
    plans = [
        ("⚡", "Sprint (6 Wochen)", "Fokus auf Kernmodule + 2 Spezialisierungen. Ca. 2h/Tag.", ["intensiv"]),
        ("🎯", "Standard (16 Wochen)", "Empfohlen: vollstaendige Lernstrecke mit Projekten. Ca. 1h/Tag.", ["balanced"]),
        ("🌳", "Tiefgang (6+ Monate)", "Mit Paper-Reproduktionen, Evaluation und Deployment in Tiefe.", ["mastery"]),
    ]
    for col, (icon, title, desc, tags) in zip(pcols, plans):
        with col:
            st.markdown(card(icon, title, desc, tags, ["amber"]), unsafe_allow_html=True)

    divider()

    section_header("Phase 1 — Fundamente", "Wochen 1-3 · Mathe und Denkwerkzeuge")
    render_card_grid([
        card("📐", "Mathe-Crashkurs", "Notation und Kernideen.", ["1 Woche"], ["blue"]),
        card("🧮", "Lineare Algebra", "Tensoren, Projektionen, Eigenwerte.", ["1 Woche"], ["blue"]),
        card("∂", "Analysis", "Gradienten und Optimierung.", ["3 Tage"], ["blue"]),
        card("🎲", "Wahrscheinlichkeit", "Bayes, Entropie, Unsicherheit.", ["4 Tage"], ["blue"]),
        card("🎮", "Tensor Playground", "Broadcasting, Reshape, Einsum.", ["2 Tage"], ["blue"]),
    ], cols=3)

    divider()

    section_header("Phase 2 — Klassische Computer Vision", "Wochen 4-6 · Bildverarbeitung verstehen")
    render_card_grid([
        card("🖼️", "Bildgrundlagen", "Pixel, Farbraeume, Sampling.", ["2 Tage"], ["blue"]),
        card("📷", "Kamera-Pipeline", "Sensor, Bayer, ISP, Noise.", ["3 Tage"], ["blue"]),
        card("🌫️", "Filter & Faltung", "Klassische Operatoren.", ["3 Tage"], ["blue"]),
        card("📏", "Kantendetektion", "Sobel, Canny, Laplace.", ["2 Tage"], ["blue"]),
        card("🔑", "Features", "SIFT, ORB, Matching.", ["3 Tage"], ["blue"]),
        card("✂️", "Segmentierung", "Threshold, Watershed, GrabCut.", ["3 Tage"], ["blue"]),
        card("🎯", "Objekterkennung & Tracking", "YOLO/DETR, NMS, MOT.", ["4 Tage"], ["blue"]),
    ], cols=3)

    divider()

    section_header("Phase 3 — Deep Learning Kern", "Wochen 7-10 · Von Baseline bis moderne Trainingspraxis")
    render_card_grid([
        card("🧠", "NN Basics", "Perzeptron, MLP, Backprop.", ["1 Woche"], ["amber"]),
        card("🔲", "CNN", "Feature-Hierarchien in Bildern.", ["1 Woche"], ["amber"]),
        card("🎯", "Training", "Loss, Optimizer, Schedules.", ["1 Woche"], ["amber"]),
        card("🏛️", "Moderne Architekturen", "ResNet, EfficientNet, U-Net.", ["4 Tage"], ["amber"]),
        card("🧪", "Self-Supervised Learning", "SimCLR, MoCo, DINO, MAE.", ["4 Tage"], ["amber"]),
        card("🎬", "Video Understanding", "Temporal Modeling, Video Transformer.", ["4 Tage"], ["amber"]),
    ], cols=3)

    divider()

    section_header("Phase 4 — Frontier Themen", "Wochen 11-14 · State of the Art")
    render_card_grid([
        card("⚡", "Transformer & ViT", "Attention fuer Vision.", ["4 Tage"], ["pink"]),
        card("👁️‍🗨️", "VLM", "CLIP, BLIP-2, LLaVA.", ["3 Tage"], ["pink"]),
        card("🌊", "Diffusion", "DDPM, Stable Diffusion.", ["4 Tage"], ["pink"]),
        card("🌐", "Multimodal & LLMs", "Bild-Text-Video Systeme.", ["3 Tage"], ["pink"]),
        card("🧊", "3D Computer Vision", "Epipolar, SfM/SLAM, NeRF.", ["5 Tage"], ["pink"]),
        card("🛰️", "RAG + Multimodal Agents", "Grounding, Tool-Use, Guardrails.", ["4 Tage"], ["pink"]),
    ], cols=3)

    divider()

    section_header("Phase 5 — Produktreife", "Wochen 15-16 · Evaluation und Deployment")
    render_card_grid([
        card("🛡️", "Evaluation & Robustness", "Calibration, OOD, Bias/Fairness.", ["3 Tage"], ["green"]),
        card("🗜️", "Model Compression", "Quantization, Pruning, Distillation.", ["3 Tage"], ["green"]),
        card("🚀", "Deployment & MLOps", "ONNX, TensorRT, FastAPI, Monitoring.", ["4 Tage"], ["green"]),
        card("💻", "Praxisprojekte", "End-to-End Umsetzung mit Ergebnissen.", ["laufend"], ["green"]),
    ], cols=2)

    divider()

    section_header("Empfohlene Spezialisierungspfade")
    render_card_grid([
        card("📹", "Video & Tracking", "Objekterkennung, MOT, Video Understanding.", ["advanced"], ["pink"]),
        card("🧠", "Foundation Models", "SSL, Transformer, VLM, RAG Agents.", ["advanced"], ["pink"]),
        card("🧭", "3D & Robotics", "Kamera-Geometrie, Pose, SfM/SLAM.", ["advanced"], ["pink"]),
    ], cols=3)

    divider()

    info_box(
        "Arbeite pro Woche mit einem festen Rhythmus: Theorie -> Lab -> Checkpoint -> Mini-Projekt. "
        "So baust du nachhaltige Kompetenz statt nur kurzfristiges Wissen auf.",
        kind="tip",
    )
