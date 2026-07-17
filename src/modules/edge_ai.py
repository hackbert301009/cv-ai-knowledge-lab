"""Edge & Embedded CV — On-Device-Inferenz auf Jetson, Mobile, TPU & Co."""
import streamlit as st

from src.components import (
    hero, section_header, divider, info_box, lab_header, key_concept, step_list,
    video_embed, video_search, render_learning_block, render_quiz_checkpoint,
)

# (Modell: Parameter in Mio., GMACs)
_MODELS = {
    "MobileNetV3-Large": (5.4, 0.22),
    "EfficientNet-B0":   (5.3, 0.39),
    "YOLO11n":           (2.6, 3.2),
    "ResNet-50":         (25.6, 4.1),
    "ViT-B/16":          (86.0, 17.6),
}
# (Gerät: INT8-TOPS, unterstützte Präzisionen)
_DEVICES = {
    "Jetson Orin Nano":      (40.0, {"INT8", "FP16"}),
    "Google Coral Edge TPU": (4.0,  {"INT8"}),
    "iPhone Neural Engine":  (35.0, {"INT8", "FP16"}),
    "Raspberry Pi 5 (CPU)":  (0.05, {"INT8", "FP16", "FP32"}),
}
_BYTES = {"FP32": 4, "FP16": 2, "INT8": 1}
_PREC_FACTOR = {"INT8": 1.0, "FP16": 0.5, "FP32": 0.25}
_UTIL = 0.35  # realistische Auslastung statt Peak-TOPS


def render():
    hero(
        eyebrow="Praxis · Edge & Embedded CV",
        title="Edge & Embedded CV",
        sub="Modelle laufen nicht immer in der Cloud. Auf Jetson, Smartphone, TPU oder Mikrocontroller "
            "zählen Latenz, Speicher und Watt — hier lernst du, wie On-Device-Inferenz funktioniert."
    )

    tabs = st.tabs([
        "🎯 Warum Edge?", "🔌 Hardware", "🛠️ Runtimes", "⚙️ Optimierung",
        "🧪 Latenz-Rechner", "🎬 Lernvideos",
    ])

    with tabs[0]:
        section_header("Warum überhaupt on-device?")
        st.markdown(r"""
Cloud-Inferenz ist bequem — aber Edge-Deployment hat handfeste Vorteile:

- **Latenz**: keine Netzwerk-Roundtrips, oft < 30 ms statt 200 ms+.
- **Datenschutz**: Rohdaten verlassen das Gerät nicht (Kamera, Mikrofon).
- **Offline-Fähigkeit**: funktioniert ohne Verbindung (Auto, Drohne, Feld).
- **Kosten**: keine laufenden Cloud-GPU-Kosten pro Anfrage.
- **Skalierung**: Rechenlast verteilt sich auf die Endgeräte.

Der Preis: **begrenzte Ressourcen** — Speicher, Rechenleistung, Energie und Wärme.
        """)
        key_concept("⏱️", "Latenz vs. Durchsatz",
                    "Edge optimiert meist Latenz (eine Eingabe schnell), Cloud oft Durchsatz (viele parallel).")
        key_concept("🔋", "Power-Budget",
                    "Ein Smartphone/Drohne hat wenige Watt — Effizienz (Inferenzen pro Joule) ist entscheidend.")

    with tabs[1]:
        section_header("Edge-Hardware-Landschaft")
        st.markdown(r"""
| Gerät | Beschleuniger | Typisch für |
|---|---|---|
| **NVIDIA Jetson** (Orin/Nano) | GPU + DLA | Robotik, Kameras, Prototypen |
| **Google Coral** | Edge TPU (nur INT8) | günstige, effiziente Klassifikation |
| **Smartphone** | NPU / Neural Engine / Hexagon | Apps, AR, Kamera-Features |
| **Hailo-8 / Axelera** | dedizierter NPU | Industrie-Kameras, hohe Effizienz |
| **Raspberry Pi** | CPU (+ optional Accelerator) | Bastel-/Lernprojekte |
| **Mikrocontroller** (TinyML) | Cortex-M / MCU | Sensoren, Wake-Word, wenige KB RAM |
        """)
        info_box("Merke: Der **Edge TPU** unterstützt ausschließlich INT8. Ein FP16/FP32-Modell "
                 "muss vorher voll quantisiert werden.", kind="warn")

    with tabs[2]:
        section_header("Runtimes & Formate")
        st.markdown(r"""
Ein trainiertes Modell (PyTorch/TensorFlow) wird für die Ziel-Hardware **exportiert und optimiert**:

- **ONNX** — portables Zwischenformat, breit unterstützt.
- **ONNX Runtime** — plattformübergreifende Inferenz (CPU/GPU/NPU).
- **TensorRT** — NVIDIA-GPU/Jetson, maximaler Durchsatz (Layer Fusion, INT8).
- **TFLite** — Android, Mikrocontroller, Coral.
- **Core ML** — Apple-Geräte (Neural Engine).
- **ncnn / MNN** — mobile, CPU/GPU-optimiert (weit verbreitet in Apps).
        """)
        step_list([
            ("Trainieren", "In PyTorch/TF wie gewohnt."),
            ("Exportieren", "Nach ONNX (oder direkt TFLite/Core ML)."),
            ("Quantisieren", "INT8 mit Kalibrierungsdaten (PTQ) oder QAT."),
            ("Kompilieren", "Für Ziel-Runtime (TensorRT-Engine, TFLite-Delegate …)."),
            ("Messen", "Latenz, Speicher, Genauigkeit auf dem echten Gerät prüfen."),
        ])

    with tabs[3]:
        section_header("Optimierungstechniken")
        st.markdown(r"""
- **Quantisierung** (FP32 → INT8): ~4× kleiner, 2–4× schneller. Siehe Modul *Model Compression*.
- **Pruning**: unwichtige Gewichte/Filter entfernen.
- **Knowledge Distillation**: kleines Student-Modell vom großen Teacher lernen.
- **Effiziente Architekturen**: MobileNet, EfficientNet-Lite, YOLO-nano.
- **Operator-Fusion**: Conv+BN+ReLU zu einem Kernel verschmelzen.
- **Input-Auflösung senken**: oft der billigste Latenz-Hebel (quadratischer Effekt).
        """)
        info_box("Regel: Erst **INT8-Quantisierung + kleineres Modell**, dann Hardware-spezifische Kernel. "
                 "Miss nach jedem Schritt Genauigkeit **und** Latenz.", kind="tip")

    # ── Tab 4: Latenz-Rechner ────────────────────────────────────────────────
    with tabs[4]:
        lab_header("Latenz- & Größen-Schätzer", "Wie schnell läuft ein Modell auf welchem Gerät? (grobe Schätzung)")
        c1, c2, c3 = st.columns(3)
        model = c1.selectbox("Modell", list(_MODELS.keys()), index=0)
        device = c2.selectbox("Gerät", list(_DEVICES.keys()), index=0)
        precision = c3.selectbox("Präzision", ["INT8", "FP16", "FP32"], index=0)

        params_m, gmacs = _MODELS[model]
        tops_int8, supported = _DEVICES[device]

        size_mb = params_m * _BYTES[precision]
        st.metric("Modellgröße (Gewichte)", f"{size_mb:.1f} MB")

        if precision not in supported:
            info_box(f"**{device}** unterstützt **{precision}** nicht "
                     f"(nur {', '.join(sorted(supported))}). Modell zuerst passend konvertieren.", kind="warn")
        else:
            eff_tops = tops_int8 * _PREC_FACTOR[precision] * _UTIL
            ops = gmacs * 2e9                      # 1 MAC = 2 Ops
            latency_ms = ops / (eff_tops * 1e12) * 1000
            fps = 1000.0 / latency_ms if latency_ms > 0 else 0
            m1, m2, m3 = st.columns(3)
            m1.metric("Geschätzte Latenz", f"{latency_ms:.1f} ms")
            m2.metric("Geschätzte FPS", f"{fps:.0f}")
            m3.metric("Effektive Rechenleistung", f"{eff_tops:.1f} TOPS")
            if fps >= 30:
                info_box("✅ Echtzeit-fähig (≥ 30 FPS) für dieses Setup.", kind="success")
            elif fps >= 10:
                info_box("🟡 Interaktiv, aber unter Echtzeit — kleineres Modell/INT8 erwägen.", kind="warn")
            else:
                info_box("🔴 Zu langsam — Modell verkleinern, quantisieren oder Auflösung senken.", kind="warn")

        st.caption("Grobe Schätzung: reale Latenz hängt von Speicherbandbreite, Kernel-Qualität, "
                   f"I/O und Auslastung ab (hier ~{int(_UTIL*100)} % angenommen). Immer auf dem echten Gerät messen.")

    with tabs[5]:
        section_header("Lernvideos")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**TinyML & Edge AI — Überblick**")
            video_search("TinyML Edge AI overview", "Edge AI & TinyML — Überblick", "")
        with col2:
            st.markdown("**Modelle für Mobile quantisieren**")
            video_search("neural network quantization mobile explained", "Quantisierung für Mobile", "")

    divider()
    render_learning_block(
        key_prefix="edge_ai",
        progression=[
            ("🟢", "Guided", "Nenne drei Gründe, ein Modell on-device statt in der Cloud laufen zu lassen.", "Guided", "green"),
            ("🟠", "Challenge", "Wähle für eine batteriebetriebene Wildkamera Modell + Präzision + Gerät.", "Challenge", "amber"),
            ("🔴", "Debug", "Ein FP32-Modell läuft nicht auf dem Coral TPU — warum, und was tust du?", "Debug", "pink"),
            ("🧩", "Mini-Projekt", "Exportiere ein Klassifikationsmodell nach ONNX und quantisiere es INT8.", "Projekt", "blue"),
        ],
        mcq_question="Welche Optimierung bringt typischerweise ~4× kleinere Modelle und 2–4× Speedup?",
        mcq_options=["INT8-Quantisierung", "Größere Eingabeauflösung", "Mehr Layer", "Höhere Batchgröße"],
        mcq_correct_option="INT8-Quantisierung",
        open_question="Warum ist 'Inferenzen pro Joule' auf einem Smartphone oft wichtiger als reine FPS?",
        cheat_sheet=[
            "Edge-Vorteile: Latenz, Datenschutz, Offline, Kosten.",
            "Coral Edge TPU = nur INT8.",
            "Pipeline: Train → ONNX → Quantisieren → Kompilieren → Messen.",
            "Auflösung senken ist oft der billigste Latenz-Hebel.",
        ],
        key_takeaways=[
            "Edge tauscht Genauigkeit/Modellgröße gegen Latenz, Speicher und Energie.",
            "Peak-TOPS ≠ reale Leistung — immer auf dem Zielgerät messen.",
        ],
        common_errors=[
            "FP32-Modell auf INT8-only-Hardware deployen wollen.",
            "Nur FPS optimieren, Energie/Wärme ignorieren.",
            "Genauigkeit nach Quantisierung nicht prüfen.",
        ],
    )
    render_quiz_checkpoint(
        key_prefix="edge_ai",
        module_id="edge_ai",
        question="Der Google Coral Edge TPU unterstützt welche Präzision?",
        options=["Ausschließlich INT8", "Nur FP32", "FP16 und FP32", "Beliebige"],
        correct_option="Ausschließlich INT8",
        checklist=[
            "Ich kenne die Vorteile von Edge-Inferenz.",
            "Ich kann Modell, Präzision und Gerät aufeinander abstimmen.",
            "Ich kenne die Export-Pipeline (ONNX/TFLite/Core ML).",
        ],
    )
