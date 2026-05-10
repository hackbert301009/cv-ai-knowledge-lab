"""Evaluation und Robustness fuer CV Modelle."""
import numpy as np
import plotly.express as px
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


def _ece(conf: np.ndarray, corr: np.ndarray, bins: int) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(bins):
        mask = (conf >= edges[i]) & (conf < edges[i + 1])
        if np.sum(mask) == 0:
            continue
        acc_bin = np.mean(corr[mask])
        conf_bin = np.mean(conf[mask])
        ece += np.sum(mask) / n * abs(acc_bin - conf_bin)
    return float(ece)


def render():
    hero(
        eyebrow="Praxis · Evaluation",
        title="Evaluation & Robustness",
        sub="Calibration, OOD, Domain Shift, Error Analysis und Bias/Fairness fuer Vision.",
    )

    tabs = st.tabs(
        [
            "📏 Kernmetriken",
            "🛡 Robustness",
            "⚖ Bias & Fairness",
            "🧪 Calibration Lab",
            "💻 Code",
            "✅ Checkliste",
            "✅ Checkpoint",
            "🎬 Videos",
        ]
    )

    with tabs[0]:
        section_header("Mehr als Accuracy")
        st.markdown(
            """
- Klassifikation: Accuracy, F1, AUROC, Calibration
- Detection: mAP, AP50/AP75, Precision/Recall
- Segmentation: mIoU, Dice
- Tracking: MOTA, IDF1, HOTA
            """
        )
        key_concept("🎯", "Calibration", "Wie gut Wahrscheinlichkeiten mit realer Trefferquote uebereinstimmen.")
        key_concept("🚨", "OOD Detection", "Erkennen, wenn Eingaben ausserhalb des Trainingsbereichs liegen.")
        key_concept("🔍", "Error Taxonomy", "Fehler nach Typen aufteilen statt nur Gesamtmetrik ansehen.")

    with tabs[1]:
        section_header("Robustness-Risiken")
        st.markdown(
            """
| Risiko | Beispiel | Gegenmassnahme |
|---|---|---|
| Domain Shift | Tag -> Nachtkamera | Domain Augmentation, Adaptation |
| Corruptions | Blur, Noise, Compression | Robust Training, Test-Time Aug |
| Adversarial | Kleine Perturbation | Defensive Distillation, Detection |
| Data Drift | Kamera/Objektmix aendert sich | Monitoring + regelmaessige Re-Trainings |
            """
        )
        info_box("Plane immer ein Shadow Deployment vor Full Rollout.", kind="warn")

    with tabs[2]:
        section_header("Bias und Fairness in Vision")
        st.markdown(
            """
Bias entsteht oft aus unausgewogenen Daten oder Labels.
Bewerte Metriken gruppenweise (z. B. Licht, Hauttonton, Kamerawinkel, Region).
            """
        )
        st.markdown(
            """
**Empfohlene Praxis**
1. Dataset Cards + klare Datendokumentation
2. Slice-based Evaluation fuer kritische Subgruppen
3. Human-in-the-loop bei sicherheitskritischen Entscheidungen
            """
        )

    with tabs[3]:
        lab_header("Calibration Simulation", "Berechne ECE fuer verschiedene Konfidenzprofile.")
        n = st.slider("Samples", 200, 5000, 1200, 100)
        sharpness = st.slider("Confidence Sharpness", 0.4, 3.0, 1.4, 0.1)
        bins = st.slider("ECE Bins", 5, 30, 15, 1)
        rng = np.random.default_rng(17)
        base = rng.beta(sharpness, 1.5, size=n)
        noise = rng.normal(0, 0.12, size=n)
        correctness = (base + noise > 0.52).astype(float)
        conf = np.clip(base, 0.01, 0.99)
        ece = _ece(conf, correctness, bins)
        st.metric("Expected Calibration Error", f"{ece:.4f}")

        df = {"confidence": conf, "correct": correctness}
        fig = px.histogram(df, x="confidence", color="correct", nbins=bins, barmode="overlay", template="plotly_dark")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        section_header("Praxis-Code")
        st.code(
            """# Temperature Scaling (klassisch)
import torch
import torch.nn.functional as F

temperature = torch.nn.Parameter(torch.ones(1) * 1.2)
logits_scaled = logits / temperature
loss = F.cross_entropy(logits_scaled, labels)
            """,
            language="python",
        )
        divider()
        st.code(
            """# Slice-based Evaluation (Pseudo)
for slice_name, mask in slices.items():
    y_true_s = y_true[mask]
    y_pred_s = y_pred[mask]
    print(slice_name, f1_score(y_true_s, y_pred_s))
            """,
            language="python",
        )

    with tabs[5]:
        section_header("Pre-Release Checkliste")
        st.markdown(
            """
- [ ] Hauptmetrik + Nebenmetriken dokumentiert
- [ ] Calibration auf Val/Test geprueft
- [ ] OOD-Strategie definiert
- [ ] Slice-Evaluation fuer sensible Gruppen vorhanden
- [ ] Monitoring-Plan fuer Drift und Ausfaelle aktiv
            """
        )
        info_box("Ohne Monitoring ist kein Modell wirklich produktionsreif.", kind="tip")

    with tabs[6]:
        render_quiz_checkpoint(
            key_prefix="evaluation_robustness",
            question="Welche Metrik misst, ob Confidence und reale Trefferquote zusammenpassen?",
            options=["ECE", "IoU", "PSNR", "BLEU"],
            correct_option="ECE",
            checklist=[
                "Ich kann Calibration und OOD Detection unterscheiden.",
                "Ich habe Slice-based Evaluation fuer kritische Gruppen geplant.",
                "Ich kenne die wichtigsten Monitoring-Signale fuer Drift.",
            ],
            capstone_prompt="Beschreibe einen Release-Plan mit Shadow Deployment, "
            "Slice-Evaluation und Drift-Monitoring fuer ein Visionmodell.",
        )

    with tabs[7]:
        section_header("Lernvideos")
        video_embed("b7M1rVQ8x6Y", "Model Calibration", "Warum Confidence oft zu optimistisch ist.")
        divider()
        video_embed("16x2bYhQj0Q", "OOD Detection", "Methoden fuer Unknown Inputs in Produktion.")
        divider()
        video_embed("xY5LqJv4fSE", "Bias in CV", "Fairness-Risiken bei realen Vision Systemen.")
