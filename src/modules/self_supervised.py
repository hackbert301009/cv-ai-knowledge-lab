"""Self-Supervised Learning für Vision."""
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
    video_embed, video_search,
)


def _contrastive_score(pos_sim: float, neg_sim: float, temperature: float) -> float:
    pos = np.exp(pos_sim / temperature)
    neg = np.exp(neg_sim / temperature)
    return float(-np.log(pos / (pos + neg + 1e-9)))


def render():
    hero(
        eyebrow="Deep Learning · SSL",
        title="Self-Supervised Learning",
        sub="SimCLR, MoCo, DINO und MAE für starke Repräsentationen ohne teure Labels.",
    )

    tabs = st.tabs(
        [
            "🧠 Warum SSL",
            "🏗 Methoden",
            "🧪 Contrastive Lab",
            "📊 Label-Budget",
            "💻 Code",
            "✅ Checkpoint",
            "🎬 Videos",
        ]
    )

    with tabs[0]:
        section_header("Warum Self-Supervised Learning?")
        st.markdown(
            """
SSL lernt aus unlabeled Daten durch ein Pretext-Ziel.
Danach wird mit wenigen Labels feinjustiert (Fine-Tuning).

**Typischer Vorteil:** Bessere Generalisierung bei kleinem Label-Budget.
            """
        )
        key_concept("🪞", "Augmentations", "Verschiedene Ansichten desselben Bildes bilden positives Paar.")
        key_concept("🧲", "Embedding Space", "Ähnliche Inhalte liegen nah beieinander.")
        key_concept("💸", "Label-Effizienz", "Weniger Annotation bei vergleichbarer Downstream-Qualität.")

    with tabs[1]:
        section_header("Die vier Klassiker")
        st.markdown(
            """
| Methode | Kernidee | Stärke |
|---|---|---|
| SimCLR | Contrastive Loss mit großen Batches | Einfach und stark |
| MoCo | Momentum-Encoder + Queue | Stabil bei kleineren Batches |
| DINO | Teacher-Student ohne Labels | Sehr gute Features für ViT |
| MAE | Masked Image Modeling | Effizientes Pretraining für Vision Transformer |
            """
        )
        info_box("DINO/MAE sind besonders relevant für moderne ViT-basierte Pipelines.", kind="success")

    with tabs[2]:
        lab_header("Contrastive Verlust", "Experimentiere mit Similarity und Temperatur.")
        pos_sim = st.slider("Positive Similarity", 0.1, 1.0, 0.8, 0.01)
        neg_sim = st.slider("Negative Similarity", -1.0, 0.9, 0.2, 0.01)
        temperature = st.slider("Temperatur tau", 0.03, 1.0, 0.1, 0.01)
        loss = _contrastive_score(pos_sim, neg_sim, temperature)
        st.metric("InfoNCE Loss", f"{loss:.4f}")
        info_box("Niedrige Temperatur erzwingt stärkere Separation im Embedding-Space.", kind="tip")

    with tabs[3]:
        lab_header("Label-Budget Simulation", "Wie stark profitiert SSL bei wenig Labels?")
        budget = np.array([1, 5, 10, 25, 50, 100], dtype=float)
        sup = np.array([32, 45, 52, 61, 67, 72], dtype=float)
        ssl = np.array([44, 57, 64, 71, 75, 79], dtype=float)
        gain_scale = st.slider("SSL Gain Skalierung", 0.5, 1.6, 1.0, 0.05)
        ssl_scaled = np.clip(sup + (ssl - sup) * gain_scale, 0, 100)

        fig = go.Figure()
        fig.add_scatter(x=budget, y=sup, mode="lines+markers", name="Supervised")
        fig.add_scatter(x=budget, y=ssl_scaled, mode="lines+markers", name="SSL + Fine-Tuning")
        fig.update_layout(
            template="plotly_dark",
            title="Accuracy vs Label-Budget",
            xaxis_title="Label-Anteil (%)",
            yaxis_title="Top-1 Accuracy",
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        section_header("Praxis-Code")
        st.code(
            """# SimCLR-artiger Forward Pass (vereinfacht)
import torch
import torch.nn.functional as F

z1 = F.normalize(encoder(view1), dim=1)
z2 = F.normalize(encoder(view2), dim=1)
logits = z1 @ z2.T / 0.1
labels = torch.arange(logits.shape[0], device=logits.device)
loss = F.cross_entropy(logits, labels)
            """,
            language="python",
        )
        divider()
        st.code(
            """# MAE-artige Idee (Pseudo)
tokens = patch_embed(images)
visible, masked_idx = random_mask(tokens, ratio=0.75)
latent = encoder(visible)
recon = decoder(latent, masked_idx)
loss = ((recon - target_patches(masked_idx)) ** 2).mean()
            """,
            language="python",
        )

    with tabs[5]:
        render_quiz_checkpoint(
            key_prefix="self_supervised",
            question="Wofür ist SSL besonders nützlich?",
            options=[
                "Wenn Labels knapp sind",
                "Nur für Textdaten",
                "Nur für GAN-Training",
                "Ausschließlich für Segmentation mit Vollannotation",
            ],
            correct_option="Wenn Labels knapp sind",
            checklist=[
                "Ich kann SimCLR, MoCo, DINO, MAE grob unterscheiden.",
                "Ich verstehe den Zusammenhang zwischen Augmentations und Contrastive Loss.",
                "Ich kann erklären, warum SSL Label-Effizienz verbessert.",
            ],
            capstone_prompt="Definiere ein SSL-Pretraining für ein kleines Industriebild-Dataset "
            "und beschreibe, wie du den Fine-Tuning Gewinn messen würdest.",
        )

    with tabs[6]:
        section_header("Lernvideos")
        video_search("SimCLR contrastive learning explained", "SimCLR", "Contrastive Learning und Augmentations.")
        divider()
        video_search("DINO self-supervised vision transformer explained", "DINO", "Self-Distillation für starke ViT-Features.")
        divider()
        video_search("Masked Autoencoders MAE vision explained", "MAE", "Masked Autoencoders für Vision Transformer.")
