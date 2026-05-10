"""Self-Supervised Learning fuer Vision."""
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


def _contrastive_score(pos_sim: float, neg_sim: float, temperature: float) -> float:
    pos = np.exp(pos_sim / temperature)
    neg = np.exp(neg_sim / temperature)
    return float(-np.log(pos / (pos + neg + 1e-9)))


def render():
    hero(
        eyebrow="Deep Learning · SSL",
        title="Self-Supervised Learning",
        sub="SimCLR, MoCo, DINO und MAE fuer starke Repräsentationen ohne teure Labels.",
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
        key_concept("🧲", "Embedding Space", "Aehnliche Inhalte liegen nah beieinander.")
        key_concept("💸", "Label-Effizienz", "Weniger Annotation bei vergleichbarer Downstream-Qualitaet.")

    with tabs[1]:
        section_header("Die vier Klassiker")
        st.markdown(
            """
| Methode | Kernidee | Staerke |
|---|---|---|
| SimCLR | Contrastive Loss mit grossen Batches | Einfach und stark |
| MoCo | Momentum-Encoder + Queue | Stabil bei kleineren Batches |
| DINO | Teacher-Student ohne Labels | Sehr gute Features fuer ViT |
| MAE | Masked Image Modeling | Effizientes Pretraining fuer Vision Transformer |
            """
        )
        info_box("DINO/MAE sind besonders relevant fuer moderne ViT-basierte Pipelines.", kind="success")

    with tabs[2]:
        lab_header("Contrastive Verlust", "Experimentiere mit Similarity und Temperatur.")
        pos_sim = st.slider("Positive Similarity", 0.1, 1.0, 0.8, 0.01)
        neg_sim = st.slider("Negative Similarity", -1.0, 0.9, 0.2, 0.01)
        temperature = st.slider("Temperatur tau", 0.03, 1.0, 0.1, 0.01)
        loss = _contrastive_score(pos_sim, neg_sim, temperature)
        st.metric("InfoNCE Loss", f"{loss:.4f}")
        info_box("Niedrige Temperatur erzwingt staerkere Separation im Embedding-Space.", kind="tip")

    with tabs[3]:
        lab_header("Label-Budget Simulation", "Wie stark profitiert SSL bei wenig Labels?")
        budget = np.array([1, 5, 10, 25, 50, 100], dtype=float)
        sup = np.array([32, 45, 52, 61, 67, 72], dtype=float)
        ssl = np.array([44, 57, 64, 71, 75, 79], dtype=float)
        gain_scale = st.slider("SSL Gain Skalierung", 0.5, 1.6, 1.0, 0.05)
        ssl_scaled = np.clip(sup + (ssl - sup) * gain_scale, 0, 100)

        fig = px.line(
            x=budget,
            y={"Supervised": sup, "SSL + Fine-Tuning": ssl_scaled},
            labels={"x": "Label-Anteil (%)", "value": "Top-1 Accuracy"},
            markers=True,
            template="plotly_dark",
            title="Accuracy vs Label-Budget",
        )
        fig.update_layout(height=420)
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
            question="Wofuer ist SSL besonders nuetzlich?",
            options=[
                "Wenn Labels knapp sind",
                "Nur fuer Textdaten",
                "Nur fuer GAN-Training",
                "Ausschliesslich fuer Segmentation mit Vollannotation",
            ],
            correct_option="Wenn Labels knapp sind",
            checklist=[
                "Ich kann SimCLR, MoCo, DINO, MAE grob unterscheiden.",
                "Ich verstehe den Zusammenhang zwischen Augmentations und Contrastive Loss.",
                "Ich kann erklaeren, warum SSL Label-Effizienz verbessert.",
            ],
            capstone_prompt="Definiere ein SSL-Pretraining fuer ein kleines Industriebild-Dataset "
            "und beschreibe, wie du den Fine-Tuning Gewinn messen wuerdest.",
        )

    with tabs[6]:
        section_header("Lernvideos")
        video_embed("sS6eMQp8R4A", "SimCLR", "Contrastive Learning und Augmentations.")
        divider()
        video_embed("h3ij3F3c9lE", "DINO", "Self-distillation fuer starke ViT Features.")
        divider()
        video_embed("YicbFdNTTyQ", "MAE", "Masked Autoencoders fuer Vision Transformer.")
