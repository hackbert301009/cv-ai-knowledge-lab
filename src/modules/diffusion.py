"""Diffusion Models — DDPM, Stable Diffusion, Flow Matching."""
import streamlit as st
from src.components import hero, section_header, divider, info_box


def render():
    hero(
        eyebrow="State-of-the-Art · Modul 17",
        title="Diffusion Models",
        sub="Wie aus reinem Rauschen Bilder werden. Die mathematische Eleganz hinter "
            "Stable Diffusion, DALL·E 3, Midjourney und Sora."
    )

    tabs = st.tabs(["🌊 Idee", "📐 DDPM", "🎯 Sampling", "🖼️ Stable Diffusion", "🔄 Flow Matching", "🎬 Video Diffusion"])

    with tabs[0]:
        section_header("Die zentrale Idee")
        st.markdown(r"""
**Forward Process**: Nimm ein Bild und addiere Schritt für Schritt Gauß-Rauschen, bis nur noch Rauschen übrig ist.

**Reverse Process**: Lerne ein neuronales Netz, das den **Schritt rückwärts** macht — vom Rauschen zurück zum Bild.

> Wenn du Schritt für Schritt rauschen kannst, kannst du auch Schritt für Schritt **entrauschen**.
> Das ist die ganze Idee.

#### Warum funktioniert das so gut?
- **Stabiles Training** (kein Mode Collapse wie bei GANs)
- **Hohe Diversität** der Outputs
- **Conditioning ist einfach** — Text, Klassen, andere Bilder
- **Skalierbar** — größere Modelle = bessere Bilder
        """)

    with tabs[1]:
        section_header("DDPM — Denoising Diffusion Probabilistic Models (Ho et al., 2020)")
        st.markdown(r"""
#### Forward Process
$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t \mathbf{I})$$

Mit dem **Reparametrisierungstrick** kann man direkt von $\mathbf{x}_0$ zu jedem $\mathbf{x}_t$ springen:
$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$$

#### Reverse Process
Das Netz $\epsilon_\theta(\mathbf{x}_t, t)$ wird trainiert, das Rauschen $\boldsymbol{\epsilon}$ vorherzusagen.

#### Trainings-Loss
$$\mathcal{L} = \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}, t} \left[ \|\boldsymbol{\epsilon} - \epsilon_\theta(\mathbf{x}_t, t)\|^2 \right]$$

Das ist's. Eine MSE zwischen wahrem und vorhergesagtem Rauschen.
        """)
        info_box(
            "Diffusion's Loss ist im Kern erstaunlich einfach: 'Sag mir, wie viel Rauschen drin ist.' "
            "Die ganze Magie steckt in der iterativen Anwendung.",
            kind="tip",
        )

    with tabs[2]:
        section_header("Sampling — vom Rauschen zum Bild")
        st.markdown(r"""
Starte bei $\mathbf{x}_T \sim \mathcal{N}(0, I)$ und iteriere $T$ Schritte rückwärts:

$$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left( \mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(\mathbf{x}_t, t) \right) + \sigma_t \mathbf{z}$$

#### Beschleunigte Sampler
- **DDIM** (Song et al. 2020): deterministisch, 50 Schritte statt 1000
- **DPM-Solver** (Lu et al. 2022): nur 10–20 Schritte, hohe Qualität
- **Consistency Models** (Song et al. 2023): **ein einziger Schritt** möglich
- **LCM, SDXL Turbo, SD3** (2023/24): wenige Schritte, fast Echtzeit

#### Classifier-Free Guidance (CFG)
$$\hat{\epsilon}_\theta(\mathbf{x}_t, c) = \epsilon_\theta(\mathbf{x}_t, \emptyset) + w \cdot (\epsilon_\theta(\mathbf{x}_t, c) - \epsilon_\theta(\mathbf{x}_t, \emptyset))$$

Mit Guidance Scale $w > 1$ wird die Generierung **stärker an die Bedingung angepasst**.
Höhere CFG = mehr Treue zum Prompt, aber weniger Diversität und potentielle Artefakte.
        """)

    with tabs[3]:
        section_header("Stable Diffusion — Latent Diffusion (2022)")
        st.markdown(r"""
**Stable Diffusion** (Rombach et al., CompVis/Stability AI) hatte einen genialen Trick:
**arbeite nicht im Pixelraum, sondern im latenten Raum eines VAE.**

#### Pipeline
1. **VAE-Encoder**: 512×512×3 → 64×64×4 (Faktor 64 weniger Daten)
2. **U-Net im Latent**: Diffusion auf 64×64×4
3. **VAE-Decoder**: 64×64×4 → 512×512×3

#### Conditioning
- **Text-Encoder** (CLIP) gibt Text-Embeddings
- Im U-Net **Cross-Attention** zwischen Bild-Features und Text-Embeddings
- Dadurch kann das Netz auf den Prompt "achten"

#### SDXL, SD3 Updates
- **SDXL**: zwei Text-Encoder, größeres U-Net, höhere Qualität
- **SD3**: **MM-DiT** statt U-Net (Transformer-basiert), Flow Matching statt DDPM
        """)

    with tabs[4]:
        section_header("Flow Matching — die neue Generation (2022/23)")
        st.markdown(r"""
**Flow Matching** (Lipman et al.) ist eine elegantere Formulierung:
Statt Rauschen schrittweise wegzunehmen, lerne direkt das **Vektorfeld**, das die Rausch-Verteilung
zur Bild-Verteilung transportiert.

$$\frac{d\mathbf{x}_t}{dt} = v_\theta(\mathbf{x}_t, t)$$

Das Netz lernt **Geschwindigkeiten**, nicht Rauschen.

#### Vorteile
- Glattere, geradlinigere Trajektorien
- Weniger Sampling-Schritte nötig
- Mathematisch sauberer (Continuous Normalizing Flows)

#### Wo es heute läuft
- **Stable Diffusion 3** und **FLUX.1** nutzen Flow Matching
- **Sora** wahrscheinlich auch (basiert auf DiT mit Flow-Matching)
        """)

    with tabs[5]:
        section_header("Video Diffusion — die nächste Grenze")
        st.markdown(r"""
**Video** ist Bild + Zeit — eine zusätzliche Dimension.

#### Herausforderungen
- **Zeitliche Konsistenz**: Objekte dürfen nicht "flackern"
- **Speicher**: 24 fps × 5s = 120 Frames, riesige Tensoren
- **Bewegungsphysik**: Modelle müssen lernen, wie Dinge sich bewegen

#### Architekturen
- **VideoLDM, AnimateDiff**: Bild-Diffusion + temporale Layer
- **Sora** (OpenAI, 2024): Latent-Diffusion-Transformer auf Video-Patches ("Spacetime-Patches")
- **Veo 2** (Google), **Gen-3** (Runway), **Kling**, **Hunyuan**: aktuelle Konkurrenten
- **Open-Sora**: Open-Source-Alternative

#### Aktueller Stand 2026
- ~10s Videos in hoher Qualität sind Standard
- Noch nicht perfekt für Physik (Wasser, Hände, Komplexität)
- Audio-Sync und längere Geschichten = nächste Frontier
        """)
