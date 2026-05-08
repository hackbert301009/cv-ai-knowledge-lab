"""Diffusion Models — DDPM, Stable Diffusion, Flow Matching."""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.components import (
    hero, section_header, divider, info_box,
    video_embed, lab_header, key_concept, step_list, card, render_card_grid, render_learning_block,
)


def render():
    hero(
        eyebrow="State-of-the-Art · Modul 17",
        title="Diffusion Models",
        sub="Wie aus reinem Rauschen Bilder werden. Die mathematische Eleganz hinter "
            "Stable Diffusion, DALL·E 3, Midjourney und Sora."
    )

    tabs = st.tabs([
        "🌊 Die Idee",
        "📐 DDPM Mathematik",
        "🎯 Sampling-Strategien",
        "🖼️ Stable Diffusion",
        "🔄 Flow Matching",
        "🎬 Video Diffusion",
        "🧪 Interaktives Lab",
        "🎬 Lernvideos",
        "🧭 Lernpfad & Übungen",
    ])

    # ------------------------------------------------------------------ #
    with tabs[0]:
        section_header("Die zentrale Idee",
                       "Vorwärts: Bild zu Rauschen. Rückwärts: Rauschen zu Bild.")
        st.markdown(r"""
Diffusion Modelle sind von einem physikalischen Prozess inspiriert: **Diffusion** —
das Ausbreiten von Teilchen in einem Medium bis zum Gleichgewicht.
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(r"""
#### Forward Process (Vorwärts)
Nimm ein Bild $\mathbf{x}_0$ und addiere Schritt für Schritt Gauß-Rauschen.

Nach $T$ Schritten (typisch $T=1000$) ist nur noch reines Rauschen übrig:
$$\mathbf{x}_0 \rightarrow \mathbf{x}_1 \rightarrow \mathbf{x}_2 \rightarrow \cdots \rightarrow \mathbf{x}_T \sim \mathcal{N}(0, I)$$

Dieser Prozess ist **fixiert** — kein Lernen nötig.
""")
        with col2:
            st.markdown(r"""
#### Reverse Process (Rückwärts)
Starte bei reinem Rauschen und entrausche schrittweise:
$$\mathbf{x}_T \rightarrow \mathbf{x}_{T-1} \rightarrow \cdots \rightarrow \mathbf{x}_0$$

Ein neuronales Netz **lernt** jeden Umkehrschritt.

> "Wenn du Schritt für Schritt rauschen kannst,
> kannst du auch Schritt für Schritt **entrauschen**."
""")

        divider()

        st.markdown("#### Warum funktioniert das so gut?")

        cards = [
            card("🏋️", "Stabiles Training",
                 "Kein Mode Collapse wie bei GANs. MSE-Loss. Sehr einfach zu trainieren.",
                 ["vs GAN"], ["green"]),
            card("🌈", "Hohe Diversität",
                 "Durch zufälliges Starting-Noise werden sehr verschiedene Outputs erzeugt.",
                 ["Generierung"], ["blue"]),
            card("🎯", "Einfaches Conditioning",
                 "Text, Klassen, Bilder — alles kann als Condition ins Netz gegeben werden via Cross-Attention.",
                 ["Flexibel"], ["amber"]),
            card("📈", "Skalierbarkeit",
                 "Größere Modelle = bessere Bilder. Keine Instabilitäten beim Skalieren.",
                 ["Scale"], ["pink"]),
            card("🔧", "Kontrolle",
                 "CFG, ControlNet, IP-Adapter — viele Wege, die Generierung zu steuern.",
                 ["Control"], ["blue"]),
            card("🔬", "Verstehbar",
                 "Mathematisch sauber begründet (ELBO, Score-Matching). Besser als GANs.",
                 ["Theorie"], ["green"]),
        ]
        render_card_grid(cards, cols=3)

    # ------------------------------------------------------------------ #
    with tabs[1]:
        section_header("DDPM — Denoising Diffusion Probabilistic Models",
                       "Ho et al., 2020 — das Paper, das alles gestartet hat.")
        st.markdown(r"""
#### Forward Process — genau formuliert
$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}\!\left(\mathbf{x}_t;\, \sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\, \beta_t \mathbf{I}\right)$$

$\beta_t$ ist der **Noise Schedule** — wie viel Rauschen pro Schritt addiert wird.
Typisch: $\beta_1 = 0.0001, \beta_T = 0.02$ (linear) oder cosine schedule.

#### Der Reparametrisierungs-Trick
Direkt von $\mathbf{x}_0$ zu jedem $\mathbf{x}_t$ springen — ohne alle Zwischenschritte:
$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$$

mit $\alpha_t = 1 - \beta_t$ und $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$.

**Bedeutung:** Wir können für beliebigen Zeitschritt $t$ sofort das verrauschte Bild berechnen.
Das macht Training effizient: zufälliges $t$ sampeln, Rauschen hinzufügen, Netz trainieren.

#### Was das Netz lernt
Das U-Net $\epsilon_\theta(\mathbf{x}_t, t)$ sagt vorher, welches Rauschen $\boldsymbol{\epsilon}$ dem Bild hinzugefügt wurde.

#### Trainings-Loss
$$\mathcal{L}_\text{simple} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[\, \|\boldsymbol{\epsilon} - \epsilon_\theta(\mathbf{x}_t, t)\|^2 \,\right]$$

**Das ist's.** Eine einfache MSE zwischen echtem und vorhergesagtem Rauschen.
Die ganze Magie steckt in der iterativen Anwendung beim Sampling.
        """)

        info_box(
            "**Intuition:** Das Netz lernt, ein verrauschtes Bild zu 'verstehen', "
            "indem es das Rauschen identifiziert. Beim Sampling benutzen wir das viele Male, "
            "um das Rauschen schrittweise zu entfernen.",
            kind="tip",
        )

        st.markdown("#### U-Net als Backbone")
        st.markdown(r"""
Das Denoising-Netz ist ein **U-Net** mit Attention:
- **Encoder**: Bild komprimieren (Downsampling via ResNet-Blöcke)
- **Bottleneck**: Self-Attention + Cross-Attention für Conditioning
- **Decoder**: Rekonstruieren (Upsampling mit Skip Connections vom Encoder)

Zeitschritt $t$ wird als **sinusoidales Embedding** ins Netz gegeben (wie Positional Encoding).
        """)

    # ------------------------------------------------------------------ #
    with tabs[2]:
        section_header("Sampling-Strategien", "Vom reinen Rauschen in wenigen Schritten zum Bild.")
        st.markdown(r"""
#### DDPM Sampling (original, 1000 Schritte)
$$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\!\left( \mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(\mathbf{x}_t, t) \right) + \sigma_t \mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(0,I)$$

1000 Sampling-Schritte × 1 Forward-Pass pro Schritt = sehr langsam.

#### Beschleunigte Sampler
| Sampler | Schritte | Deterministisch? | Qualität |
|---|---|---|---|
| **DDPM** | 1000 | Nein (stochastisch) | Baseline |
| **DDIM** | 50–100 | Ja | Fast genauso gut |
| **DPM-Solver++** | 10–20 | Ja | Sehr gut |
| **PNDM** | 20–50 | Ja | Gut |
| **LCM** | 4–8 | Ja | Gut |
| **SDXL-Turbo** | 1–4 | Ja | Beeindruckend |
| **Consistency Models** | 1 | Ja | Einzelner Schritt! |

#### Classifier-Free Guidance (CFG)
Das Netz wird **sowohl conditional (mit Text) als auch unconditional (ohne Text)** trainiert.
Beim Sampling kombinieren wir beide:

$$\hat{\epsilon}_\theta(\mathbf{x}_t, c) = \epsilon_\theta(\mathbf{x}_t, \emptyset) + w \cdot \big(\epsilon_\theta(\mathbf{x}_t, c) - \epsilon_\theta(\mathbf{x}_t, \emptyset)\big)$$

**Guidance Scale $w$:**
- $w=1$: Nur das conditional Modell → mehr Diversität
- $w=7.5$: Standard bei Stable Diffusion → gute Balance
- $w=20+$: Sehr prompt-treu → weniger Diversität, oft Artefakte

Diese clevere Idee hat keine extra Classifier nötig — daher "Classifier-**Free**".
        """)

    # ------------------------------------------------------------------ #
    with tabs[3]:
        section_header("Stable Diffusion — Latent Diffusion (2022)",
                       "Rombach et al., CompVis/Stability AI, CVPR 2022")
        st.markdown(r"""
**Das Problem mit Pixel-Diffusion:** Ein 512×512×3 Bild hat 786.432 Werte.
U-Net auf diesem Raum zu trainieren ist teuer.

**Stable Diffusions Lösung:** Arbeite im **latenten Raum eines VAE**.
        """)

        step_list([
            ("VAE Encoder: Bild → Latent",
             "512×512×3 → 64×64×4 — Faktor 64 weniger Werte. Der VAE encodiert nur visuell bedeutsame Information."),
            ("Diffusion im latenten Raum",
             "Das U-Net arbeitet auf 64×64×4. Viel schneller! Die Strukturen bleiben, Details werden komprimiert."),
            ("Text-Conditioning via CLIP",
             "CLIP-Text-Encoder gibt Text-Embeddings. Im U-Net: Cross-Attention zwischen Latents und Text → Prompt-Kontrolle."),
            ("VAE Decoder: Latent → Bild",
             "64×64×4 → 512×512×3. Der Decoder rekonstruiert hoch-qualitative Pixel aus den Latents."),
        ])

        st.markdown(r"""
#### SDXL (2023) — Die nächste Generation
- **Zwei Text-Encoder**: CLIP ViT-L + OpenCLIP ViT-bigG für besseres Textverständnis
- **Größeres U-Net**: 3,5B Parameter, 2048-dimensionale Latents
- **Refiner-Modell**: Zweite Diffusion-Stufe für hochauflösende Details
- **Output**: 1024×1024 native, sehr viel besser bei Text im Bild

#### Stable Diffusion 3 (2024) — MM-DiT
- **Keine U-Net mehr**: Multimodal Diffusion Transformer (MM-DiT)
- **Flow Matching** statt DDPM
- **Drei Text-Encoder**: CLIP ViT-L, OpenCLIP ViT-bigG, T5-XXL
- **Viel bessere Textgenerierung** im Bild

#### FLUX.1 (2024)
- Weiterentwicklung von SD3
- **FLUX.1 [schnell]**: 4 Schritte, sehr schnell
- **FLUX.1 [dev]**: Beste open-source Bildqualität 2024
        """)

        st.code("""
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import torch

# SDXL laden (HuggingFace Diffusers)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

# Bild generieren
result = pipe(
    prompt="A photorealistic image of a neural network visualized as glowing nodes",
    negative_prompt="blurry, low quality, distorted",
    num_inference_steps=25,
    guidance_scale=7.5,
    width=1024, height=1024,
    generator=torch.manual_seed(42),
)
image = result.images[0]
image.save("output.png")
        """, language="python")

    # ------------------------------------------------------------------ #
    with tabs[4]:
        section_header("Flow Matching — die neue Generation (2022/23)",
                       "Lipman et al. / Liu et al. — eleganter als DDPM.")
        st.markdown(r"""
**Flow Matching** ist eine elegantere Formulierung:
Statt Rauschen schrittweise wegzunehmen, lerne direkt das **Vektorfeld**,
das die Rausch-Verteilung zur Bild-Verteilung transportiert.

$$\frac{d\mathbf{x}_t}{dt} = v_\theta(\mathbf{x}_t, t), \quad t \in [0, 1]$$

Das Netz lernt **Geschwindigkeiten** (Vektoren), nicht Rauschen.

#### Vorteile gegenüber DDPM
| Aspekt | DDPM | Flow Matching |
|---|---|---|
| Trajektorien | Gebogen, komplex | Geradlinig (straight paths) |
| Sampling-Schritte | Viele nötig | Weniger nötig |
| Mathematik | Stochastische DGLs | Gewöhnliche DGLs |
| Qualität | Sehr gut | Mindestens genauso gut |
| Sampling-Geschwindigkeit | Langsam | Schneller |

#### Intuition: Warum geradlinige Pfade besser sind
DDPM lernt gebogene, chaotische Pfade durch den Daten-Raum.
Flow Matching lernt direkte Pfade (Transport entlang gerader Linien).
Weniger Kurven = weniger Schritte nötig = schneller.

#### Wo Flow Matching heute läuft
- **Stable Diffusion 3** — erster Mainstream-Einsatz
- **FLUX.1** — bester open-source Generator 2024
- **Sora** (OpenAI) — wahrscheinlich Flow Matching auf Video-Patches
- **Meta Movie Gen** — großskaliges Video-Generierungsmodell
        """)

    # ------------------------------------------------------------------ #
    with tabs[5]:
        section_header("Video Diffusion — Bilder + Zeit",
                       "Die nächste Frontier: temporale Konsistenz und Physik.")
        st.markdown(r"""
**Video** ist Bild + Zeit — eine zusätzliche Dimension.
Das schafft neue Herausforderungen:

#### Die drei großen Herausforderungen
        """)

        key_concept("🔄", "Temporale Konsistenz",
                    "Objekte dürfen nicht zwischen Frames 'flackern'. Das gleiche Gesicht muss von Frame zu Frame konsistent bleiben. "
                    "Sehr schwer, da jeder Frame separat entrauscht wird.")
        key_concept("💾", "Speicher-Skalierung",
                    "24 fps × 10s = 240 Frames. Jeder Frame 512×512×3. = 1,13 GB unkomprimiert. "
                    "Attention über alle Frame-Patches: O((240·N)²) — riesig!")
        key_concept("🌍", "Bewegungsphysik",
                    "Wasser, Feuer, Haare, Hände — alles mit komplexer Physik. "
                    "Modelle lernen Statistik, nicht Physik → Artefakte bei ungewöhnlichen Bewegungen.")

        st.markdown(r"""
#### Aktuelle Video-Generierungsmodelle 2025/26

| Modell | Organisation | Besonderheit |
|---|---|---|
| **Sora** | OpenAI | DiT auf Spacetime-Patches, bis 60s |
| **Veo 2** | Google DeepMind | Physik-realistisch, sehr hoch qualitativ |
| **Gen-3 Alpha** | Runway | Sehr konsistent, API verfügbar |
| **Kling 2.0** | Kuaishou | Günstig, gute Qualität |
| **Hunyuan Video** | Tencent | Open-Source! 13B Parameter |
| **Wan 2.1** | Alibaba | Open-Source, bester open-model 2025 |
| **CogVideoX** | Zhipu AI | Open-Source, gute Baseline |

#### Architekturen
- **VideoLDM / AnimateDiff**: Bild-Diffusion + temporale Attention-Layer
- **Sora**: Latent-Diffusion-Transformer (DiT) auf Video-Patches ("Spacetime Tubes")
- **Video DiT**: Generalisierung von DiT (Scalable Diffusion Transformer)

#### Aktueller Stand 2026
- ~10–30s Videos in sehr hoher Qualität möglich
- Physik bleibt ein offenes Problem (Wasser, Hände, Schrift)
- Audio-Sync und lange kohärente Narrative = nächste Frontier
- Open-Source holt rasant auf (Wan, Hunyuan, CogVideoX)
        """)

    # ------------------------------------------------------------------ #
    with tabs[6]:
        lab_header("Diffusion Forward-Process Simulator",
                   "Sieh, wie ein Bild schrittweise zu Rauschen wird.")

        st.markdown("Wähle Bild-Muster und simuliere den Forward-Noising-Prozess.")

        lc1, lc2 = st.columns(2)
        pattern = lc1.selectbox("Start-Bild", ["Schachbrett", "Streifen", "Kreis", "Gradient"])
        img_size = lc2.slider("Bildgröße", 32, 128, 64, step=32)
        t_step = st.slider("Zeitschritt t (0 = sauber, 1000 = reines Rauschen)", 0, 1000, 500)

        # Bild erstellen
        if pattern == "Schachbrett":
            s = img_size // 8
            y_idx, x_idx = np.indices((img_size, img_size))
            base = ((y_idx // s + x_idx // s) % 2).astype(float)
        elif pattern == "Streifen":
            base = np.tile(np.linspace(0, 1, 8), img_size // 8 + 1)[:img_size]
            base = np.tile(base, (img_size, 1))
        elif pattern == "Kreis":
            y_idx, x_idx = np.indices((img_size, img_size))
            dist = np.sqrt((y_idx - img_size/2)**2 + (x_idx - img_size/2)**2)
            base = (dist < img_size * 0.35).astype(float)
        else:
            y_idx, x_idx = np.indices((img_size, img_size))
            base = (y_idx / img_size + x_idx / img_size) / 2.0

        # DDPM Forward Process: x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε
        T = 1000
        beta_min, beta_max = 0.0001, 0.02
        betas = np.linspace(beta_min, beta_max, T)
        alphas = 1 - betas
        alpha_bar = np.cumprod(alphas)

        t_idx = max(0, min(t_step - 1, T - 1))
        ab = alpha_bar[t_idx]
        noise = np.random.default_rng(42).standard_normal(base.shape)
        x_t = np.sqrt(ab) * base + np.sqrt(1 - ab) * noise

        # Anzeigen
        show_cols = st.columns(5)
        show_steps = [0, 100, 250, 500, t_step]
        for col, ts in zip(show_cols, show_steps):
            ti = max(0, min(ts - 1, T - 1))
            ab_i = alpha_bar[ti]
            x_i = np.sqrt(ab_i) * base + np.sqrt(1 - ab_i) * np.random.default_rng(ti + 1).standard_normal(base.shape)
            x_i_vis = np.clip(x_i, -2, 2)
            x_i_vis = ((x_i_vis + 2) / 4 * 255).astype(np.uint8)
            col.markdown(f"**t={ts}**")
            col.image(x_i_vis, use_container_width=True, clamp=True)
            col.caption(f"SNR: {ab_i/(1-ab_i):.2f}")

        # Signal-to-Noise Ratio Kurve
        snr = alpha_bar / (1 - alpha_bar)
        ts_plot = np.arange(T)
        fig_snr = go.Figure()
        fig_snr.add_trace(go.Scatter(
            x=ts_plot, y=snr, fill="tozeroy",
            line=dict(color="#7C3AED", width=2),
            fillcolor="rgba(124,58,237,0.15)",
            name="SNR = ᾱ_t / (1-ᾱ_t)",
        ))
        fig_snr.add_vline(x=t_step, line_color="#EC4899", line_width=2,
                          annotation_text=f"t={t_step}", annotation_position="top right")
        fig_snr.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=220,
            xaxis=dict(title="Zeitschritt t"),
            yaxis=dict(title="SNR", type="log"),
            margin=dict(l=40, r=10, t=10, b=40),
            title="Signal-to-Noise Ratio über den Noise Schedule",
        )
        st.plotly_chart(fig_snr, use_container_width=True)
        info_box(
            f"Bei t={t_step}: SNR = {snr[t_idx]:.3f}. "
            "SNR > 1 → Signal dominiert. SNR < 1 → Rauschen dominiert. "
            "Bei t=1000 ist praktisch nur noch Rauschen übrig (SNR ≈ 0).",
            kind="info",
        )

    # ------------------------------------------------------------------ #
    with tabs[7]:
        section_header("Lernvideos", "Die besten Erklärungen zu Diffusion Models.")

        st.markdown("#### Diffusion Models (Explained) — Ari Seff (DeepMind)")
        video_embed("fbLgFrlYC3g",
                    "Denoising Diffusion Probabilistic Models — Ari Seff",
                    "Mathematisch korrekte, trotzdem verständliche Erklärung von DDPM. ~20 Minuten.")

        divider()

        st.markdown("#### Stable Diffusion, explained — Computerphile")
        video_embed("1CIpzeNxIhU",
                    "Stable Diffusion — Computerphile",
                    "Wie Stable Diffusion intern funktioniert, erklärt von Mike Pound. ~25 Minuten.")

        divider()

        st.markdown("#### The Math Behind Stable Diffusion (fast.ai)")
        video_embed("ISHKqgel_bw",
                    "Practical Deep Learning — Diffusion Math",
                    "Jeremy Howard und Jonathan Whitaker erklären die Mathematik tief und anwendungsorientiert.")

        info_box(
            "Nach diesen drei Videos hast du ein solides Verständnis von DDPM, "
            "Stable Diffusion und der zugrundeliegenden Mathematik. "
            "Dann lies das Original-DDPM Paper (Ho et al., 2020) — es ist sehr gut geschrieben.",
            kind="tip",
        )

    # ------------------------------------------------------------------ #
    with tabs[8]:
        render_learning_block(
            key_prefix="diffusion",
            section_title="Lernpfad für Diffusion",
            section_sub="Vom Prinzip bis zum eigenen Generator-Experiment",
            progression=[
                ("🟢", "Guided Lab", "Forward-Process simulieren und SNR entlang t interpretieren.", "Beginner", "green"),
                ("🟠", "Challenge Lab", "Vergleiche DDIM, DPM-Solver und Schritte vs. Qualität.", "Intermediate", "amber"),
                ("🔴", "Debug Lab", "Analysiere Artefakte durch Guidance Scale und Samplerwahl.", "Advanced", "pink"),
                ("🏁", "Mini-Projekt", "Prompt-Study: 20 Prompts, Metriken, kurze Ergebnisanalyse.", "Abschluss", "blue"),
            ],
            mcq_question="Wofür steht Classifier-Free Guidance im Kern?",
            mcq_options=[
                "Zusätzlicher externer Classifier steuert Sampling",
                "Kombination aus conditional und unconditional Noise-Schätzung",
                "Ausschließlich deterministisches Sampling",
                "Nur für Video-Diffusion nutzbar",
            ],
            mcq_correct_option="Kombination aus conditional und unconditional Noise-Schätzung",
            mcq_success_message="Genau. CFG kombiniert beide Vorhersagen über den Guidance-Faktor.",
            mcq_retry_message="Noch nicht. Lies die CFG-Formel im Sampling-Tab erneut.",
            open_question="Offene Frage: Wann würdest du Guidance reduzieren statt erhöhen?",
            code_task="""# Code-Aufgabe: experimentelles Grid über guidance_scale
scales = [3.0, 5.0, 7.5, 10.0, 14.0]
images = []
for gs in scales:
    out = pipe(
        prompt=prompt,
        num_inference_steps=25,
        guidance_scale=gs,
    ).images[0]
    images.append(out)
# TODO: Qualität vs. Prompttreue systematisch bewerten
""",
            community_rows=[
                {"Element": "Diskussion", "Frage": "Wo kippt CFG bei dir in Artefakte?", "Lieferobjekt": "2 Beispiele"},
                {"Element": "Peer-Review", "Frage": "Ist Samplerwahl begründet?", "Lieferobjekt": "Kurzkommentar"},
                {"Element": "Challenge", "Frage": "Bestes Bild bei <= 10 Schritten", "Lieferobjekt": "Prompt + Seed + Bild"},
            ],
            cheat_sheet=[
                "DDPM: robust, aber langsam; DDIM/DPM-Solver: schneller.",
                "CFG 6-8 oft guter Startbereich.",
                "Seed, Steps, Sampler und Prompt immer mitloggen.",
            ],
            key_takeaways=[
                "Diffusion lernt iteratives Entrauschen statt direkte Pixelvorhersage.",
                "Sampler und Guidance entscheiden massiv über Qualität und Stil.",
            ],
            common_errors=[
                "Zu hohe Guidance -> Übersättigung/Artefakte.",
                "Zu wenige Schritte bei komplexen Prompts.",
                "Samplerwechsel ohne gleiche Seeds/Settings.",
                "Fehlende negative prompts bei kritischen Motiven.",
                "Keine qualitative + quantitative Evaluation kombiniert.",
            ],
        )
