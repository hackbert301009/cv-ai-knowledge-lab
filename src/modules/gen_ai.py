"""Generative KI — GANs, VAEs, Autoregressive Modelle."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.components import (
    hero, section_header, divider, info_box, lab_header,
    render_learning_block, render_quiz_checkpoint,
)


def _decode(z: np.ndarray, size: int = 96) -> np.ndarray:
    """Spielzeug-'Decoder': latenter Vektor -> Bild (Summe von Sinus-Wellen)."""
    ys, xs = np.mgrid[0:size, 0:size] / size
    img = np.zeros((size, size))
    for k in range(0, len(z), 3):
        fx, fy, ph = z[k], z[k + 1], z[k + 2]
        img += np.sin(2 * np.pi * (fx * xs + fy * ys) + ph)
    img = (img - img.min()) / (img.ptp() + 1e-8)
    return img


def _latent(seed: int, dim: int = 12) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=dim)
    z[0::3] = np.abs(z[0::3]) * 3 + 1   # Frequenzen x
    z[1::3] = np.abs(z[1::3]) * 3 + 1   # Frequenzen y
    return z


def render():
    hero(
        eyebrow="State-of-the-Art · Generative KI",
        title="Generative KI",
        sub="Wie Maschinen kreativ werden. GANs, VAEs, Autoregressive Modelle — "
            "die Vorgänger und Verwandten von Diffusion."
    )

    tabs = st.tabs(["🎭 GANs", "🌀 VAEs", "📜 Autoregressive", "🆚 Vergleich", "🧪 Interaktiv"])

    with tabs[0]:
        section_header("Generative Adversarial Networks (Goodfellow, 2014)")
        st.markdown(r"""
**GAN** ist ein zweiteiliges Spiel:
- **Generator** $G$: erzeugt Bilder aus Rauschen $\mathbf{z}$
- **Discriminator** $D$: unterscheidet echte von generierten Bildern

#### Min-Max-Spiel
$$\min_G \max_D \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z}[\log(1 - D(G(\mathbf{z})))]$$

$D$ versucht, das Ganze zu durchschauen. $G$ versucht, $D$ zu täuschen.
Im Gleichgewicht erzeugt $G$ Bilder, die $D$ nicht von echten unterscheiden kann.

#### Berühmte Vertreter
- **DCGAN** (2015): erste stabile Bild-GANs
- **StyleGAN** (NVIDIA, 2018): hyperrealistische Gesichter, latente Style-Manipulation
- **CycleGAN** (2017): unpaired Image-to-Image (Pferd ↔ Zebra)
- **BigGAN** (2018): klassen-bedingt, hohe Auflösung
- **GigaGAN** (2023): Comeback — kann mit Diffusion mithalten

#### Probleme
- **Mode Collapse**: $G$ erzeugt nur wenige Variationen
- **Trainingsinstabilität**: $G$ und $D$ müssen sich balancieren
- **Schwer zu konditionieren** (im Vergleich zu Diffusion)

GANs sind seit Diffusion in der Defensive, aber **immer noch stark für Echtzeit-Generierung**
und Single-Step-Sampling.
        """)

    with tabs[1]:
        section_header("Variational Autoencoders (Kingma & Welling, 2013)")
        st.markdown(r"""
Ein **VAE** ist ein Autoencoder mit probabilistischem Bottleneck.

#### Architektur
- **Encoder** $q_\phi(\mathbf{z} | \mathbf{x})$: Bild → Verteilung im Latent (Mittelwert + Varianz)
- **Decoder** $p_\theta(\mathbf{x} | \mathbf{z})$: Latent-Sample → Bild

#### Loss
$$\mathcal{L} = \underbrace{\mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x}|\mathbf{z})]}_{\text{Rekonstruktion}} - \underbrace{D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))}_{\text{Regularisierung}}$$

Der KL-Term zwingt den Latent-Raum, einer Standardverteilung $\mathcal{N}(0, I)$ zu folgen.
Dadurch wird der Latent-Raum **glatt und sample-bar**.

#### Stärken
- Strukturierter Latent-Raum (man kann interpolieren)
- Stabiles Training
- Gute Repräsentationen für Downstream-Tasks
- Wird in Stable Diffusion als **Encoder/Decoder** benutzt

#### Schwächen
- Generierte Bilder oft **verschwommen** (Pixel-Average-Effekt)
- Wenn Schärfe wichtig ist: VAE-GAN Hybride oder Diffusion
        """)

    with tabs[2]:
        section_header("Autoregressive Modelle")
        st.markdown(r"""
Idee: Bilder als Sequenzen von Pixeln (oder Tokens) und vorhersagen Pixel für Pixel.

$$p(\mathbf{x}) = \prod_i p(x_i | x_{<i})$$

#### Bekannte Vertreter
- **PixelRNN, PixelCNN** (2016): Pixel für Pixel direkt
- **VQ-VAE + Transformer**: Bild → diskrete Tokens, dann LLM-Stil
- **DALL·E 1** (2021): genau dieser Ansatz mit Text-Conditioning
- **Parti** (Google 2022): autoregressiver Text-zu-Bild
- **MUSE / MaskGIT**: parallele Generierung statt sequenziell

#### Vorteile
- **Exaktes Likelihood** (gut für Density Estimation)
- **Stabiles Training**
- Profitiert von LLM-Tools (KV-Cache, etc.)

#### Nachteile
- **Langsam** beim Sampling (sequenziell)
- Diffusion hat sie für Bildqualität überholt

#### Aktuelles Comeback
- **Flow Matching mit DiT** und neue autoregressive Bild-Modelle in 2024/25
- Multimodale LLMs wie GPT-4o sind im Kern autoregressiv
        """)

    with tabs[3]:
        section_header("Vergleich: GAN vs. VAE vs. Autoregressive vs. Diffusion")
        st.markdown("""
| Eigenschaft | GAN | VAE | Autoregr. | Diffusion |
|---|---|---|---|---|
| **Qualität** | Sehr hoch | Mittel | Hoch | Höchste (2024) |
| **Diversität** | Niedrig (mode collapse) | Hoch | Hoch | Sehr hoch |
| **Sampling-Speed** | 🚀 Eine Forward-Pass | 🚀 Schnell | 🐢 Sequenziell | 🐢 Iterativ (10–50 Steps) |
| **Training-Stabilität** | Schwierig | Stabil | Stabil | Stabil |
| **Likelihood** | ❌ | Untere Schranke | ✅ Exakt | Untere Schranke |
| **Conditioning** | Mittel | Mittel | Stark | Sehr stark |
| **Echtzeit-Inferenz** | ✅ | ✅ | ❌ | Mittlerweile ✅ (Consistency, LCM) |
""")
        info_box(
            "Heute (2025/26): Diffusion dominiert für Bildqualität, GANs für Echtzeit, "
            "Autoregressive sind das Rückgrat multimodaler LLMs, VAEs leben als Encoder weiter.",
            kind="tip",
        )

    # ── Tab 4: Interaktiv ────────────────────────────────────────────────────
    with tabs[4]:
        lab_header("GAN Min-Max Spiel", "Wie gut täuscht der Generator den Discriminator?")
        skill = st.slider("Generator-Qualität (0 = Rauschen, 1 = perfekt echt)", 0.0, 1.0, 0.5, 0.05)
        # Discriminator-Ausgabe: Wahrscheinlichkeit "echt" für gefälschte Bilder
        d_fake = 0.05 + 0.9 * skill        # steigt mit Generator-Qualität
        d_real = 0.95                       # D erkennt echte Bilder gut
        g_loss = -np.log(d_fake + 1e-8)     # G will D(fake) -> 1
        d_loss = -(np.log(d_real + 1e-8) + np.log(1 - d_fake + 1e-8))
        c1, c2, c3 = st.columns(3)
        c1.metric("D(gefälscht)", f"{d_fake:.2f}", help="Wahrscheinlichkeit, dass D das Fake für echt hält")
        c2.metric("Generator-Loss", f"{g_loss:.2f}")
        c3.metric("Discriminator-Loss", f"{d_loss:.2f}")
        if skill > 0.9:
            info_box("Nash-Gleichgewicht nahe: D rät nur noch (~0.5) — G gewinnt.", kind="success")
        elif skill < 0.2:
            info_box("G ist zu schwach — D durchschaut jedes Fake. Hoher Generator-Loss.", kind="warn")

        divider()
        lab_header("VAE Latent-Interpolation", "Zwischen zwei Punkten im Latent-Raum morphen — der Kern eines glatten Latent-Raums.")
        cc1, cc2 = st.columns(2)
        seed_a = cc1.number_input("Seed A", 0, 9999, 7)
        seed_b = cc2.number_input("Seed B", 0, 9999, 42)
        alpha = st.slider("Interpolation α (A → B)", 0.0, 1.0, 0.5, 0.05)
        za, zb = _latent(int(seed_a)), _latent(int(seed_b))
        z = (1 - alpha) * za + alpha * zb
        g1, g2, g3 = st.columns(3)
        g1.image(_decode(za), caption="A (α=0)", use_container_width=True, clamp=True)
        g2.image(_decode(z), caption=f"Interpoliert (α={alpha:g})", use_container_width=True, clamp=True)
        g3.image(_decode(zb), caption="B (α=1)", use_container_width=True, clamp=True)
        info_box(
            "In einem gut regularisierten VAE-Latent-Raum ergibt jede Zwischenposition ein **plausibles** Bild — "
            "genau das macht den KL-Term. Bei einem klassischen Autoencoder wären die Zwischenpunkte oft Müll.",
            kind="info",
        )

    divider()
    render_learning_block(
        key_prefix="gen_ai",
        progression=[
            ("🟢", "Guided", "Erkläre das Min-Max-Ziel eines GAN in eigenen Worten.", "Guided", "green"),
            ("🟠", "Challenge", "Ordne 4 Anwendungen dem besten Modelltyp zu (Echtzeit-Avatar, Density Estimation, …).", "Challenge", "amber"),
            ("🔴", "Debug", "Ein GAN erzeugt nur ein einziges Motiv — benenne das Phänomen und eine Gegenmaßnahme.", "Debug", "pink"),
            ("🧩", "Mini-Projekt", "Trainiere einen kleinen VAE auf MNIST und interpoliere im Latent-Raum.", "Projekt", "blue"),
        ],
        mcq_question="Was verursacht 'Mode Collapse' bei GANs?",
        mcq_options=[
            "Der Generator produziert nur wenige Varianten, die D zuverlässig täuschen",
            "Der Discriminator hat zu wenige Parameter",
            "Die Lernrate ist zu niedrig",
            "Es fehlt der KL-Term",
        ],
        mcq_correct_option="Der Generator produziert nur wenige Varianten, die D zuverlässig täuschen",
        open_question="Warum sind VAE-Bilder oft verschwommen, Diffusion-Bilder aber scharf?",
        cheat_sheet=[
            "GAN: Generator vs. Discriminator, Min-Max-Spiel.",
            "VAE: Rekonstruktion + KL-Regularisierung → glatter Latent-Raum.",
            "Autoregressiv: p(x)=∏ p(xᵢ|x<ᵢ), exaktes Likelihood, langsam.",
            "Diffusion: iteratives Entrauschen, heute Qualitätsführer.",
        ],
        key_takeaways=[
            "Jede Familie tauscht Qualität, Diversität, Speed und Likelihood anders ab.",
            "VAEs und Autoregressive leben in modernen Systemen (Diffusion-VAE, multimodale LLMs) weiter.",
        ],
        common_errors=[
            "GAN-Training ohne Balance zwischen G und D.",
            "VAE als scharfen Bildgenerator erwarten.",
            "Autoregressive Modelle für Echtzeit einsetzen.",
        ],
    )
    render_quiz_checkpoint(
        key_prefix="gen_ai",
        module_id="gen_ai",
        question="Welche Rolle spielt der KL-Term im VAE-Loss?",
        options=[
            "Er zwingt den Latent-Raum in Richtung N(0, I) und macht ihn glatt/sample-bar",
            "Er misst die Bildschärfe",
            "Er ersetzt den Discriminator",
            "Er beschleunigt das Sampling",
        ],
        correct_option="Er zwingt den Latent-Raum in Richtung N(0, I) und macht ihn glatt/sample-bar",
        checklist=[
            "Ich kann GAN, VAE, Autoregressiv und Diffusion gegeneinander abgrenzen.",
            "Ich verstehe Mode Collapse.",
            "Ich weiß, warum VAE-Latent-Interpolation glatt ist.",
        ],
    )
