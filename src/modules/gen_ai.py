"""Generative KI — GANs, VAEs, Autoregressive Modelle."""
import streamlit as st
from src.components import hero, section_header, divider, info_box


def render():
    hero(
        eyebrow="State-of-the-Art · Modul 18",
        title="Generative KI",
        sub="Wie Maschinen kreativ werden. GANs, VAEs, Autoregressive Modelle — "
            "die Vorgänger und Verwandten von Diffusion."
    )

    tabs = st.tabs(["🎭 GANs", "🌀 VAEs", "📜 Autoregressive", "🆚 Vergleich"])

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

GANs sind seit Diffusion in der Defensive, aber **immer noch state-of-the-art für Echtzeit-Generierung**
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
- Multimodale LLMs wie GPT-4o sind im Kern autoregressive
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
