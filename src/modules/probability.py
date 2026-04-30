"""Wahrscheinlichkeit & Statistik für ML."""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.components import hero, section_header, divider, info_box


def render():
    hero(
        eyebrow="Grundlagen · Modul 4",
        title="Wahrscheinlichkeit &amp; Statistik",
        sub="Bayes, Verteilungen, Cross-Entropy, KL-Divergenz — die Werkzeuge, "
            "mit denen ML-Modelle Unsicherheit ausdrücken und messen."
    )

    tabs = st.tabs(["🎲 Basics", "📊 Verteilungen", "🔄 Bayes", "📉 Cross-Entropy", "🌊 KL-Divergenz"])

    with tabs[0]:
        section_header("Wahrscheinlichkeit — die Grundregeln")
        st.markdown(r"""
- **Zufallsvariable** $X$: Eine Variable, deren Wert von einem Zufallsexperiment abhängt.
- **Wahrscheinlichkeitsverteilung** $p(x)$: Wie wahrscheinlich jeder mögliche Wert ist.
- **Erwartungswert** $\mathbb{E}[X] = \sum_x x \cdot p(x)$ (oder $\int x p(x) dx$ für stetige).
- **Varianz** $\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2]$ — wie weit Werte streuen.

#### Bedingte Wahrscheinlichkeit
$$p(A | B) = \frac{p(A \cap B)}{p(B)}$$

Wahrscheinlichkeit von $A$, **gegeben** dass $B$ schon passiert ist.
Ein Klassifikator lernt $p(\text{Klasse} | \text{Bild})$.
        """)

    with tabs[1]:
        section_header("Verteilungen, die du immer wieder siehst")
        st.markdown(r"""
| Verteilung | Wofür? | Notation |
|---|---|---|
| **Bernoulli** | Binäre Klassifikation (Hund/Katze) | $\text{Bern}(p)$ |
| **Kategorial** | Multi-Class Klassifikation (10 Klassen) | $\text{Cat}(\boldsymbol{\pi})$ |
| **Gauß / Normal** | Rauschen, Initialisierung, Diffusion | $\mathcal{N}(\mu, \sigma^2)$ |
| **Uniform** | Zufällige Initialisierung, Augmentation | $\mathcal{U}(a, b)$ |

#### Die Gauß-Verteilung
$$p(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

Sie taucht **überall** auf:
- Gewichtsinitialisierung (He-Init, Xavier-Init)
- Diffusion Models (Forward Process fügt Gauß-Rauschen hinzu)
- Variational Autoencoders (latenter Raum ist gauß-verteilt)
        """)

        # interaktive Gauß
        c1, c2 = st.columns(2)
        mu = c1.slider("μ (Mittelwert)", -3.0, 3.0, 0.0, 0.1)
        sigma = c2.slider("σ (Standardabweichung)", 0.2, 3.0, 1.0, 0.1)
        x = np.linspace(-6, 6, 300)
        y = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
        fig = go.Figure(go.Scatter(x=x, y=y, fill="tozeroy",
                                   line=dict(color="#7C3AED", width=2),
                                   fillcolor="rgba(124,58,237,0.2)"))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=350,
                          title=f"𝒩({mu:.1f}, {sigma:.1f}²)")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        section_header("Satz von Bayes")
        st.markdown(r"""
$$p(A | B) = \frac{p(B | A) \cdot p(A)}{p(B)}$$

In ML-Sprache:
$$\underbrace{p(\theta | \mathcal{D})}_{\text{Posterior}} = \frac{\overbrace{p(\mathcal{D} | \theta)}^{\text{Likelihood}} \cdot \overbrace{p(\theta)}^{\text{Prior}}}{\underbrace{p(\mathcal{D})}_{\text{Evidence}}}$$

> **In Worten:** Was ich nach Sehen der Daten glaube = Was die Daten gegeben meiner Hypothese sagen × Was ich vorher glaubte.

#### Wo das auftaucht
- **Naive Bayes** Klassifikatoren
- **Bayesian Neural Networks** — Unsicherheit auf Gewichten
- **Variational Inference** — Annäherung an den Posterior
- **Bayesian Optimization** — clever Hyperparameter suchen
        """)

    with tabs[3]:
        section_header("Cross-Entropy — der Standard-Loss")
        st.markdown(r"""
Cross-Entropy zwischen wahrer Verteilung $p$ und vorhergesagter $q$:

$$H(p, q) = -\sum_i p_i \log q_i$$

Bei Klassifikation ist $p$ ein **One-Hot-Vektor** (z.B. $[0,1,0]$ für Klasse 2),
$q$ die Softmax-Ausgabe des Netzes. Dann reduziert sich Cross-Entropy zu:

$$\mathcal{L} = -\log q_{\text{wahr}}$$

Also: **negativer Log der Wahrscheinlichkeit, die das Netz der wahren Klasse gibt**.
- Netz ist sich sicher und richtig: $q_{\text{wahr}} \approx 1 \Rightarrow \mathcal{L} \approx 0$ ✅
- Netz ist sich sicher und falsch: $q_{\text{wahr}} \approx 0 \Rightarrow \mathcal{L} \to \infty$ ❌

Genau das wollen wir.
        """)
        st.code("""
import torch
import torch.nn.functional as F

logits  = torch.tensor([[2.0, 1.0, 0.1]])   # rohe Netz-Outputs
target  = torch.tensor([0])                  # wahre Klasse: 0

loss = F.cross_entropy(logits, target)
print(loss.item())   # Kombiniert Softmax + NLL in einem Schritt
        """, language="python")

    with tabs[4]:
        section_header("KL-Divergenz")
        st.markdown(r"""
$$D_{KL}(p \| q) = \sum_i p_i \log \frac{p_i}{q_i}$$

Misst, wie sehr eine Verteilung $q$ von einer Verteilung $p$ abweicht.

#### Eigenschaften
- $D_{KL}(p \| q) \geq 0$ immer.
- $D_{KL}(p \| q) = 0$ genau dann, wenn $p = q$.
- **Asymmetrisch**: $D_{KL}(p \| q) \neq D_{KL}(q \| p)$.

#### Wo wirst du KL begegnen?
- **VAE-Loss**: Regularisiert den latenten Raum gegen $\mathcal{N}(0, I)$
- **Knowledge Distillation**: Schüler-Modell soll Lehrer-Verteilung imitieren
- **Diffusion Models**: ELBO-Herleitung
- **Reinforcement Learning**: TRPO, PPO begrenzen Policy-Updates über KL
        """)
        info_box(
            "KL ist ein Maß, wie viel Information du verlierst, wenn du $p$ durch $q$ approximierst. "
            "Cross-Entropy = Entropie + KL — deshalb sind die beiden so eng verwandt.",
            kind="tip",
        )
