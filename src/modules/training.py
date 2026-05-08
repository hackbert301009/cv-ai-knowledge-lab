"""Training, Loss & Optimizer — die handwerkliche Seite des Deep Learnings."""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.components import (
    hero, section_header, divider, info_box,
    video_embed, lab_header, key_concept, step_list, render_learning_block,
)


def _cosine_lr(t, T, eta_min=1e-6, eta_max=1e-3):
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * t / T))


def _warmup_cosine_lr(t, warmup, T, eta_max=1e-3, eta_min=1e-6):
    if t < warmup:
        return eta_max * t / max(warmup, 1)
    return _cosine_lr(t - warmup, T - warmup, eta_min, eta_max)


def render():
    hero(
        eyebrow="Deep Learning · Modul 13",
        title="Training, Loss &amp; Optimizer",
        sub="Wie ein Netz wirklich lernt: Loss-Funktionen, Optimizer, Learning-Rate Scheduling, "
            "Regularisierung. Die handwerklichen Tricks, die zwischen 70% und 95% Accuracy entscheiden."
    )

    tabs = st.tabs([
        "📉 Loss-Funktionen",
        "🚀 Optimizer",
        "📊 LR-Schedule Lab",
        "🛡️ Regularisierung",
        "🔍 Training Debuggen",
        "🧪 Loss-Visualisierung",
        "🎬 Lernvideos",
        "🧭 Lernpfad & Übungen",
    ])

    # ------------------------------------------------------------------ #
    with tabs[0]:
        section_header("Loss-Funktionen", "Das Ziel, das das Training optimiert.")
        st.markdown(r"""
Der Loss (auch Cost Function, Objective) misst, wie weit die Vorhersage vom Ziel entfernt ist.
**Backpropagation** berechnet den Gradienten des Losses bezüglich aller Parameter.

#### Klassifikation
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(r"""
**Cross-Entropy (Multi-Class)**
$$\mathcal{L} = -\sum_{c=1}^{C} y_c \log \hat{y}_c$$

Standard für alle Klassifikationsaufgaben.
$y_c \in \{0,1\}$ (One-Hot), $\hat{y}_c$ = Softmax-Output.

**Binary Cross-Entropy**
$$\mathcal{L} = -[y \log \hat{y} + (1-y)\log(1-\hat{y})]$$

Für binäre Klassifikation oder Multi-Label (mehrere Klassen gleichzeitig).
""")
        with col2:
            st.markdown(r"""
**Focal Loss** (für unbalancierte Daten)
$$\mathcal{L}_\text{focal} = -(1-\hat{y})^\gamma \log \hat{y}$$

Mit $\gamma=2$ werden einfache Beispiele (hohe $\hat{y}$) sehr klein gewichtet.
Das Netz fokussiert auf schwierige Fälle.
Basis von RetinaNet — revolutionär für Object Detection.

**Label Smoothing**
One-Hot $[0, 1, 0]$ → $[0.05, 0.9, 0.05]$.
Modell wird nicht overconfident. Regularisiert.
""")

        st.markdown("#### Regression & Segmentation")
        st.markdown(r"""
| Aufgabe | Loss | Wann? |
|---|---|---|
| **Regression** | MSE ($\frac{1}{n}\sum(y-\hat{y})^2$) | Wenn Ausreißer ok. Stark bestraft. |
| **Robuste Regression** | MAE ($\frac{1}{n}\sum|y-\hat{y}|$) | Ausreißer-robust. |
| **Regression** | Huber/SmoothL1 | Beste von beiden: MSE nahe 0, MAE weit weg. |
| **Segmentation** | Dice Loss | $1 - \frac{2|Y \cap \hat{Y}|}{|Y| + |\hat{Y}|}$ — unbalancierte Klassen. |
| **Segmentation** | IoU Loss | Direkte Optimierung der Jaccard-Metrik. |
| **Detection** | GIoU / CIoU | Bounding-Box-Regression — geometrisch korrekt. |
| **Embedding** | Triplet/Contrastive | Ähnliche nah, unähnliche weit. Basis von CLIP. |
| **Self-Supervised** | InfoNCE | Verallgemeinerung von Contrastive. SimCLR, DINO. |
        """)

        # Focal Loss Visualisierung
        lab_header("Focal Loss Visualisierung")
        gamma_fl = st.slider("Gamma γ", 0.0, 5.0, 2.0, 0.5)
        p_range = np.linspace(0.01, 0.99, 200)
        ce = -np.log(p_range)
        focal = -(1 - p_range)**gamma_fl * np.log(p_range)

        fig_fl = go.Figure()
        fig_fl.add_trace(go.Scatter(x=p_range, y=ce, name="Cross-Entropy (γ=0)",
                                    line=dict(color="#9CA3AF", width=2, dash="dot")))
        fig_fl.add_trace(go.Scatter(x=p_range, y=focal,
                                    name=f"Focal Loss (γ={gamma_fl:.1f})",
                                    line=dict(color="#EC4899", width=2.5)))
        fig_fl.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=280,
            xaxis=dict(title="Vorhergesagte Wahrscheinlichkeit p̂"),
            yaxis=dict(title="Loss", range=[0, 5]),
            legend=dict(orientation="h", y=1.05),
            margin=dict(l=40, r=10, t=30, b=40),
        )
        st.plotly_chart(fig_fl, use_container_width=True)
        info_box(
            f"Mit γ={gamma_fl:.1f}: Bei p̂=0.9 (einfaches Beispiel) ist der Focal Loss "
            f"{focal[int(0.9 * 200 - 1)]:.3f} vs. CE {ce[int(0.9 * 200 - 1)]:.3f} — "
            "der Fokus auf schwierige Beispiele ist deutlich sichtbar.",
            kind="info",
        )

    # ------------------------------------------------------------------ #
    with tabs[1]:
        section_header("Optimizer", "Wie Gradienten in Gewichts-Updates übersetzt werden.")
        st.markdown(r"""
#### SGD — der Klassiker
$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}$$

Der einfachste Optimizer. Funktioniert, ist aber langsam und sensitiv gegenüber der Lernrate.

**SGD mit Momentum** (praktisch immer besser):
$$v_{t+1} = \mu v_t + \nabla_\theta \mathcal{L}$$
$$\theta_{t+1} = \theta_t - \eta v_{t+1}$$

Momentum $\mu = 0.9$ ist Standard. Glättet Schwankungen, beschleunigt in flachen Richtungen.

#### Adam — der Allrounder
Adam kombiniert **Momentum** (erste Momente) und **adaptive Lernraten** (zweite Momente):

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(Erster Moment / Momentum)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(Zweiter Moment / RMS)}$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \quad \text{(Bias-Korrektur)}$$
$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Standard: $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$, $\eta=10^{-3}$.

#### AdamW — Weight Decay korrekt
Adam hat einen Bug mit L2-Regularisierung. AdamW entkoppelt Weight Decay:

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda \theta_t$$

Der letzte Term (Weight Decay) wird **separat** angewendet, nicht via Gradienten.
**AdamW ist Standard für Transformers, ViT, ConvNeXt.**

#### Neuere Optimizer
| Optimizer | Vorteil | Nachteil |
|---|---|---|
| **Lion** (2023) | Nur Vorzeichen → 3× weniger Speicher | Experimentell |
| **Sophia** (2023) | Newton-Schritt via Hessian-Schätzung | Komplex |
| **Muon** (2024) | Orthogonaler Momentum-Update | Sehr neu |
| **Adam-mini** (2024) | Mini-Batch Adam, weniger Speicher | Nischenanwendung |
        """)

        info_box(
            "**Faustregel:** AdamW mit lr=3e-4 als Universal-Default. "
            "Bei CNNs: SGD+Momentum oft 0.5-1% besser, wenn du gut tunen kannst. "
            "Bei Transformers: AdamW + Cosine LR + Warmup.",
            kind="tip",
        )

    # ------------------------------------------------------------------ #
    with tabs[2]:
        lab_header("Learning-Rate Schedule Explorer",
                   "Vergleiche verschiedene LR-Schedules interaktiv.")

        tc1, tc2, tc3 = st.columns(3)
        total_epochs = tc1.slider("Gesamt-Epochen", 10, 300, 100)
        warmup_epochs = tc2.slider("Warmup-Epochen", 0, 50, 10)
        eta_max = tc3.select_slider(
            "Max. Lernrate",
            options=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
            value=1e-3,
            format_func=lambda x: f"{x:.0e}",
        )

        t = np.arange(total_epochs)

        def step_decay(t, T, eta, drop=0.5, drop_every=30):
            return eta * (drop ** (t // drop_every))

        def one_cycle(t, T, eta):
            if t < T * 0.45:
                return eta_max * 0.1 + (eta - eta_max * 0.1) * t / (T * 0.45)
            elif t < T * 0.9:
                return eta - (eta - eta_max * 0.01) * (t - T * 0.45) / (T * 0.45)
            else:
                return eta_max * 0.01 * (1 - (t - T * 0.9) / (T * 0.1))

        schedules = {
            "Konstant": np.full(len(t), eta_max),
            "Step Decay (×0.5 alle 30)": np.array([step_decay(ti, total_epochs, eta_max) for ti in t]),
            "Cosine Annealing": np.array([_cosine_lr(ti, total_epochs, eta_max * 0.001, eta_max) for ti in t]),
            "Warmup + Cosine": np.array([_warmup_cosine_lr(ti, warmup_epochs, total_epochs, eta_max) for ti in t]),
            "One-Cycle": np.array([one_cycle(ti, total_epochs, eta_max) for ti in t]).clip(0),
        }

        colors_lr = ["#9CA3AF", "#F59E0B", "#3B82F6", "#EC4899", "#10B981"]
        fig_lr = go.Figure()
        for (name, vals), color in zip(schedules.items(), colors_lr):
            fig_lr.add_trace(go.Scatter(x=t, y=vals, name=name,
                                         line=dict(color=color, width=2)))
        fig_lr.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=380,
            xaxis=dict(title="Epoche"),
            yaxis=dict(title="Lernrate", type="log"),
            legend=dict(orientation="v", x=1.01),
            margin=dict(l=40, r=160, t=10, b=40),
        )
        st.plotly_chart(fig_lr, use_container_width=True)

        st.markdown("#### Welchen Schedule wann?")
        st.markdown(r"""
| Schedule | Wann ideal? |
|---|---|
| **Konstant** | Nur zum Debuggen. Nie in Produktion. |
| **Step Decay** | CNNs der alten Schule. Einfach, funktioniert. |
| **Cosine Annealing** | Standard für die meisten CNNs. |
| **Warmup + Cosine** | **Standard für Transformer und ViT.** |
| **One-Cycle** | Fast.ai Empfehlung — schnelles Konvergieren. |
        """)

        st.code("""
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)

# Warmup: 10 Epochen linear von 1% auf 100% der LR
warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=10)

# Cosine: danach sinusförmig auf eta_min
cosine = CosineAnnealingLR(optimizer, T_max=90, eta_min=1e-6)

# Kombiniert: erst Warmup, dann Cosine
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[10])

# Training Loop
for epoch in range(100):
    train_one_epoch(model, optimizer)
    scheduler.step()
    print(f"Epoch {epoch}: LR = {scheduler.get_last_lr()[0]:.6f}")
        """, language="python")

    # ------------------------------------------------------------------ #
    with tabs[3]:
        section_header("Regularisierung — gegen Overfitting kämpfen",
                       "Mehr Techniken ≠ immer besser. Systematisch vorgehen.")
        st.markdown(r"""
**Overfitting**: Das Modell lernt die Trainingsdaten auswendig, generalisiert aber schlecht.

Erkennbar an: Training-Loss fällt, Validation-Loss steigt (Divergenz).
        """)

        cols_reg = st.columns(2)
        with cols_reg[0]:
            st.markdown(r"""
**Weight Decay (L2-Regularisierung)**
$$\mathcal{L}_\text{total} = \mathcal{L} + \frac{\lambda}{2}\|\theta\|^2$$

Bestraft große Gewichte. Zwingt das Netz, sparse Lösungen zu bevorzugen.
Typisch: $\lambda = 0.01$ bis $0.1$ (AdamW: 0.05).

**Dropout** (Srivastava et al., 2014)
Zufällig $p$ Anteil der Aktivierungen auf 0 setzen **beim Training**.
Beim Inference: alle Neuronen, aber mit Faktor $(1-p)$ skaliert.
Effektiv: trainiert ein Ensemble von $2^n$ verschiedenen Netzen.

**Batch Normalization**
Reguliert implizit durch die Rausch-Einführung beim Batch-Sampling.
Dazu stabiler Training, ermöglicht höhere LR.
""")
        with cols_reg[1]:
            st.markdown(r"""
**Data Augmentation** (die wichtigste Technik!)
Rotieren, Flipping, Color Jitter, CutOut, MixUp, CutMix.
Vergrößert den effektiven Datensatz virtuell.
Für ImageNet: RandAugment ist Standard.

**Label Smoothing**
Statt hartem One-Hot: Soft Labels.
$y = 0.9 \cdot \text{one\_hot} + 0.1/C$
Verhindert overconfident Predictions.

**Stochastic Depth** (DropPath)
Ganze Residual-Blöcke zufällig überspringen.
Standard in DeiT, Swin, ConvNeXt.

**Early Stopping**
Training beenden, wenn Validation-Loss nicht mehr fällt.
Die simpelste, aber oft effektivste Regularisierung.
""")

        info_box(
            "**Reihenfolge zum Testen:** 1) Mehr Daten / bessere Augmentation, "
            "2) Weight Decay erhöhen, 3) Dropout hinzufügen, 4) Modell kleiner machen. "
            "Mehr Regularisierung ist nicht immer besser — manche Tasks sind einfach und brauchen keines.",
            kind="tip",
        )

        divider()
        st.markdown("#### MixUp und CutMix — moderne Augmentation")
        st.markdown(r"""
**MixUp** (Zhang et al., 2017):
$$\tilde{x} = \lambda x_i + (1-\lambda) x_j, \quad \tilde{y} = \lambda y_i + (1-\lambda) y_j$$

Bilder linear mischen, Labels auch. $\lambda \sim \text{Beta}(\alpha, \alpha)$.

**CutMix** (Yun et al., 2019):
Rechteckiger Bereich von Bild A durch Bereich aus Bild B ersetzen.
Labels proportional zur Fläche mischen.
Noch stärker als MixUp für viele Tasks.
        """)
        st.code("""
import torch

def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0))
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Training mit MixUp
for batch in dataloader:
    x, y = batch
    x, y_a, y_b, lam = mixup_data(x, y, alpha=0.4)
    pred = model(x)
    loss = mixup_criterion(criterion, pred, y_a, y_b, lam)
    loss.backward()
    optimizer.step()
        """, language="python")

    # ------------------------------------------------------------------ #
    with tabs[4]:
        section_header("Training Debuggen",
                       "Systematische Diagnose wenn das Netz nicht lernt.")

        step_list([
            ("Daten anschauen — immer zuerst!",
             "Visualisiere einen Batch nach allen Augmentations. Sind Labels korrekt? "
             "Bilder lesbar? Falsche Labels sind der häufigste Bug."),
            ("Auf einem Mini-Batch overfitten",
             "Nimm 5-10 Bilder und trainiere bis Accuracy 100%. "
             "Falls es nicht klappt: Netz zu klein, Loss falsch, oder Optimizer-Bug."),
            ("Lernrate prüfen",
             "LR ist meistens das Problem. Halbiere sie / verdoppele sie. "
             "LR-Finder: von winziger LR aufwärts steigern, Loss beobachten."),
            ("Loss-Kurven aufzeichnen",
             "Training vs. Validation Loss über Epochen. Divergenz → Overfitting. "
             "Kein Lernen → LR zu klein, falscher Loss, oder Daten-Bug."),
            ("Gradient-Normen beobachten",
             "`torch.nn.utils.clip_grad_norm_` und logge die Norm. "
             "Sehr große Norms → explodierender Gradient. Plötzlich 0 → Vanishing Gradient."),
            ("Lernendes Netz vs. Zufälliges",
             "Am Anfang sollte Loss ≈ -log(1/C) für C Klassen sein (z.B. 2.30 für 10 Klassen). "
             "Falls Loss viel höher ist, ist etwas grundlegend falsch."),
        ])

        st.markdown("#### Häufige Bugs")
        st.markdown("""
| Symptom | Wahrscheinliche Ursache |
|---|---|
| Loss = NaN nach wenigen Steps | LR zu hoch, Explodierender Gradient, Division durch 0 |
| Loss fällt nicht | LR zu klein, Bug im Forward Pass, Labels falsch |
| Training gut, Val schlecht | Overfitting, Augmentation nur auf Training |
| Training = Val schlecht | Underfitting — Modell zu klein oder LR-Schedule falsch |
| Accuracy springt | shuffle=False, Batch-Reihenfolge nicht zufällig |
| Training langsam | Daten-Loading Bottleneck — `num_workers` erhöhen, Daten in RAM |
""")
        info_box(
            "Karpathys berühmter Tipp: **'Werde eins mit dem Datensatz.'** "
            "90% aller Bugs sind Daten-Bugs, nicht Code-Bugs. "
            "Schaue 100 Bilder an, bevor du anfängst zu trainieren.",
            kind="warn",
        )

    # ------------------------------------------------------------------ #
    with tabs[5]:
        lab_header("Loss-Kurven Simulator",
                   "Typische Training-Verläufe live erkunden.")

        st.markdown("Wähle ein Szenario und sieh, wie die Loss-Kurven aussehen.")
        scenario = st.selectbox("Szenario", [
            "Gut trainiertes Modell",
            "Overfitting (zu groß, kein Dropout)",
            "Underfitting (zu klein / LR zu niedrig)",
            "Explodierender Gradient (LR zu hoch)",
            "LR zu hoch → divergiert → LR verringert (manuell)",
        ])

        n_epochs = 100
        eps = np.arange(1, n_epochs + 1)
        rng = np.random.default_rng(7)

        if scenario == "Gut trainiertes Modell":
            train_l = 2.3 * np.exp(-0.035 * eps) + 0.05 + rng.normal(0, 0.02, n_epochs)
            val_l = 2.3 * np.exp(-0.03 * eps) + 0.08 + rng.normal(0, 0.03, n_epochs)
        elif scenario == "Overfitting (zu groß, kein Dropout)":
            train_l = 2.3 * np.exp(-0.06 * eps) + 0.01 + rng.normal(0, 0.01, n_epochs)
            val_l = 2.3 * np.exp(-0.02 * eps) + 0.3 + 0.004 * eps + rng.normal(0, 0.04, n_epochs)
        elif scenario == "Underfitting (zu klein / LR zu niedrig)":
            train_l = 2.3 * np.exp(-0.008 * eps) + 0.8 + rng.normal(0, 0.03, n_epochs)
            val_l = 2.3 * np.exp(-0.007 * eps) + 0.85 + rng.normal(0, 0.03, n_epochs)
        elif scenario == "Explodierender Gradient (LR zu hoch)":
            train_l = np.where(eps < 15,
                               2.3 - 0.1 * eps,
                               2.3 - 0.1 * 15 + np.exp(0.3 * (eps - 15)))
            train_l = np.clip(train_l, -1, 100)
            val_l = train_l + rng.normal(0, 0.5, n_epochs)
        else:
            train_l = np.where(
                eps < 20,
                2.3 + 0.2 * eps / 20 + rng.normal(0, 0.1, n_epochs),
                np.where(eps < 25, 4.0 - 0.2 * (eps - 20) + rng.normal(0, 0.1, n_epochs),
                         2.3 * np.exp(-0.03 * (eps - 25)) + 0.1 + rng.normal(0, 0.02, n_epochs)),
            )
            val_l = train_l + 0.05 + rng.normal(0, 0.05, n_epochs)

        train_l = np.clip(train_l, 0, 15)
        val_l = np.clip(val_l, 0, 15)

        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=eps, y=train_l, name="Training Loss",
                                       line=dict(color="#7C3AED", width=2.5)))
        fig_loss.add_trace(go.Scatter(x=eps, y=val_l, name="Validation Loss",
                                       line=dict(color="#EC4899", width=2.5, dash="dot")))
        if "Overfitting" in scenario:
            best_epoch = int(np.argmin(val_l)) + 1
            fig_loss.add_vline(x=best_epoch, line_color="#10B981", line_dash="dash",
                               annotation_text=f"Early Stop @ {best_epoch}", annotation_position="top right")
        fig_loss.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=350,
            xaxis=dict(title="Epoche"),
            yaxis=dict(title="Loss"),
            legend=dict(orientation="h", y=1.05),
            margin=dict(l=40, r=10, t=30, b=40),
        )
        st.plotly_chart(fig_loss, use_container_width=True)

        diagnoses = {
            "Gut trainiertes Modell": "✅ Beide Kurven fallen gleichmäßig, Lücke bleibt klein. Gutes Training.",
            "Overfitting (zu groß, kein Dropout)": "⚠️ Training fällt weiter, Validation steigt. Early Stopping, Dropout, Weight Decay oder mehr Daten!",
            "Underfitting (zu klein / LR zu niedrig)": "❌ Beide Kurven fallen kaum. Modell zu klein oder LR zu niedrig. Netz vergrößern oder LR erhöhen.",
            "Explodierender Gradient (LR zu hoch)": "💥 Loss explodiert. LR sofort reduzieren! Gradient Clipping einbauen.",
            "LR zu hoch → divergiert → LR verringert (manuell)": "🔄 Instabiler Start, dann Stabilisierung nach LR-Reduktion. Warmup würde das verhindern.",
        }
        info_box(diagnoses[scenario], kind="warn" if "⚠️" in diagnoses[scenario] or "❌" in diagnoses[scenario] else "success")

    # ------------------------------------------------------------------ #
    with tabs[6]:
        section_header("Lernvideos", "Training verstehen — mit den besten Erklärern.")

        st.markdown("#### Gradient descent, how neural networks learn (3Blue1Brown)")
        video_embed("IHZwWFHWa-w",
                    "Gradient descent — 3Blue1Brown",
                    "Wie Gradient Descent wirklich funktioniert — intuitive Visualisierung.")

        divider()

        st.markdown("#### What is backpropagation really doing? (3Blue1Brown)")
        video_embed("Ilg3gGewQ5U",
                    "Backpropagation — 3Blue1Brown",
                    "Die Kettenregel verständlich gemacht. Pflichtanschauen.")

        divider()

        st.markdown("#### A Recipe for Training Neural Networks — Andrej Karpathy (Blog + Talk)")
        video_embed("ORrStCArmP4",
                    "A Recipe for Training Neural Networks — Karpathy",
                    "Karpathys systematischer Ansatz zum Debuggen und Trainieren. "
                    "Goldstandard für Praktiker.")

        info_box(
            "Der ultimative Resource: Karpathys Blog-Post **'A Recipe for Training Neural Networks'** "
            "(google.com/search). Lese ihn zweimal — einmal am Anfang, einmal nach deinem ersten Projekt.",
            kind="tip",
        )

    # ------------------------------------------------------------------ #
    with tabs[7]:
        st.graphviz_chart(
            """
            digraph G {
                rankdir=LR;
                node [shape=box, style=rounded];
                A [label="Loss verstehen"];
                B [label="Optimizer wählen"];
                C [label="LR Schedule"];
                D [label="Debugging"];
                E [label="Mini-Projekt"];
                A -> B -> C -> D -> E;
            }
            """
        )
        render_learning_block(
            key_prefix="training",
            section_title="Lernpfad für Training & Optimizer",
            progression=[
                ("🟢", "Guided Lab", "Trainiere ein Basis-CNN mit AdamW und dokumentiere Loss/Val-Kurven.", "Beginner", "green"),
                ("🟠", "Challenge Lab", "Steigere Val-Accuracy durch Warmup + Cosine + Augmentation.", "Intermediate", "amber"),
                ("🔴", "Debug Lab", "Behebe NaN-Loss, Underfitting oder Overfitting systematisch.", "Advanced", "pink"),
                ("🏁", "Mini-Projekt", "Erstelle einen reproduzierbaren Training-Report mit Konfigurationsvergleich.", "Abschluss", "blue"),
            ],
            mcq_question="Welche Kombination ist meist ein robuster Default für moderne Transformer?",
            mcq_options=["SGD + konstante LR", "AdamW + Warmup + Cosine", "RMSProp + Step Decay", "Adam ohne Weight Decay"],
            mcq_correct_option="AdamW + Warmup + Cosine",
            mcq_success_message="Richtig. Das ist in vielen Setups der stabile Praxis-Default.",
            mcq_retry_message="Noch nicht korrekt. Prüfe die Optimizer-Sektion und LR-Schedules.",
            open_question="Offene Frage: Wie entscheidest du zwischen Overfitting und zu hoher Lernrate, wenn die Kurven instabil sind?",
            code_task="""# Code-Aufgabe: Early-Stopping korrekt ergänzen
best_val = float("inf")
patience = 5
counter = 0

for epoch in range(epochs):
    train_one_epoch(...)
    val_loss = validate(...)
    # TODO: speichere bestes Modell, erhöhe counter, stoppe bei patience
""",
            community_rows=[
                {"Format": "Diskussion", "Fokus": "Welche Hyperparameter brachten den größten Effekt?", "Output": "Kurzbegründung"},
                {"Format": "Peer-Feedback", "Fokus": "Ist der Trainingsvergleich fair und reproduzierbar?", "Output": "2 Stärken + 1 Verbesserung"},
                {"Format": "Challenge", "Fokus": "Bestes Ergebnis bei gleichem Epochenbudget", "Output": "Config + Kurve"},
            ],
            cheat_sheet=[
                "Erst Baseline, dann genau eine Änderung pro Experiment.",
                "LR-Finder oder grober LR-Sweep vor langem Training.",
                "Seeds, Scheduler und Augmentation immer mitloggen.",
            ],
            key_takeaways=[
                "Optimizer-Wahl, LR und Datenqualität entscheiden häufiger als Modellgröße.",
                "Debugging beginnt bei den Daten, nicht beim neuesten Trick.",
            ],
            common_errors=[
                "Mehrere Hyperparameter gleichzeitig ändern.",
                "Validation-Metrik nicht konsequent tracken.",
                "NaN-Fehler ohne Gradient-/LR-Check ignorieren.",
                "Kein Checkpoint des besten Modells.",
                "Fehlende Reproduzierbarkeit (Seed/Versionen).",
            ],
        )
