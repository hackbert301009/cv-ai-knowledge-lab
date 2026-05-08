"""Neuronale Netze von Grund auf."""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.components import (
    hero, section_header, divider, info_box,
    video_embed, lab_header, key_concept, step_list, render_learning_block,
)


# ---- Hilfsfunktionen für das Lab -----
def _sigmoid(x): return 1 / (1 + np.exp(-x))
def _relu(x):    return np.maximum(0, x)
def _tanh(x):    return np.tanh(x)
def _gelu(x):    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
def _leaky(x):   return np.where(x > 0, x, 0.01 * x)


def render():
    hero(
        eyebrow="Deep Learning · Modul 11",
        title="Neuronale Netze von Grund auf",
        sub="Vom einzelnen Perzeptron zum tiefen Netz. Die fundamentale Architektur, "
            "auf der **alles** in modernem Deep Learning aufbaut — Schritt für Schritt erklärt."
    )

    tabs = st.tabs([
        "⚛️ Perzeptron",
        "🧠 MLP",
        "🎚️ Aktivierungen",
        "⬅️ Backpropagation",
        "🔥 Universal Approx",
        "🧪 Interaktives Lab",
        "💻 Eigenes Netz",
        "🎬 Lernvideos",
        "🧭 Lernpfad & Übungen",
    ])

    # ------------------------------------------------------------------ #
    with tabs[0]:
        section_header("Das Perzeptron — die einfachste Form",
                       "McCulloch & Pitts (1943), Rosenblatt (1958)")
        st.markdown(r"""
Das **Perzeptron** ist das einfachste "Neuron" — inspiriert von biologischen Neuronen.

$$y = \sigma\!\left(\sum_{i} w_i x_i + b\right) = \sigma(\mathbf{w}^\top \mathbf{x} + b)$$

| Symbol | Bedeutung |
|---|---|
| $\mathbf{x} \in \mathbb{R}^n$ | Eingabe (z.B. Pixel-Werte) |
| $\mathbf{w} \in \mathbb{R}^n$ | Gewichte (was das Neuron gelernt hat) |
| $b \in \mathbb{R}$ | Bias (verschiebt die Aktivierungsschwelle) |
| $\sigma$ | Aktivierungsfunktion (z.B. Stufenfunktion, Sigmoid) |
| $y$ | Ausgabe (Vorhersage) |

#### Biologische Analogie
| Biologisch | Künstlich |
|---|---|
| Dendriten | Eingaben $x_i$ |
| Synapsen | Gewichte $w_i$ |
| Zellkörper (Soma) | Gewichtete Summe + Bias |
| Aktionspotenzial | Aktivierungsfunktion $\sigma$ |
| Axon | Ausgabe $y$ |

#### Was kann ein Perzeptron lösen?
- **Kann**: AND, OR, NOT (linear trennbare Probleme)
- **Kann nicht**: XOR — das hat in den 1970ern den ersten **KI-Winter** ausgelöst (Minsky & Papert)

**Lösung:** Mehrere Neuronen in mehreren Schichten stapeln.
        """)

        key_concept("⚡", "Linearer Klassifikator",
                    "Ein einzelnes Perzeptron trennt den Eingaberaum durch eine Hyperebene (Gerade in 2D). "
                    "Nur linear trennbare Probleme sind lösbar.")
        key_concept("🧬", "Biologische Inspiration",
                    "Obwohl von Neuronen inspiriert, sind künstliche Netze viel einfacher. "
                    "Das Gehirn hat ~86 Milliarden Neuronen mit 100-10.000 Synapsen je — und lernt völlig anders.")

    # ------------------------------------------------------------------ #
    with tabs[1]:
        section_header("Multi-Layer Perceptron (MLP)", "Mehrere Schichten = nichtlineare Grenzflächen.")
        st.markdown(r"""
Ein **MLP** (auch Fully Connected Network, Dense Network) stapelt mehrere Schichten von Neuronen:

$$\mathbf{h}^{(1)} = \sigma_1\!\left(\mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}\right)$$
$$\mathbf{h}^{(2)} = \sigma_2\!\left(\mathbf{W}^{(2)} \mathbf{h}^{(1)} + \mathbf{b}^{(2)}\right)$$
$$\vdots$$
$$\hat{\mathbf{y}} = \mathbf{W}^{(L)} \mathbf{h}^{(L-1)} + \mathbf{b}^{(L)}$$

#### Anatomie eines MLPs
        """)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
| Layer | Rolle |
|---|---|
| **Input Layer** | Nimmt rohe Daten auf |
| **Hidden Layers** | Lernen interne Repräsentationen |
| **Output Layer** | Klassifikation oder Regression |
| **Aktivierungen** | Nichtlinearität zwischen Layern |
""")
        with col2:
            # MLP Netzdiagramm mit Plotly
            fig = go.Figure()
            layer_sizes = [3, 4, 4, 2]
            layer_x = [0, 1, 2, 3]
            colors = ["#3B82F6", "#7C3AED", "#7C3AED", "#EC4899"]
            labels = ["Input", "Hidden 1", "Hidden 2", "Output"]
            pos = []
            for li, (n, lx) in enumerate(zip(layer_sizes, layer_x)):
                ys = np.linspace(0, 1, n)
                for yi in ys:
                    pos.append((lx, yi, li))
                # Draw connections to next layer
                if li < len(layer_sizes) - 1:
                    next_ys = np.linspace(0, 1, layer_sizes[li + 1])
                    for y1 in ys:
                        for y2 in next_ys:
                            fig.add_trace(go.Scatter(
                                x=[lx, lx + 1], y=[y1, y2],
                                mode="lines", line=dict(color="rgba(255,255,255,0.07)", width=1),
                                showlegend=False,
                            ))
            for lx, ly, li in pos:
                fig.add_trace(go.Scatter(
                    x=[lx], y=[ly], mode="markers",
                    marker=dict(size=20, color=colors[li], line=dict(color="white", width=1.5)),
                    showlegend=False,
                ))
            for li, (lx, label) in enumerate(zip(layer_x, labels)):
                fig.add_annotation(x=lx, y=-0.15, text=label, showarrow=False,
                                   font=dict(size=10, color="#9CA3AF"))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", height=240,
                margin=dict(l=10, r=10, t=10, b=40),
                xaxis=dict(visible=False), yaxis=dict(visible=False),
            )
            st.plotly_chart(fig, use_container_width=True)

        info_box(
            "**Warum Aktivierungsfunktionen?** Ohne sie kollabiert das gesamte MLP zu einer einzigen "
            "linearen Transformation — egal wie viele Layer. $\\mathbf{W}^{(2)}(\\mathbf{W}^{(1)}\\mathbf{x}) = (\\mathbf{W}^{(2)}\\mathbf{W}^{(1)})\\mathbf{x}$",
            kind="warn",
        )

    # ------------------------------------------------------------------ #
    with tabs[2]:
        section_header("Aktivierungsfunktionen", "Die Nichtlinearitäten, die alles ermöglichen.")
        st.markdown(r"""
| Funktion | Formel | Vorteil | Nachteil |
|---|---|---|---|
| **Sigmoid** | $\frac{1}{1+e^{-x}}$ | Output in $(0,1)$ — Wahrscheinlichkeit | Vanishing Gradient; nicht zero-centered |
| **Tanh** | $\tanh(x)$ | Zero-centered; Output in $(-1,1)$ | Vanishing Gradient |
| **ReLU** | $\max(0, x)$ | Kein Vanishing; sehr schnell | "Sterbende ReLUs" (Dead ReLU Problem) |
| **Leaky ReLU** | $\max(0.01x, x)$ | Löst Dead-ReLU-Problem | Hyperparameter (Slope) |
| **ELU** | $\begin{cases} x & x>0 \\ \alpha(e^x-1) & x\leq 0 \end{cases}$ | Glatter als ReLU, zero-centered | Teurer zu berechnen |
| **GELU** | $x \cdot \Phi(x)$ | Standard in BERT, GPT, ViT | Etwas teurer |
| **Swish/SiLU** | $x \cdot \sigma(x)$ | Glatte Version von ReLU; sehr gut | — |
| **Softmax** | $\frac{e^{x_i}}{\sum_j e^{x_j}}$ | Normalisiert zu Wahrscheinlichkeiten | Nur für Output-Layer |

#### Das Vanishing-Gradient-Problem
Bei Sigmoid/Tanh: $|\sigma'(x)| \leq 0.25$. Nach $L$ Layern Backpropagation:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(1)}} \approx \prod_{l=2}^{L} |\sigma'| \cdot \ldots \leq (0.25)^{L-1}$$

Bei 10 Layern: $(0.25)^9 \approx 0.0000004$ — der Gradient ist praktisch 0.
ReLU löst dies: $|\text{ReLU}'(x)| = 1$ für $x > 0$.
        """)

        x = np.linspace(-5, 5, 400)
        funcs = {
            "Sigmoid": _sigmoid(x),
            "Tanh":    _tanh(x),
            "ReLU":    _relu(x),
            "GELU":    _gelu(x),
            "Leaky ReLU": _leaky(x),
        }
        colors_f = ["#7C3AED", "#EC4899", "#F59E0B", "#06B6D4", "#10B981"]
        fig_act = go.Figure()
        for (name, y), c in zip(funcs.items(), colors_f):
            fig_act.add_trace(go.Scatter(x=x, y=y, name=name,
                                         line=dict(color=c, width=2.5)))
        fig_act.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=1)
        fig_act.add_vline(x=0, line_color="rgba(255,255,255,0.15)", line_width=1)
        fig_act.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=380,
            legend=dict(orientation="h", y=1.05),
            xaxis=dict(title="x"), yaxis=dict(title="σ(x)"),
            margin=dict(l=40, r=10, t=30, b=40),
        )
        st.plotly_chart(fig_act, use_container_width=True)

    # ------------------------------------------------------------------ #
    with tabs[3]:
        section_header("Backpropagation", "Der Algorithmus, der Training erst möglich macht.")
        st.markdown(r"""
**Backpropagation** (Rumelhart et al., 1986) ist die Anwendung der **Kettenregel** der
Analysis auf tiefe Netze.

#### Intuition
1. **Forward Pass**: Eingabe durch das Netz → Ausgabe $\hat{y}$
2. **Loss berechnen**: $\mathcal{L} = \text{CrossEntropy}(\hat{y}, y)$
3. **Backward Pass**: Wie viel hat jedes Gewicht zum Fehler beigetragen?

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{\partial \mathcal{L}}{\partial h^{(l)}} \cdot \frac{\partial h^{(l)}}{\partial W^{(l)}}$$

#### Die Kettenregel im Detail
Für eine Komposition von Funktionen $y = f(g(x))$:
$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$$

Für ein MLP mit $L$ Layern:
$$\frac{\partial \mathcal{L}}{\partial W^{(1)}} = \frac{\partial \mathcal{L}}{\partial h^{(L)}} \cdot \frac{\partial h^{(L)}}{\partial h^{(L-1)}} \cdots \frac{\partial h^{(2)}}{\partial h^{(1)}} \cdot \frac{\partial h^{(1)}}{\partial W^{(1)}}$$

Dieser rekursive "Rücklauf" der Gradienten ist **Backpropagation**.
        """)

        step_list([
            ("Forward Pass ausführen",
             "Alle Aktivierungen h⁽¹⁾, h⁽²⁾, ... h⁽ᴸ⁾ berechnen und cachen (wir brauchen sie für den Backward Pass)."),
            ("Loss berechnen",
             "L = CrossEntropy(ŷ, y) oder MSE. Das ist der Fehler, den wir minimieren wollen."),
            ("Output-Gradient berechnen",
             "∂L/∂ŷ — oft sehr simpel. Für Softmax+CrossEntropy: ∂L/∂ŷᵢ = ŷᵢ - yᵢ."),
            ("Layer für Layer rückwärts",
             "Kettenregel anwenden. Jeder Layer empfängt ∂L/∂h⁽ˡ⁾ und gibt ∂L/∂h⁽ˡ⁻¹⁾ weiter."),
            ("Gewichte aktualisieren",
             "θ ← θ - η · ∂L/∂θ. Das ist Gradient Descent."),
        ])

        info_box(
            "**PyTorch macht das automatisch!** `loss.backward()` berechnet alle Gradienten via Autograd. "
            "Du musst Backprop nur einmal von Hand verstehen — dann nutzt du PyTorch.",
            kind="success",
        )

    # ------------------------------------------------------------------ #
    with tabs[4]:
        section_header("Universal Approximation Theorem",
                       "Warum ein MLP theoretisch alles lernen kann.")
        st.markdown(r"""
> **Theorem (Hornik, Cybenko, 1989):**
> Ein MLP mit **einer einzigen Hidden Layer** und genug Neuronen kann
> **jede stetige Funktion** auf einem kompakten Raum beliebig genau approximieren.

$$\forall \varepsilon > 0, \exists N: \quad \sup_{x \in K} |f(x) - \text{MLP}_N(x)| < \varepsilon$$

#### Was das Theorem sagt — und was nicht
| Was es sagt | Was es **nicht** sagt |
|---|---|
| Approximation ist möglich | Wie viele Neuronen gebraucht werden |
| Für stetige Funktionen | Wie man die Gewichte findet |
| Auf kompakten Mengen | Ob Training konvergiert |
| Bei genügend Neuronen | Ob die Lösung generalisiert |

#### Warum dann tiefe Netze?
- **Effizienz**: Tiefe Netze brauchen exponentiell weniger Parameter für viele Funktionen
- **Hierarchische Repräsentationen**: Kanten → Texturen → Teile → Objekte
- **Induktiver Bias**: Tiefe entspricht Kompositionsstufen — die Welt ist kompositionell

#### Das Depth vs Width Trade-off
- **Breites, flaches Netz**: Kann alles, aber ineffizient
- **Tiefes, schmales Netz**: Effizient für hierarchisch strukturierte Daten

Empirisch: Tiefe Netze generalisieren für reale Daten (Bilder, Sprache) viel besser.
        """)

        lab_header("Universal Approximation Demo", "Ein Mini-MLP lernt sin(x).")
        np.random.seed(42)
        x_demo = np.linspace(-np.pi, np.pi, 200)
        y_target = np.sin(x_demo)

        n_neurons = st.slider("Anzahl Neuronen in der Hidden Layer", 2, 64, 16)
        W1 = np.random.randn(n_neurons, 1) * 1.5
        b1 = np.random.randn(n_neurons) * 0.5
        W2 = np.random.randn(1, n_neurons) * 0.5
        b2 = np.zeros(1)

        h = np.maximum(0, x_demo[:, None] @ W1.T + b1)
        y_approx = h @ W2.T + b2
        y_approx = y_approx.squeeze()

        fig_approx = go.Figure()
        fig_approx.add_trace(go.Scatter(x=x_demo, y=y_target, name="sin(x) (Ziel)",
                                         line=dict(color="#EC4899", width=2.5)))
        fig_approx.add_trace(go.Scatter(x=x_demo, y=y_approx, name=f"MLP {n_neurons} Neuronen (zufällig)",
                                         line=dict(color="#F59E0B", width=2, dash="dot")))
        fig_approx.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=300,
            legend=dict(orientation="h", y=1.1),
            margin=dict(l=20, r=10, t=30, b=20),
        )
        st.plotly_chart(fig_approx, use_container_width=True)
        info_box(
            f"Mit {n_neurons} Neuronen (noch **untrainiert** — zufällige Gewichte). "
            "Nach Training mit Gradient Descent konvergiert die Approximation.",
            kind="info",
        )

    # ------------------------------------------------------------------ #
    with tabs[5]:
        lab_header("Interaktives Neuron-Lab",
                   "Stelle Gewichte und Bias ein und sieh, wie ein einzelnes Neuron reagiert.")

        st.markdown("#### Ein einzelnes Neuron mit 2 Eingaben")
        c1, c2, c3 = st.columns(3)
        w1 = c1.slider("Gewicht w₁", -3.0, 3.0, 1.0, 0.1)
        w2 = c2.slider("Gewicht w₂", -3.0, 3.0, 0.5, 0.1)
        bias = c3.slider("Bias b", -3.0, 3.0, 0.0, 0.1)
        act_choice = st.selectbox("Aktivierungsfunktion", ["Sigmoid", "ReLU", "Tanh", "GELU"])

        act_fn = {"Sigmoid": _sigmoid, "ReLU": _relu, "Tanh": _tanh, "GELU": _gelu}[act_choice]

        # 2D-Grid der Ausgabe
        x1_range = np.linspace(-3, 3, 100)
        x2_range = np.linspace(-3, 3, 100)
        X1, X2 = np.meshgrid(x1_range, x2_range)
        Z = act_fn(w1 * X1 + w2 * X2 + bias)

        fig_neuron = go.Figure(go.Heatmap(
            x=x1_range, y=x2_range, z=Z,
            colorscale=[
                [0, "#0B0B0F"], [0.5, "#7C3AED"], [1, "#F59E0B"]
            ],
            showscale=True,
            colorbar=dict(title="Ausgabe", thickness=12, len=0.8),
        ))

        # Entscheidungsgrenze einzeichnen (wo z ≈ 0.5 für Sigmoid, 0 für andere)
        threshold = 0.5 if act_choice == "Sigmoid" else 0.0
        x1_boundary = np.linspace(-3, 3, 100)
        if abs(w2) > 1e-6:
            x2_boundary = (-w1 * x1_boundary - bias + threshold) / w2
            valid = (x2_boundary >= -3) & (x2_boundary <= 3)
            fig_neuron.add_trace(go.Scatter(
                x=x1_boundary[valid], y=x2_boundary[valid],
                mode="lines", line=dict(color="#EC4899", width=2, dash="dash"),
                name="Grenze",
            ))

        fig_neuron.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=380,
            xaxis=dict(title="Eingabe x₁", range=[-3, 3]),
            yaxis=dict(title="Eingabe x₂", range=[-3, 3]),
            margin=dict(l=40, r=40, t=20, b=40),
        )
        st.plotly_chart(fig_neuron, use_container_width=True)

        st.markdown(
            f"**Aktueller Wert:** $z = {w1:.1f} \\cdot x_1 + {w2:.1f} \\cdot x_2 + ({bias:.1f})$, "
            f"Ausgabe $= {act_choice}(z)$"
        )

        divider()
        st.markdown("#### XOR-Problem — warum ein Neuron nicht reicht")
        st.markdown(r"""
XOR gibt 1 aus, wenn genau eine der beiden Eingaben 1 ist.
Kein einzelnes Neuron (keine Gerade) kann die vier Punkte korrekt trennen.

| x₁ | x₂ | XOR |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |
        """)
        xor_pts = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
        cols_xor = [("🔴", "#EF4444"), ("🔵", "#3B82F6")]
        fig_xor = go.Figure()
        for x1v, x2v, label in xor_pts:
            fig_xor.add_trace(go.Scatter(
                x=[x1v], y=[x2v], mode="markers+text",
                marker=dict(size=22, color=cols_xor[label][1], line=dict(color="white", width=2)),
                text=[f"XOR={label}"], textposition="top center",
                showlegend=False,
            ))
        fig_xor.add_annotation(x=0.5, y=0.5, text="Keine Gerade trennt 0s und 1s!",
                                showarrow=False, font=dict(color="#F59E0B", size=13))
        fig_xor.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=280,
            xaxis=dict(range=[-0.5, 1.5], title="x₁"),
            yaxis=dict(range=[-0.5, 1.5], title="x₂"),
            margin=dict(l=40, r=10, t=10, b=40),
        )
        st.plotly_chart(fig_xor, use_container_width=True)

    # ------------------------------------------------------------------ #
    with tabs[6]:
        section_header("Eigenes Mini-MLP in 30 Zeilen",
                       "Vollständiges neuronales Netz in reinem NumPy — kein Framework.")
        st.code("""
import numpy as np

class MLP:
    def __init__(self, sizes):
        # He-Initialisierung für ReLU-Netze: σ = sqrt(2/n_in)
        self.W = [np.random.randn(o, i) * np.sqrt(2.0 / i)
                  for i, o in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(o) for o in sizes[1:]]

    def relu(self, x):   return np.maximum(0, x)
    def relu_d(self, x): return (x > 0).astype(float)

    def forward(self, x):
        self.cache = [x]
        for W, b in zip(self.W[:-1], self.b[:-1]):
            z = W @ self.cache[-1] + b
            self.cache.append(self.relu(z))
        # Output-Layer ohne Aktivierung (Regression) oder mit Softmax (Klassifikation)
        return self.W[-1] @ self.cache[-1] + self.b[-1]

    def backward(self, y_true, y_pred, lr=1e-3):
        # MSE-Gradient: dL/dŷ = 2(ŷ - y)
        delta = 2.0 * (y_pred - y_true)
        for i in reversed(range(len(self.W))):
            # Gewichts-Gradient: outer product von delta und Cache
            grad_W = np.outer(delta, self.cache[i])
            grad_b = delta
            # Gradient für vorherige Schicht via Kettenregel
            if i > 0:
                delta = (self.W[i].T @ delta) * self.relu_d(self.cache[i])
            # Gradient Descent Update
            self.W[i] -= lr * grad_W
            self.b[i] -= lr * grad_b

# Beispiel: Lerne y = sin(x) mit einem 1-32-32-1 Netz
net = MLP([1, 32, 32, 1])
losses = []

for epoch in range(3000):
    x = np.random.uniform(-np.pi, np.pi)
    y_true = np.sin(x)
    y_pred = net.forward(np.array([x]))
    net.backward(np.array([y_true]), y_pred, lr=5e-4)
    if epoch % 100 == 0:
        # Test auf einem Gitter
        test_x = np.linspace(-np.pi, np.pi, 50)
        mse = np.mean([(net.forward([xi]) - np.sin(xi))**2 for xi in test_x])
        losses.append(float(mse))
        print(f"Epoch {epoch:4d}: MSE = {mse:.6f}")

print("Training fertig! Das Netz hat sin(x) gelernt.")
        """, language="python")

        info_box(
            "**Das ist ein vollständiges, lauffähiges neuronales Netz — in 30 Zeilen NumPy.** "
            "PyTorch macht intern dasselbe, nur schneller (CUDA), sicherer (Autograd) und mit mehr Features.",
            kind="success",
        )

        st.markdown("#### In PyTorch — 10 Zeilen")
        st.code("""
import torch
import torch.nn as nn

# Gleiches Netz in PyTorch
model = nn.Sequential(
    nn.Linear(1, 32), nn.ReLU(),
    nn.Linear(32, 32), nn.ReLU(),
    nn.Linear(32, 1),
)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
loss_fn = nn.MSELoss()

for epoch in range(3000):
    x = torch.randn(64, 1) * torch.pi    # Batch von 64
    y = torch.sin(x)
    pred = model(x)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()                      # Autograd berechnet alle Gradienten
    optimizer.step()

print("Training fertig!")
        """, language="python")

    # ------------------------------------------------------------------ #
    with tabs[7]:
        section_header("Lernvideos", "3Blue1Brown: die besten Visualisierungen zu Neural Networks.")

        st.markdown("#### 1 — But what is a neural network? (3Blue1Brown)")
        video_embed("aircAruvnKk",
                    "But what is a neural network? — 3Blue1Brown",
                    "Grant Sanderson erklärt Neural Networks mit wunderschönen Animationen. ~19 Minuten. Pflicht.")

        divider()

        st.markdown("#### 2 — Gradient descent, how neural networks learn (3Blue1Brown)")
        video_embed("IHZwWFHWa-w",
                    "Gradient descent — 3Blue1Brown",
                    "Wie Gradientenabstieg funktioniert — mit echter Intuition. ~21 Minuten.")

        divider()

        st.markdown("#### 3 — What is backpropagation really doing? (3Blue1Brown)")
        video_embed("Ilg3gGewQ5U",
                    "Backpropagation — 3Blue1Brown",
                    "Die Kettenregel visuell erklärt. ~14 Minuten. Macht Backprop endlich verständlich.")

        info_box(
            "Nach diesen drei Videos hast du ein solides intuitives Verständnis von Neural Networks. "
            "Dann komm zurück und lies den mathematischen Teil nochmal — er wird viel klarer sein.",
            kind="tip",
        )

    # ------------------------------------------------------------------ #
    with tabs[8]:
        render_learning_block(
            key_prefix="nn_basics",
            section_title="Lernpfad für NN-Basics",
            progression=[
                ("🟢", "Guided Lab", "Ein MLP auf einem einfachen Datensatz trainieren und Kurven lesen.", "Beginner", "green"),
                ("🟠", "Challenge Lab", "XOR mit minimaler Architektur lösen und dokumentieren.", "Intermediate", "amber"),
                ("🔴", "Debug Lab", "Vanishing Gradients oder Dead ReLUs diagnostizieren.", "Advanced", "pink"),
                ("🏁", "Mini-Projekt", "Kleiner Klassifikator mit Fehleranalyse und sauberem Training-Log.", "Abschluss", "blue"),
            ],
            mcq_question="Warum sind Aktivierungsfunktionen in Hidden Layers unverzichtbar?",
            mcq_options=[
                "Nur damit das Modell schneller läuft",
                "Ohne sie bleibt das Netz linear, egal wie tief es ist",
                "Sie ersetzen den Optimizer",
                "Damit kein Loss benötigt wird",
            ],
            mcq_correct_option="Ohne sie bleibt das Netz linear, egal wie tief es ist",
            mcq_success_message="Richtig. Nichtlinearität ist die Kernidee tiefer Netze.",
            mcq_retry_message="Noch nicht korrekt. Prüfe den MLP-Abschnitt.",
            open_question="Offene Frage: Wann würdest du Leaky ReLU statt ReLU einsetzen?",
            code_task="""# Code-Aufgabe: einfache Forward-Funktion ergänzen
import numpy as np

def forward(x, W1, b1, W2, b2):
    # TODO: hidden = relu(x @ W1 + b1)
    # TODO: out = hidden @ W2 + b2
    return out
""",
            community_rows=[
                {"Format": "Diskussion", "Fokus": "Welche Aktivierung funktionierte bei dir stabil?", "Output": "Begründung"},
                {"Format": "Peer-Feedback", "Fokus": "Ist die Backprop-Intuition korrekt erklärt?", "Output": "Kurzkommentar"},
                {"Format": "Challenge", "Fokus": "Bestes XOR-Modell mit wenig Parametern", "Output": "Architektur + Ergebnis"},
            ],
            cheat_sheet=[
                "Perzeptron ist linear, MLP mit Aktivierungen ist nichtlinear.",
                "Backprop = Kettenregel über Layer.",
                "ReLU/Leaky ReLU sind starke Defaults.",
            ],
            key_takeaways=[
                "Tiefe + Nichtlinearität machen komplexe Funktionen lernbar.",
                "Trainingserfolg hängt von Aktivierung, Loss und LR zusammen.",
            ],
            common_errors=[
                "Falsche Output-Aktivierung.",
                "Loss passt nicht zur Aufgabe.",
                "Keine Feature-Skalierung.",
                "Zu hohe LR führt zu Instabilität.",
                "Keine Kontrolle der Gradienten.",
            ],
        )
