"""Analysis & Gradienten — die Sprache des Lernens."""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.components import hero, section_header, divider, info_box


def render():
    hero(
        eyebrow="Grundlagen · Modul 3",
        title="Analysis &amp; Gradienten",
        sub="Wie ein neuronales Netz lernt: Ableitungen, Kettenregel, Gradient Descent. "
            "Die Mathematik hinter dem 'Lernen' in Machine Learning."
    )

    tabs = st.tabs(["📈 Ableitung", "🌐 Gradient", "⛓️ Kettenregel", "⬇️ Gradient Descent", "🔬 Backprop"])

    # ---------- Ableitung ----------
    with tabs[0]:
        section_header("Die Ableitung — Änderungsrate")
        st.markdown(r"""
Die Ableitung $f'(x)$ einer Funktion misst, wie sich $f$ ändert, wenn man $x$ ein bisschen ändert:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

#### In CV begegnet sie dir als Bildgradient
Ein **Bild-Gradient** ist die Ableitung der Helligkeit in Richtung $x$ oder $y$:

$$G_x = \frac{\partial I}{\partial x}, \quad G_y = \frac{\partial I}{\partial y}$$

Hohe Gradienten ⇒ **Kanten**. Genau das messen Sobel-/Canny-Filter.
        """)

        # Visualisierung: f(x) und f'(x)
        x = np.linspace(-3, 3, 200)
        f = x**3 - 3*x
        df = 3*x**2 - 3
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=f, name="f(x) = x³ − 3x", line=dict(color="#7C3AED", width=3)))
        fig.add_trace(go.Scatter(x=x, y=df, name="f'(x) = 3x² − 3", line=dict(color="#EC4899", width=3)))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=400, title="Funktion und ihre Ableitung")
        st.plotly_chart(fig, use_container_width=True)

    # ---------- Gradient ----------
    with tabs[1]:
        section_header("Der Gradient — Ableitung in mehreren Dimensionen")
        st.markdown(r"""
Bei einer Funktion $f: \mathbb{R}^n \to \mathbb{R}$ ist der **Gradient** der Vektor aller partiellen Ableitungen:

$$\nabla f(\mathbf{x}) = \begin{pmatrix} \partial f / \partial x_1 \\ \vdots \\ \partial f / \partial x_n \end{pmatrix}$$

#### Eigenschaften
- Zeigt in die Richtung des **steilsten Anstiegs**.
- Sein Betrag ist die Steilheit dort.
- Senkrecht zu Höhenlinien (Niveaulinien).

#### Was lernt ein Netz wirklich?
Die Verlustfunktion $\mathcal{L}(\theta)$ hängt von Millionen Parametern $\theta$ ab.
Wir berechnen $\nabla_\theta \mathcal{L}$ und gehen einen kleinen Schritt **gegen** die Richtung
des Gradienten — das ist Gradient Descent.

$$\theta_{\text{neu}} = \theta_{\text{alt}} - \eta \nabla_\theta \mathcal{L}$$

$\eta$ ist die **Lernrate**.
        """)

    # ---------- Kettenregel ----------
    with tabs[2]:
        section_header("Kettenregel — das Geheimnis von Backprop")
        st.markdown(r"""
Wenn $z = f(g(x))$, dann:

$$\frac{dz}{dx} = \frac{dz}{dg} \cdot \frac{dg}{dx}$$

In neuronalen Netzen sind alle Layer **verkettet**:

$$\mathcal{L} = L(\sigma_3(\mathbf{W}_3 \sigma_2(\mathbf{W}_2 \sigma_1(\mathbf{W}_1 \mathbf{x}))))$$

Um $\partial \mathcal{L} / \partial \mathbf{W}_1$ zu berechnen, hagelt es Kettenregel —
**rückwärts** durch das Netzwerk. Das ist **Backpropagation**.
        """)
        info_box(
            "Backpropagation ist nichts Magisches — es ist die Kettenregel, dynamisch programmiert. "
            "Frameworks wie PyTorch erledigen das automatisch (autograd).",
            kind="tip",
        )

    # ---------- Gradient Descent ----------
    with tabs[3]:
        section_header("Gradient Descent — interaktiv")
        st.markdown("Beobachte, wie ein Punkt einen Funktionsverlauf 'hinunterrutscht'. Wähle Startpunkt und Lernrate:")

        c1, c2 = st.columns(2)
        x0 = c1.slider("Startpunkt x₀", -3.0, 3.0, 2.5, 0.1, key="gd_x0")
        lr = c2.slider("Lernrate η",     0.01, 0.5, 0.1, 0.01, key="gd_lr")

        # f(x) = x^4 - 3x^2 + 2x  -> f'(x) = 4x^3 - 6x + 2
        def f(x):  return x**4 - 3*x**2 + 2*x
        def df(x): return 4*x**3 - 6*x + 2

        path = [x0]
        x = x0
        for _ in range(30):
            x -= lr * df(x)
            path.append(x)

        xs = np.linspace(-2.5, 2.5, 200)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=f(xs), name="f(x)", line=dict(color="#7C3AED", width=3)))
        fig.add_trace(go.Scatter(x=path, y=[f(p) for p in path],
                                 mode="markers+lines", name="Gradient Descent",
                                 line=dict(color="#EC4899", width=2),
                                 marker=dict(size=6)))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=420)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"Endposition nach 30 Schritten: **x = {path[-1]:.3f}**, f(x) = **{f(path[-1]):.3f}**")
        if lr > 0.3:
            info_box("Hohe Lernrate — beobachte, ob der Punkt überschwingt oder divergiert.", kind="warn")

    # ---------- Backprop ----------
    with tabs[4]:
        section_header("Backpropagation — Schritt für Schritt")
        st.markdown(r"""
Ein winziges Netz: $y = \sigma(w_2 \cdot \sigma(w_1 \cdot x))$ mit Verlust $\mathcal{L} = (y - t)^2$.

**Forward:**
1. $a_1 = w_1 \cdot x$
2. $h_1 = \sigma(a_1)$
3. $a_2 = w_2 \cdot h_1$
4. $y = \sigma(a_2)$
5. $\mathcal{L} = (y - t)^2$

**Backward (Kettenregel rückwärts):**
$$\frac{\partial \mathcal{L}}{\partial w_2} = \underbrace{2(y-t)}_{\partial \mathcal{L}/\partial y} \cdot \underbrace{\sigma'(a_2)}_{\partial y/\partial a_2} \cdot \underbrace{h_1}_{\partial a_2/\partial w_2}$$

Genau das macht `loss.backward()` in PyTorch — automatisch, für **alle** Parameter.
        """)
        st.code("""
import torch

x = torch.tensor([1.0])
t = torch.tensor([0.5])
w1 = torch.tensor([0.8], requires_grad=True)
w2 = torch.tensor([0.3], requires_grad=True)

# Forward
h1 = torch.sigmoid(w1 * x)
y  = torch.sigmoid(w2 * h1)
loss = (y - t)**2

# Backward
loss.backward()

print(w1.grad, w2.grad)   # Gradienten — automatisch berechnet
        """, language="python")
