"""Neuronale Netze von Grund auf."""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.components import hero, section_header, divider, info_box


def render():
    hero(
        eyebrow="Deep Learning · Modul 11",
        title="Neuronale Netze von Grund auf",
        sub="Vom einzelnen Perzeptron zum Multi-Layer-Netz. Die fundamentale Architektur, "
            "auf der **alles** in modernem Deep Learning aufbaut."
    )

    tabs = st.tabs(["⚛️ Perzeptron", "🧠 MLP", "🎚️ Aktivierungen", "🔥 Universal Approx", "💻 Eigenes Netz"])

    with tabs[0]:
        section_header("Das Perzeptron — die einfachste Form")
        st.markdown(r"""
Ein **Perzeptron** ist ein einzelnes "Neuron":

$$y = \sigma(\mathbf{w}^\top \mathbf{x} + b)$$

- $\mathbf{x}$: Eingabe (Vektor)
- $\mathbf{w}$: Gewichte (was das Neuron gelernt hat)
- $b$: Bias (Verschiebung)
- $\sigma$: Aktivierungsfunktion (z.B. Sigmoid)

#### Was kann es?
Ein einzelnes Perzeptron kann nur **linear trennbare** Probleme lösen — z.B. AND, OR.
Es kann **kein** XOR. Das hat in den 70ern den ersten KI-Winter ausgelöst.

#### Lösung: Mehr Neuronen, mehr Schichten.
        """)

    with tabs[1]:
        section_header("Multi-Layer Perceptron (MLP)")
        st.markdown(r"""
Ein **MLP** stapelt mehrere Schichten von Neuronen:

$$\mathbf{h}^{(1)} = \sigma(\mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)})$$
$$\mathbf{h}^{(2)} = \sigma(\mathbf{W}^{(2)} \mathbf{h}^{(1)} + \mathbf{b}^{(2)})$$
$$\vdots$$
$$\mathbf{y} = \mathbf{W}^{(L)} \mathbf{h}^{(L-1)} + \mathbf{b}^{(L)}$$

#### Anatomie
- **Input Layer**: rohe Daten
- **Hidden Layers**: lernen Repräsentationen
- **Output Layer**: liefert Klassifikation oder Regression
- **Aktivierungen** zwischen Layern — sonst kollabiert alles zu einer einzigen linearen Operation
        """)

    with tabs[2]:
        section_header("Aktivierungsfunktionen")
        st.markdown(r"""
| Funktion | Formel | Eigenschaften |
|---|---|---|
| **Sigmoid** | $\sigma(x) = \frac{1}{1+e^{-x}}$ | Output in (0,1). Sättigt → Vanishing Gradient |
| **Tanh** | $\tanh(x)$ | Output in (-1,1). Zero-centered, aber sättigt auch |
| **ReLU** | $\max(0, x)$ | Standard. Schnell. Kann "sterben" |
| **Leaky ReLU** | $\max(0.01x, x)$ | Variante, die nicht stirbt |
| **GELU** | $x \cdot \Phi(x)$ | Standard in Transformers |
| **Swish/SiLU** | $x \cdot \sigma(x)$ | Glatte Variante von ReLU |
| **Softmax** | $\frac{e^{x_i}}{\sum_j e^{x_j}}$ | Output Layer für Multi-Class |
        """)

        # Plotten
        x = np.linspace(-5, 5, 200)
        funcs = {
            "Sigmoid":    1 / (1 + np.exp(-x)),
            "Tanh":       np.tanh(x),
            "ReLU":       np.maximum(0, x),
            "GELU":       0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))),
        }
        fig = go.Figure()
        colors = ["#7C3AED", "#EC4899", "#F59E0B", "#06B6D4"]
        for (name, y), c in zip(funcs.items(), colors):
            fig.add_trace(go.Scatter(x=x, y=y, name=name, line=dict(color=c, width=2.5)))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=380,
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        section_header("Universal Approximation Theorem")
        st.markdown(r"""
> Ein MLP mit **einer einzigen Hidden Layer** und genug Neuronen kann **jede stetige Funktion**
> auf einem kompakten Raum beliebig genau approximieren.

— Hornik, Cybenko (1989)

#### Warum dann tiefe Netze?
- Theorem sagt nichts über **Effizienz**.
- Tiefe Netze brauchen **exponentiell weniger Parameter** für viele realistische Funktionen.
- Tiefe Netze lernen **hierarchische Repräsentationen** (Kanten → Texturen → Teile → Objekte).
        """)
        info_box("Tiefe ≈ Komposition. Die Welt ist kompositionell — also passen tiefe Netze gut.", kind="tip")

    with tabs[4]:
        section_header("Eigenes Mini-MLP in 30 Zeilen")
        st.code("""
import numpy as np

class MLP:
    def __init__(self, sizes):
        self.W = [np.random.randn(o, i) * np.sqrt(2/i)
                  for i, o in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(o) for o in sizes[1:]]

    def relu(self, x): return np.maximum(0, x)
    def relu_d(self, x): return (x > 0).astype(float)

    def forward(self, x):
        self.cache = [x]
        for W, b in zip(self.W[:-1], self.b[:-1]):
            z = W @ self.cache[-1] + b
            self.cache.append(self.relu(z))
        # Output ohne Aktivierung
        return self.W[-1] @ self.cache[-1] + self.b[-1]

    def backward(self, y_true, y_pred, lr=1e-3):
        # MSE-Gradient
        delta = 2 * (y_pred - y_true)
        for i in reversed(range(len(self.W))):
            grad_W = np.outer(delta, self.cache[i])
            grad_b = delta
            if i > 0:
                delta = (self.W[i].T @ delta) * self.relu_d(self.cache[i])
            self.W[i] -= lr * grad_W
            self.b[i] -= lr * grad_b

# Beispiel: Lerne y = sin(x)
net = MLP([1, 32, 32, 1])
for epoch in range(2000):
    x = np.random.uniform(-3.14, 3.14)
    y = np.sin(x)
    pred = net.forward(np.array([x]))
    net.backward(np.array([y]), pred)
        """, language="python")
        info_box(
            "Das ist ein vollständiges, lauffähiges neuronales Netz. "
            "Du kannst damit echte Funktionen lernen. PyTorch macht im Kern dasselbe — nur schneller und mit GPU.",
            kind="success",
        )
