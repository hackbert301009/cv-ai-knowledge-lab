"""Tensor Playground — NumPy-Operationen live ausprobieren."""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.components import (
    hero, section_header, divider, info_box,
    lab_header, key_concept, card, render_card_grid,
)


def _shape_str(arr):
    return str(arr.shape)


def _show_tensor(arr, label="Tensor", max_rows=8):
    """Zeigt Tensor als formatierte Tabelle oder Text."""
    st.markdown(f"**{label}** — Shape: `{arr.shape}`, dtype: `{arr.dtype}`")
    flat = arr.flatten()
    st.markdown(
        f"min={flat.min():.3f} | max={flat.max():.3f} | "
        f"mean={flat.mean():.3f} | std={flat.std():.3f}"
    )
    if arr.ndim <= 2:
        if arr.size <= 200:
            st.dataframe(
                arr.reshape(-1, arr.shape[-1] if arr.ndim > 1 else 1),
                use_container_width=True,
                hide_index=False,
            )
        else:
            st.code(np.array2string(arr, precision=3, suppress_small=True,
                                    threshold=64), language="text")
    elif arr.ndim == 3:
        st.code(f"3D-Tensor [{arr.shape[0]} × {arr.shape[1]} × {arr.shape[2]}]\n"
                + np.array2string(arr[0], precision=3, suppress_small=True, threshold=32)
                + f"\n... (nur Slice [0] gezeigt)", language="text")
    else:
        st.code(f"Shape: {arr.shape}\n"
                + np.array2string(arr.flatten()[:16], precision=3, suppress_small=True)
                + " ...", language="text")


def _heatmap(arr, title=""):
    z = arr if arr.ndim == 2 else arr[0] if arr.ndim == 3 else arr.reshape(1, -1)
    fig = go.Figure(go.Heatmap(
        z=z, colorscale=[[0, "#0B0B0F"], [0.5, "#7C3AED"], [1, "#F59E0B"]],
        showscale=True, colorbar=dict(thickness=10, len=0.8),
    ))
    fig.update_layout(
        title=title, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=260, margin=dict(l=10, r=40, t=30, b=10),
        xaxis=dict(title="Spalte"), yaxis=dict(title="Zeile", autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render():
    hero(
        eyebrow="Praxis · Interaktives Tool",
        title="Tensor Playground",
        sub="NumPy-Tensor-Operationen live ausprobieren — reshape, einsum, broadcasting, matmul. "
            "Verstehe, was im Forward-Pass wirklich passiert."
    )

    tabs = st.tabs([
        "🔢 Erstelle Tensoren",
        "🔄 Reshape & Transpose",
        "➕ Broadcasting",
        "✖️ Matmul & Einsum",
        "🔍 Slicing & Indexing",
        "📊 Statistiken",
        "🧠 NN Forward Pass",
        "📖 Spickzettel",
    ])

    # ------------------------------------------------------------ #
    with tabs[0]:
        lab_header("Tensor erstellen", "Wähle Form und Initialisierung.")

        c1, c2, c3 = st.columns(3)
        init_type = c1.selectbox("Initialisierung", [
            "Zufällig (Normal)", "Zufällig (Uniform)", "Nullen", "Einsen",
            "Identität", "Arange", "Linspace",
        ])
        seed = c2.number_input("Seed", 0, 999, 42)
        dtype_choice = c3.selectbox("dtype", ["float32", "float64", "int32", "int64"])
        dtype_map = {"float32": np.float32, "float64": np.float64,
                     "int32": np.int32, "int64": np.int64}
        dtype = dtype_map[dtype_choice]

        shape_str = st.text_input("Shape (kommagetrennt)", "3, 4")
        try:
            shape = tuple(int(s.strip()) for s in shape_str.split(",") if s.strip())
        except ValueError:
            st.error("Ungültige Shape — Beispiel: 3, 4 oder 2, 3, 4")
            shape = (3, 4)

        rng = np.random.default_rng(seed)
        if init_type == "Zufällig (Normal)":
            T = rng.standard_normal(shape).astype(dtype)
        elif init_type == "Zufällig (Uniform)":
            T = rng.uniform(-1, 1, shape).astype(dtype)
        elif init_type == "Nullen":
            T = np.zeros(shape, dtype=dtype)
        elif init_type == "Einsen":
            T = np.ones(shape, dtype=dtype)
        elif init_type == "Identität":
            n = shape[0]
            T = np.eye(n, dtype=dtype)
        elif init_type == "Arange":
            T = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
        else:
            T = np.linspace(0, 1, np.prod(shape), dtype=dtype).reshape(shape)

        _show_tensor(T, "Dein Tensor T")
        if T.ndim == 2:
            _heatmap(T, "Heatmap")

        st.session_state["playground_T"] = T

        st.markdown("#### Generierter NumPy-Code")
        code_map = {
            "Zufällig (Normal)": f"T = np.random.randn{shape}.astype(np.{dtype_choice})",
            "Zufällig (Uniform)": f"T = np.random.uniform(-1, 1, {shape}).astype(np.{dtype_choice})",
            "Nullen": f"T = np.zeros({shape}, dtype=np.{dtype_choice})",
            "Einsen": f"T = np.ones({shape}, dtype=np.{dtype_choice})",
            "Identität": f"T = np.eye({shape[0]}, dtype=np.{dtype_choice})",
            "Arange": f"T = np.arange({np.prod(shape)}).reshape{shape}.astype(np.{dtype_choice})",
            "Linspace": f"T = np.linspace(0, 1, {np.prod(shape)}).reshape{shape}.astype(np.{dtype_choice})",
        }
        st.code(code_map.get(init_type, ""), language="python")

    # ------------------------------------------------------------ #
    with tabs[1]:
        lab_header("Reshape & Transpose", "Dimensionen ändern ohne Daten zu kopieren.")

        T = st.session_state.get("playground_T", np.arange(12).reshape(3, 4).astype(np.float32))
        st.markdown(f"**Eingabe T** — Shape: `{T.shape}`")

        op = st.selectbox("Operation", [
            "reshape", "flatten", "ravel", "transpose / T",
            "squeeze", "expand_dims (unsqueeze)", "swapaxes",
        ])

        try:
            if op == "reshape":
                new_shape_s = st.text_input("Neue Shape", f"{np.prod(T.shape)}")
                new_shape = tuple(int(x.strip()) for x in new_shape_s.split(","))
                result = T.reshape(new_shape)
                code = f"T.reshape({new_shape})"
            elif op == "flatten":
                result = T.flatten()
                code = "T.flatten()  # Kopie"
            elif op == "ravel":
                result = T.ravel()
                code = "T.ravel()  # View wenn möglich"
            elif op == "transpose / T":
                result = T.T
                code = "T.T  # oder np.transpose(T)"
            elif op == "squeeze":
                ex = np.expand_dims(T, axis=0)
                result = np.squeeze(ex)
                code = f"np.squeeze(T)  # Entfernt Dim der Größe 1\n# Input hatte Shape {ex.shape}"
            elif op == "expand_dims (unsqueeze)":
                ax = st.slider("Achse", 0, T.ndim, 0)
                result = np.expand_dims(T, axis=ax)
                code = f"np.expand_dims(T, axis={ax})"
            else:
                a0 = st.slider("Achse 1", 0, max(T.ndim - 1, 1), 0)
                a1 = st.slider("Achse 2", 0, max(T.ndim - 1, 1), min(1, T.ndim - 1))
                result = np.swapaxes(T, a0, a1)
                code = f"np.swapaxes(T, {a0}, {a1})"

            c1, c2 = st.columns(2)
            with c1:
                _show_tensor(T, "Vorher")
            with c2:
                _show_tensor(result, "Nachher")
            st.code(f"result = {code}", language="python")

        except Exception as e:
            st.error(f"Fehler: {e}")

    # ------------------------------------------------------------ #
    with tabs[2]:
        lab_header("Broadcasting", "NumPys mächtigste (und verwirrendste) Funktion.")
        st.markdown(r"""
**Broadcasting-Regeln** (von rechts nach links):
1. Dimensionen werden von rechts ausgerichtet
2. Fehlende Dimensionen werden als 1 angesehen
3. Dimensionen der Größe 1 werden auf die Größe des anderen Arrays "gestreckt"
        """)

        bc1, bc2 = st.columns(2)
        shape_a = bc1.text_input("Shape A", "3, 4")
        shape_b = bc2.text_input("Shape B", "4")

        try:
            sa = tuple(int(x.strip()) for x in shape_a.split(","))
            sb = tuple(int(x.strip()) for x in shape_b.split(","))
            A = np.ones(sa)
            B = np.ones(sb)

            try:
                C = A + B
                result_shape = C.shape

                st.markdown(
                    f"""<div style="background:rgba(16,185,129,0.1);border:1px solid #10B981;
                        border-radius:10px;padding:1rem;margin:1rem 0;">
                      <div style="font-weight:700;color:#6EE7B7;">✅ Broadcasting erfolgreich!</div>
                      <div style="color:#E5E7EB;margin-top:0.5rem;">
                        <code>{sa}</code> + <code>{sb}</code> → <code>{result_shape}</code>
                      </div>
                    </div>""",
                    unsafe_allow_html=True,
                )

                # Broadcasting-Schritte visualisieren
                max_dims = max(len(sa), len(sb))
                sa_padded = (1,) * (max_dims - len(sa)) + sa
                sb_padded = (1,) * (max_dims - len(sb)) + sb

                st.markdown("#### Broadcasting-Schritte")
                rows = []
                for i, (da, db) in enumerate(zip(sa_padded, sb_padded)):
                    bcast = max(da, db)
                    a_str = f"`{da}`" + (" → `1 gestreckt zu " + str(bcast) + "`" if da == 1 and db != 1 else "")
                    b_str = f"`{db}`" + (" → `1 gestreckt zu " + str(bcast) + "`" if db == 1 and da != 1 else "")
                    rows.append(f"| Dim {i} | {a_str} | {b_str} | **{bcast}** |")

                st.markdown("| Dim | A | B | Resultat |")
                st.markdown("|---|---|---|---|")
                for r in rows:
                    st.markdown(r)

            except ValueError as e:
                st.markdown(
                    f"""<div style="background:rgba(239,68,68,0.1);border:1px solid #EF4444;
                        border-radius:10px;padding:1rem;margin:1rem 0;">
                      <div style="font-weight:700;color:#FCA5A5;">❌ Broadcasting nicht möglich</div>
                      <div style="color:#9CA3AF;margin-top:0.5rem;">{e}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

        except Exception as e:
            st.error(f"Ungültige Shape: {e}")

        divider()
        st.markdown("#### Broadcasting in der Praxis")
        st.code("""
import numpy as np

# Bild normalisieren (sehr häufig in CV!)
img   = np.random.randint(0, 256, (224, 224, 3))  # Shape: [224, 224, 3]
mean  = np.array([0.485, 0.456, 0.406])            # Shape: [3]
std   = np.array([0.229, 0.224, 0.225])            # Shape: [3]

# Broadcasting: [224,224,3] - [3] → [224,224,3]
normalized = (img / 255.0 - mean) / std            # kein Schleifen nötig!

# Batch-Bias addieren
logits = np.random.randn(32, 10)  # [Batch, Klassen]
bias   = np.random.randn(10)      # [Klassen]
out    = logits + bias            # [32,10] + [10] → [32,10]

# Äußeres Produkt via Broadcasting
a = np.array([1, 2, 3])[:, None]  # [3, 1]
b = np.array([10, 20, 30])[None, :]  # [1, 3]
outer = a * b                        # [3, 3] — Äußeres Produkt!
        """, language="python")

    # ------------------------------------------------------------ #
    with tabs[3]:
        lab_header("Matmul & Einsum", "Matrixmultiplikation und das mächtige einsum.")

        op2 = st.selectbox("Operation", ["@ (matmul)", "np.dot", "np.einsum"])
        rng2 = np.random.default_rng(0)

        if op2 in ["@ (matmul)", "np.dot"]:
            c1, c2 = st.columns(2)
            m = c1.slider("m", 2, 8, 3)
            n = c2.slider("n", 2, 8, 4)
            k = st.slider("k", 2, 8, 5)
            A = rng2.standard_normal((m, n)).round(2)
            B = rng2.standard_normal((n, k)).round(2)
            C = A @ B

            col1, col2, col3 = st.columns(3)
            with col1:
                _show_tensor(A, f"A ({m}×{n})")
            col2.markdown(f"<div style='text-align:center;font-size:2rem;padding-top:3rem;'>@</div>",
                          unsafe_allow_html=True)
            with col2:
                _show_tensor(B, f"B ({n}×{k})")
            col3.markdown(f"<div style='text-align:center;font-size:2rem;padding-top:3rem;'>=</div>",
                          unsafe_allow_html=True)
            with col3:
                _show_tensor(C, f"C ({m}×{k})")

            st.code(f"""
A = np.random.randn({m}, {n})
B = np.random.randn({n}, {k})
C = A @ B   # Shape: ({m}, {k})
# Jedes C[i,j] = Skalarprodukt von Zeile i aus A und Spalte j aus B
            """, language="python")

        else:
            st.markdown("#### np.einsum — die universelle Notation")
            st.markdown(r"""
`einsum` beschreibt Operationen durch Buchstaben für jede Dimension.
Einer der mächtigsten NumPy-Befehle — lohnt sich zu lernen!
            """)
            einsum_examples = {
                "'ij,jk->ik' (Matmul)": ("ij,jk->ik", (3, 4), (4, 5)),
                "'ij->ji' (Transpose)": ("ij->ji", (3, 4), None),
                "'ii' (Trace)": ("ii", (4, 4), None),
                "'ij,ij->ij' (Elementweise)": ("ij,ij->ij", (3, 4), (3, 4)),
                "'ij,ij->' (Frobenius-Skalar)": ("ij,ij->", (3, 4), (3, 4)),
                "'bik,bkj->bij' (Batched Matmul)": ("bik,bkj->bij", (2, 3, 4), (2, 4, 5)),
                "'bhqk,bhkd->bhqd' (Attention)": ("bhqk,bhkd->bhqd", (2, 8, 6, 6), (2, 8, 6, 64)),
            }
            choice = st.selectbox("Beispiel", list(einsum_examples.keys()))
            expr, sha, shb = einsum_examples[choice]

            A = rng2.standard_normal(sha).round(2)
            B = rng2.standard_normal(shb).round(2) if shb else None

            try:
                result = np.einsum(expr, A) if B is None else np.einsum(expr, A, B)
                st.code(f"np.einsum('{expr}', A{', B' if B is not None else ''})\n"
                        f"A.shape={sha}{'  B.shape='+str(shb) if shb else ''} → {result.shape}",
                        language="python")
                _show_tensor(result, f"Ergebnis: {result.shape}")
            except Exception as e:
                st.error(str(e))

    # ------------------------------------------------------------ #
    with tabs[4]:
        lab_header("Slicing & Fancy Indexing")
        T2 = np.arange(24).reshape(4, 6).astype(np.float32)
        _show_tensor(T2, "Basis-Array T (4×6)")
        _heatmap(T2, "Basis-Array")

        slicing_examples = {
            "T[1, 3]       — Einzelner Wert": T2[1, 3:4],
            "T[0:2, :]     — Erste 2 Zeilen": T2[0:2, :],
            "T[:, 2:5]     — Spalten 2-4": T2[:, 2:5],
            "T[::2, :]     — Jede 2. Zeile": T2[::2, :],
            "T[[0,2,3], :] — Fancy (Zeilen 0,2,3)": T2[[0, 2, 3], :],
            "T[T > 12]     — Boolean Mask": T2[T2 > 12].reshape(1, -1),
        }
        choice2 = st.selectbox("Slicing-Beispiel", list(slicing_examples.keys()))
        result2 = slicing_examples[choice2]
        _show_tensor(result2, f"Ergebnis: {result2.shape}")

        st.code(f"result = {choice2.split('—')[0].strip()}", language="python")

    # ------------------------------------------------------------ #
    with tabs[5]:
        lab_header("Statistiken & Reduction-Operationen")

        T3 = st.session_state.get("playground_T", np.random.randn(5, 6).astype(np.float32))
        st.markdown(f"**Tensor** — Shape: `{T3.shape}`")
        if T3.ndim == 2:
            _heatmap(T3, "Tensor")

        ops = {
            "np.mean(T)": np.mean(T3),
            "np.mean(T, axis=0)": np.mean(T3, axis=0),
            "np.mean(T, axis=1)": np.mean(T3, axis=-1),
            "np.std(T)": np.std(T3),
            "np.min(T)": np.min(T3),
            "np.max(T)": np.max(T3),
            "np.argmax(T)": np.argmax(T3),
            "np.sum(T)": np.sum(T3),
            "np.cumsum(T.flatten())[:6]": np.cumsum(T3.flatten())[:6],
        }
        st.markdown("| Operation | Ergebnis | Shape |")
        st.markdown("|---|---|---|")
        for name, val in ops.items():
            v = np.atleast_1d(val)
            st.markdown(f"| `{name}` | `{np.round(v[:4], 3)}{'...' if len(v)>4 else ''}` | `{v.shape}` |")

    # ------------------------------------------------------------ #
    with tabs[6]:
        lab_header("Simulierter NN Forward Pass", "Was passiert intern wenn du model(x) aufrufst?")
        st.markdown("Ein MLP Forward Pass in reinem NumPy — Schicht für Schicht sichtbar.")

        rng3 = np.random.default_rng(7)
        batch = st.slider("Batch Size", 1, 8, 4)
        n_in, n_h1, n_h2, n_out = 8, 16, 8, 4

        x   = rng3.standard_normal((batch, n_in)).astype(np.float32)
        W1  = rng3.standard_normal((n_in, n_h1)).astype(np.float32) * np.sqrt(2/n_in)
        b1  = np.zeros(n_h1, dtype=np.float32)
        W2  = rng3.standard_normal((n_h1, n_h2)).astype(np.float32) * np.sqrt(2/n_h1)
        b2  = np.zeros(n_h2, dtype=np.float32)
        W3  = rng3.standard_normal((n_h2, n_out)).astype(np.float32) * np.sqrt(2/n_h2)
        b3  = np.zeros(n_out, dtype=np.float32)

        z1  = x @ W1 + b1
        h1  = np.maximum(0, z1)
        z2  = h1 @ W2 + b2
        h2  = np.maximum(0, z2)
        z3  = h2 @ W3 + b3
        exp = np.exp(z3 - z3.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)

        steps_fp = [
            (f"Input x", x, f"[{batch}, {n_in}]"),
            (f"Linear 1: z1 = x @ W1 + b1", z1, f"[{batch}, {n_h1}]"),
            (f"ReLU: h1 = max(0, z1)", h1, f"[{batch}, {n_h1}]"),
            (f"Linear 2: z2 = h1 @ W2 + b2", z2, f"[{batch}, {n_h2}]"),
            (f"ReLU: h2 = max(0, z2)", h2, f"[{batch}, {n_h2}]"),
            (f"Output: z3 = h2 @ W3 + b3", z3, f"[{batch}, {n_out}]"),
            (f"Softmax: probs", probs, f"[{batch}, {n_out}]"),
        ]
        for name, arr, shape_s in steps_fp:
            st.markdown(f"**{name}** → Shape `{shape_s}`")
            if arr.shape[1] <= 20:
                _heatmap(arr.T, "")

    # ------------------------------------------------------------ #
    with tabs[7]:
        section_header("NumPy Spickzettel", "Die wichtigsten Operationen auf einen Blick.")
        st.code("""
import numpy as np

# --- Erstellen ---
np.zeros((3, 4))              # Nullen
np.ones((3, 4))               # Einsen
np.eye(4)                     # Identitätsmatrix
np.arange(12).reshape(3, 4)  # 0..11 als Matrix
np.linspace(0, 1, 100)       # 100 Werte von 0 bis 1
np.random.randn(3, 4)         # Normal verteilt
np.random.uniform(0, 1, (3,4)) # Uniform [0,1)

# --- Shape ---
x.shape           # Tuple der Dimensionen
x.ndim            # Anzahl Dimensionen
x.size            # Gesamtanzahl Elemente
x.reshape(a, b)   # Neue Form (Daten unverändert)
x.flatten()       # 1D Kopie
x.ravel()         # 1D View (wenn möglich)
x.T               # Transpose
np.expand_dims(x, 0)  # Neue Dim vorne
x.squeeze()            # Dims der Größe 1 entfernen

# --- Rechnen ---
x + y, x - y, x * y  # Elementweise
x @ y                  # Matmul (2D) / Batched (3D+)
np.dot(x, y)           # Skalar- oder Matrixprodukt
np.einsum('ij,jk->ik', A, B)  # Universell

# --- Statistik ---
x.mean(), x.mean(axis=0)  # Mittelwert gesamt / pro Spalte
x.std(), x.var()
x.min(), x.max()
x.argmin(), x.argmax()    # Index des Min/Max
x.sum(), x.cumsum()

# --- Slicing ---
x[2, 3]         # Einzelwert
x[0:3, :]       # Erste 3 Zeilen
x[:, ::2]       # Jede 2. Spalte
x[[0,2], :]     # Fancy Indexing
x[x > 0]        # Boolean Mask

# --- Kombinieren ---
np.concatenate([a, b], axis=0)  # Verbinden
np.stack([a, b], axis=0)         # Neue Dim
np.split(x, 3, axis=0)          # Aufteilen

# --- PyTorch Entsprechungen ---
# np.zeros     → torch.zeros
# x.reshape    → x.view oder x.reshape
# x @ y        → x @ y  (gleich!)
# x.T          → x.T oder x.permute(1,0)
# np.einsum    → torch.einsum  (gleiche Syntax!)
        """, language="python")
