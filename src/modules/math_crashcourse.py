"""Mathe-Crashkurs — die wichtigsten Konzepte kompakt."""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.components import hero, section_header, divider, info_box, lab_header, step_list, video_embed


def render():
    hero(
        eyebrow="Grundlagen · Modul 1",
        title="Mathe-Crashkurs",
        sub="Die mathematische Sprache von CV und KI auf einen Blick. "
            "Keine Beweise, keine Trockenheit — nur das, was du wirklich brauchst."
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Notation", "🎯 Skalare → Tensoren", "🔢 Operationen",
        "🔄 Sortieralgorithmen", "📚 Cheatsheet",
    ])

    with tab1:
        section_header("Notation, die du immer wieder sehen wirst")
        st.markdown(r"""
| Symbol | Bedeutung | Beispiel |
|---|---|---|
| $x$ | Skalar | $x = 3.14$ |
| $\mathbf{x}$ | Vektor (klein, fett) | $\mathbf{x} \in \mathbb{R}^{n}$ |
| $\mathbf{X}$ | Matrix (groß, fett) | $\mathbf{X} \in \mathbb{R}^{m \times n}$ |
| $\mathcal{X}$ | Tensor / Menge | Bild-Tensor $\mathcal{X} \in \mathbb{R}^{H \times W \times C}$ |
| $\|\mathbf{x}\|_2$ | L2-Norm | $\sqrt{\sum_i x_i^2}$ |
| $\langle \mathbf{x}, \mathbf{y} \rangle$ | Skalarprodukt | $\sum_i x_i y_i$ |
| $\nabla f$ | Gradient | $(\partial f/\partial x_1, \dots)$ |
| $\mathbb{E}[X]$ | Erwartungswert | $\sum_x x \cdot p(x)$ |
""")

    with tab2:
        section_header("Vom Skalar zum Tensor")
        st.markdown(r"""
**Skalar** — eine Zahl. Pixelhelligkeit eines Graustufenbildes an Position (3, 5): $x = 127$.

**Vektor** — eine Liste. Ein RGB-Pixel: $\mathbf{p} = (210, 87, 34)$.

**Matrix** — eine 2D-Tabelle. Ein 8×8 Graustufenbild ist eine $\mathbf{X} \in \mathbb{R}^{8 \times 8}$.

**Tensor** — beliebige Dimension. Ein Farbbild ist ein 3D-Tensor:
$$\mathcal{X} \in \mathbb{R}^{H \times W \times 3}$$

Ein Batch von Bildern für ein neuronales Netz ist 4D:
$$\mathcal{B} \in \mathbb{R}^{N \times C \times H \times W}$$
        """)
        info_box(
            "Merke: PyTorch nutzt $(N, C, H, W)$ — Batch, Channels, Height, Width. "
            "TensorFlow nutzt $(N, H, W, C)$. Das ist der häufigste Bug-Grund beim Wechsel.",
            kind="warn",
        )

    with tab3:
        section_header("Operationen, die du verstehen musst")
        st.markdown(r"""
### Skalarprodukt
$$\mathbf{a} \cdot \mathbf{b} = \sum_i a_i b_i$$
Misst Ähnlichkeit zweier Vektoren. **Cosine Similarity** ist normalisiertes Skalarprodukt — Basis von CLIP & Co.

### Matrixmultiplikation
$$(\mathbf{A}\mathbf{B})_{ij} = \sum_k A_{ik} B_{kj}$$
**Das** Herzstück jedes neuronalen Netzes. Forward-Pass eines Layers: $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$.

### Transponieren
$\mathbf{A}^\top$ tauscht Zeilen und Spalten. Wichtig für: Self-Attention, Backprop.

### Outer Product
$$\mathbf{a} \otimes \mathbf{b} = \mathbf{a}\mathbf{b}^\top$$
Ergibt eine Matrix. Brauchst du, wenn du Attention-Maps verstehen willst.

### Norm
$$\|\mathbf{x}\|_p = \left(\sum_i |x_i|^p\right)^{1/p}$$
- $p=1$: L1-Norm (Manhattan), erzeugt sparse Lösungen
- $p=2$: L2-Norm (Euklidisch), Standard für Distanz
- $p=\infty$: Maximum

### Gradient
$$\nabla f(\mathbf{x}) = \begin{pmatrix} \partial f / \partial x_1 \\ \vdots \\ \partial f / \partial x_n \end{pmatrix}$$
Zeigt in die Richtung des steilsten Anstiegs. Gradient Descent geht $-\nabla f$ entlang.
        """)

    with tab4:
        section_header("Sortieralgorithmen — Theorie & Praxis")
        st.markdown(r"""
Sortieren ist eines der **fundamentalsten Probleme der Informatik** — und ein perfektes Lernfeld
für Algorithmus-Analyse. Warum hier, im Mathe-Modul? Weil **Laufzeitkomplexität** ($\mathcal{O}$-Notation)
reine Mathematik ist.
        """)

        section_header("Big-O Übersicht")
        st.markdown(r"""
| Algorithmus | Best Case | Average Case | Worst Case | Speicher | Stabil? |
|-------------|-----------|--------------|------------|----------|---------|
| **Bubble Sort** | $\mathcal{O}(n)$ | $\mathcal{O}(n^2)$ | $\mathcal{O}(n^2)$ | $\mathcal{O}(1)$ | ✅ |
| **Selection Sort** | $\mathcal{O}(n^2)$ | $\mathcal{O}(n^2)$ | $\mathcal{O}(n^2)$ | $\mathcal{O}(1)$ | ❌ |
| **Insertion Sort** | $\mathcal{O}(n)$ | $\mathcal{O}(n^2)$ | $\mathcal{O}(n^2)$ | $\mathcal{O}(1)$ | ✅ |
| **Merge Sort** | $\mathcal{O}(n \log n)$ | $\mathcal{O}(n \log n)$ | $\mathcal{O}(n \log n)$ | $\mathcal{O}(n)$ | ✅ |
| **Quick Sort** | $\mathcal{O}(n \log n)$ | $\mathcal{O}(n \log n)$ | $\mathcal{O}(n^2)$ | $\mathcal{O}(\log n)$ | ❌ |
| **Heap Sort** | $\mathcal{O}(n \log n)$ | $\mathcal{O}(n \log n)$ | $\mathcal{O}(n \log n)$ | $\mathcal{O}(1)$ | ❌ |
| **Tim Sort** *(Python built-in)* | $\mathcal{O}(n)$ | $\mathcal{O}(n \log n)$ | $\mathcal{O}(n \log n)$ | $\mathcal{O}(n)$ | ✅ |

> **Stabil** = gleiche Werte behalten ihre relative Reihenfolge (wichtig z.B. für Datenbanken).
        """)

        divider()

        algo = st.selectbox("Algorithmus anzeigen", [
            "Bubble Sort", "Selection Sort", "Insertion Sort",
            "Merge Sort", "Quick Sort", "Heap Sort",
        ])

        # ── Erklärungen + Code je Algorithmus ───────────────────────────────
        if algo == "Bubble Sort":
            st.markdown(r"""
#### Bubble Sort — der Klassiker
Vergleicht immer zwei benachbarte Elemente und tauscht sie, falls nötig.
Größte Elemente "blubbern" ans Ende. Einfach zu verstehen, langsam in der Praxis.

**Idee:** Jeder Pass bringt das größte verbleibende Element an seine finale Position.
Nach $k$ Passes sind die letzten $k$ Elemente korrekt.
            """)
            step_list([
                ("Vergleich", "arr[i] > arr[i+1]? Falls ja → tauschen."),
                ("Wiederholen", "Für jedes Element von links nach rechts."),
                ("Optimierung", "Abbruch, wenn kein Tausch stattgefunden hat (already sorted)."),
            ])
            st.code("""
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):   # letzte i Elemente bereits sortiert
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:   # Frühzeitiger Abbruch — schon sortiert
            break
    return arr

print(bubble_sort([64, 34, 25, 12, 22, 11, 90]))
# → [11, 12, 22, 25, 34, 64, 90]
            """, language="python")

        elif algo == "Selection Sort":
            st.markdown(r"""
#### Selection Sort — das Minimum suchen
Findet bei jedem Durchlauf das **Minimum** der noch nicht sortierten Elemente
und tauscht es an die richtige Position.

**Vorteil**: Genau $n-1$ Tauschoperationen — gut wenn Schreiben teuer ist.
**Nachteil**: Immer $\mathcal{O}(n^2)$ Vergleiche, auch bei fast-sortierten Daten.
            """)
            step_list([
                ("Minimum finden", "In arr[i..n-1] das kleinste Element suchen."),
                ("Tauschen", "Mit arr[i] tauschen."),
                ("i erhöhen", "i = i + 1 → nächste Position."),
            ])
            st.code("""
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]   # einziger Tausch pro Runde
    return arr

print(selection_sort([64, 25, 12, 22, 11]))
# → [11, 12, 22, 25, 64]
            """, language="python")

        elif algo == "Insertion Sort":
            st.markdown(r"""
#### Insertion Sort — wie Karten sortieren
Nimmt ein Element und **fügt es an die richtige Position** im bereits sortierten
linken Teil ein — wie beim Kartenspielen.

**Warum er trotz $\mathcal{O}(n^2)$ in der Praxis oft gut ist:**
- Best Case $\mathcal{O}(n)$ für fast-sortierte Daten
- Tim Sort (Pythons built-in) nutzt Insertion Sort für kleine Teilarrays
- Sehr cache-freundlich
            """)
            step_list([
                ("Schlüssel nehmen", "arr[i] als aktuelles Element."),
                ("Einordnen", "Solange arr[j] > key: arr[j+1] = arr[j], j--."),
                ("Einfügen", "arr[j+1] = key."),
            ])
            st.code("""
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:   # Platz schaffen
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key   # einfügen
    return arr

print(insertion_sort([12, 11, 13, 5, 6]))
# → [5, 6, 11, 12, 13]
            """, language="python")

        elif algo == "Merge Sort":
            st.markdown(r"""
#### Merge Sort — Divide & Conquer
**Teile** das Array in zwei Hälften, **sortiere** jede rekursiv,
**füge** die zwei sortierten Hälften zusammen.

Garantiert $\mathcal{O}(n \log n)$ — egal wie das Array aussieht.
Braucht aber $\mathcal{O}(n)$ extra Speicher für das Merge.

**Rekurrenzformel:** $T(n) = 2T(n/2) + \mathcal{O}(n)$ → Master Theorem: $\mathcal{O}(n \log n)$
            """)
            step_list([
                ("Divide", "Array in linke und rechte Hälfte aufteilen (Mitte)."),
                ("Conquer", "merge_sort(left), merge_sort(right) — rekursiv."),
                ("Merge", "Zwei sortierte Arrays zu einem sortierten zusammenführen."),
                ("Basisfall", "Array der Länge ≤ 1 ist bereits sortiert."),
            ])
            st.code("""
def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left  = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return _merge(left, right)

def _merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    return result + left[i:] + right[j:]

print(merge_sort([38, 27, 43, 3, 9, 82, 10]))
# → [3, 9, 10, 27, 38, 43, 82]
            """, language="python")

        elif algo == "Quick Sort":
            st.markdown(r"""
#### Quick Sort — der schnellste in der Praxis
Wählt ein **Pivot-Element**, **partitioniert** das Array so dass
links < Pivot < rechts steht, dann rekursiv weiter.

Average $\mathcal{O}(n \log n)$, Worst Case $\mathcal{O}(n^2)$ (schlechter Pivot, z.B. immer Minimum).
**Pivot-Strategien**: Erstes, Letztes, Mittleres, Random, Median-of-3.

In der Praxis schneller als Merge Sort — in-place, cache-freundlich.
            """)
            step_list([
                ("Pivot wählen", "Hier: letztes Element. Besser: zufällig oder Median-of-3."),
                ("Partitionieren", "Alle ≤ Pivot nach links, alle > Pivot nach rechts."),
                ("Rekursion", "quick_sort(links), quick_sort(rechts)."),
            ])
            st.code("""
def quick_sort(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    if low < high:
        pi = _partition(arr, low, high)
        quick_sort(arr, low, pi - 1)
        quick_sort(arr, pi + 1, high)
    return arr

def _partition(arr, low, high):
    pivot = arr[high]   # Pivot = letztes Element
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

print(quick_sort([10, 7, 8, 9, 1, 5]))
# → [1, 5, 7, 8, 9, 10]
            """, language="python")

        elif algo == "Heap Sort":
            st.markdown(r"""
#### Heap Sort — Heap-Datenstruktur
Nutzt einen **Max-Heap** (Binärbaum-Property: Elternteil ≥ Kinder).

1. **Build Max-Heap** aus dem Array — $\mathcal{O}(n)$
2. **Extract Max**: Wurzel (Maximum) mit letztem Element tauschen, Heap-Größe shrink, heapify — $n$ mal à $\mathcal{O}(\log n)$

Immer $\mathcal{O}(n \log n)$, in-place, aber nicht stabil und schlechte Cache-Performance.
            """)
            step_list([
                ("Heapify", "Teilbaum ab Index i als Max-Heap aufbauen."),
                ("Build Heap", "Alle Nicht-Blatt-Knoten von unten nach oben heapify."),
                ("Sort", "Max entnehmen → ans Ende → Heap um 1 verkleinern → heapify."),
            ])
            st.code("""
def heap_sort(arr):
    n = len(arr)

    # Max-Heap aufbauen
    for i in range(n // 2 - 1, -1, -1):
        _heapify(arr, n, i)

    # Element für Element ans Ende extrahieren
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]   # Max an Ende
        _heapify(arr, i, 0)               # Heap reparieren
    return arr

def _heapify(arr, n, i):
    largest = i
    left, right = 2 * i + 1, 2 * i + 2
    if left  < n and arr[left]  > arr[largest]: largest = left
    if right < n and arr[right] > arr[largest]: largest = right
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        _heapify(arr, n, largest)

print(heap_sort([12, 11, 13, 5, 6, 7]))
# → [5, 6, 7, 11, 12, 13]
            """, language="python")

        divider()

        # ── Interaktiver Sortierschritt-Visualizer ───────────────────────────
        lab_header("Schritt-für-Schritt Visualizer", "Sieh jeden Tausch live")

        col_l, col_r = st.columns([1, 2])
        with col_l:
            vis_algo = st.selectbox("Algorithmus", [
                "Bubble Sort", "Selection Sort", "Insertion Sort",
            ], key="vis_algo")
            arr_input = st.text_input(
                "Array (kommagetrennt)",
                value="5, 3, 8, 1, 9, 2, 7, 4, 6",
                key="vis_arr",
            )
            try:
                arr_raw = [int(x.strip()) for x in arr_input.split(",") if x.strip()]
                arr_raw = arr_raw[:12]  # max 12 Elemente
            except ValueError:
                arr_raw = [5, 3, 8, 1, 9, 2, 7, 4, 6]

        # Alle Sortierschritte vorab berechnen
        def steps_bubble(a):
            a = a[:]
            steps, highlights = [a[:]], [(-1, -1)]
            n = len(a)
            for i in range(n):
                for j in range(n - i - 1):
                    if a[j] > a[j + 1]:
                        a[j], a[j + 1] = a[j + 1], a[j]
                        steps.append(a[:])
                        highlights.append((j, j + 1))
            return steps, highlights

        def steps_selection(a):
            a = a[:]
            steps, highlights = [a[:]], [(-1, -1)]
            n = len(a)
            for i in range(n):
                min_idx = i
                for j in range(i + 1, n):
                    if a[j] < a[min_idx]:
                        min_idx = j
                if min_idx != i:
                    a[i], a[min_idx] = a[min_idx], a[i]
                    steps.append(a[:])
                    highlights.append((i, min_idx))
            return steps, highlights

        def steps_insertion(a):
            a = a[:]
            steps, highlights = [a[:]], [(-1, -1)]
            for i in range(1, len(a)):
                key = a[i]
                j = i - 1
                while j >= 0 and a[j] > key:
                    a[j + 1] = a[j]
                    j -= 1
                    a[j + 1] = key
                    steps.append(a[:])
                    highlights.append((j + 1, j + 2))
            return steps, highlights

        fn_map = {
            "Bubble Sort": steps_bubble,
            "Selection Sort": steps_selection,
            "Insertion Sort": steps_insertion,
        }
        all_steps, all_hl = fn_map[vis_algo](arr_raw)

        with col_l:
            step_idx = st.slider("Schritt", 0, len(all_steps) - 1, 0, key="vis_step")
            st.caption(f"Schritt {step_idx} / {len(all_steps) - 1} · {len(all_steps) - 1} Tauschoperationen")

        current = all_steps[step_idx]
        hl_a, hl_b = all_hl[step_idx]
        colors = []
        for k in range(len(current)):
            if k == hl_a or k == hl_b:
                colors.append("#EC4899")   # getauschte Elemente — pink
            elif current[k] == sorted(arr_raw)[k]:
                colors.append("#10B981")   # final sortiert — grün
            else:
                colors.append("#7C3AED")   # normal — lila

        with col_r:
            fig = go.Figure(go.Bar(
                x=list(range(len(current))),
                y=current,
                marker_color=colors,
                text=current,
                textposition="outside",
            ))
            fig.update_layout(
                template="plotly_dark",
                height=320,
                showlegend=False,
                xaxis=dict(showticklabels=False),
                yaxis=dict(showgrid=False, showticklabels=False),
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="#0F172A",
                paper_bgcolor="#0F172A",
            )
            st.plotly_chart(fig, use_container_width=True)

        if hl_a >= 0:
            st.caption(f"🔄 Tausch: Index {hl_a} ({current[hl_a]}) ↔ Index {hl_b} ({current[hl_b]})")
        else:
            st.caption("📋 Ausgangszustand")

        divider()
        section_header("Wann welchen Algorithmus?")
        st.markdown(r"""
| Situation | Empfehlung |
|-----------|-----------|
| Kleines Array (< 20 Elemente) | Insertion Sort |
| Fast sortiertes Array | Insertion Sort oder Tim Sort |
| Sortierung im RAM, Geschwindigkeit wichtig | Quick Sort (mit Randomisierung) |
| Garantiertes $\mathcal{O}(n \log n)$ nötig | Merge Sort oder Heap Sort |
| Stabiles Sortieren nötig | Merge Sort oder Tim Sort |
| Einfach in Python | `sorted()` oder `list.sort()` — Tim Sort, immer richtig |
| Externe Sortierung (Dateien zu groß für RAM) | Merge Sort (chunked) |
        """)
        info_box(
            "In Python immer zuerst `sorted()` oder `.sort()` verwenden — Tim Sort ist "
            "hochoptimiert (Hybrid aus Merge + Insertion Sort) und schlägt naive Implementierungen "
            "in der Praxis um Faktoren. Eigene Implementierung nur zum Lernen!",
            kind="tip",
        )

        divider()
        section_header("Lernvideos")
        video_embed("kgBjXUE_Vre", "Sorting Algorithms — CS50", "Harvard CS50: alle Sortieralgorithmen visuell erklärt")
        st.markdown("---")
        video_embed("ZZuD6iUe3Pc", "Sorting Algorithms Visualized", "Visueller Vergleich aller Algorithmen — Bubble, Merge, Quick, Heap")

    with tab5:
        section_header("Cheatsheet: Was wofür?")
        st.markdown("""
| Du willst... | Brauchst... | Modul |
|---|---|---|
| Bilder als Daten verstehen | Tensoren, Matrixops | Lineare Algebra |
| NNs trainieren | Gradienten, Kettenregel | Analysis |
| Klassifikation verstehen | Cross-Entropy, Softmax | Wahrscheinlichkeit |
| CNNs verstehen | Faltung als Operation | Lineare Algebra + Filter |
| Attention verstehen | Skalarprodukte + Softmax | Lineare Algebra |
| Diffusion verstehen | SDEs, Wahrscheinlichkeit | Wahrscheinlichkeit (advanced) |
| Backprop selbst rechnen | Jacobi-Matrizen | Analysis (advanced) |
""")

    divider()
    info_box(
        "Lass dich nicht von der Notation einschüchtern. Lies die Formeln laut vor, "
        "übersetze sie in Worte. Mit der Zeit wird Mathe zu einer zweiten Sprache.",
        kind="tip",
    )
