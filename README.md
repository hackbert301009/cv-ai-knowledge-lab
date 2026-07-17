# 🧠 CV & AI Knowledge Lab — Master Edition

> **Das deutschsprachige Wissenshub für Computer Vision und Künstliche Intelligenz.**
> Vom mathematischen Fundament über klassische Bildverarbeitung bis zu Transformern,
> Diffusion Models und multimodaler KI — interaktiv, mit Code und Visualisierungen.

[![Made with Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-7C3AED.svg)](LICENSE)

---

## ✨ Was ist das?

Ein **Mega-Wissenshub** mit **43 Lernmodulen**, der den kompletten Bogen von
"Was ist ein Pixel?" bis "Wie funktioniert Sora?" spannt. Drei Dinge zeichnen es aus:

- 🎨 **Modernes, dunkles Design** — Inter & JetBrains Mono, Gradient-Akzente, Cards mit Hover, hand-crafted CSS.
- 🧪 **Interaktive Demos** — Upload eigene Bilder, sieh Filter live, spiele mit Gradient Descent, visualisiere Transformationen.
- 📚 **Tiefe & Breite** — von $\nabla f$ bis Stable Diffusion, in **deutscher Sprache**, technisch fundiert aber lesbar.

---

## 🗺️ Inhalt

### 🏠 Übersicht
- **Startseite** — Hero, Stats, Lernpfad-Vorschau
- **Roadmap & Lernpfad** — strukturierter Weg vom Anfänger zum Experten

### 🧮 Grundlagen
- **Mathe-Crashkurs** — Notation, was du wirklich brauchst
- **Lineare Algebra** — Tensoren, Matrizen, Eigenwerte (interaktive 2D-Transformationen)
- **Analysis & Gradienten** — Backprop, Gradient Descent (Live-Visualisierung)
- **Wahrscheinlichkeit** — Bayes, Cross-Entropy, KL, Gauß-Verteilung interaktiv

### 🖼️ Bildverarbeitung
- **Bildgrundlagen** — Pixel, Farbräume, Sampling, Aliasing
- **Filter & Faltung** — Live-Demo: 8 Kernel auf eigenen Bildern
- **Kantendetektion** — Sobel, Canny, Laplace mit Sliders
- **Feature Detection** — SIFT, ORB, Harris, Live-Keypoints
- **Morphologie** — Erosion, Dilatation, Opening, Closing live
- **Segmentierung** — Threshold, Otsu, Adaptive, K-Means
- **Objekterkennung & Tracking** — YOLO/DETR, mAP, NMS, MOT, ByteTrack/DeepSORT

### 🤖 Deep Learning
- **Neuronale Netze von Grund auf** — Perzeptron, MLP, MLP-Implementation
- **Convolutional Neural Networks** — von LeNet bis ConvNeXt
- **Training, Loss & Optimizer** — alles, was zwischen 70% und 95% entscheidet
- **Moderne Architekturen** — ResNet, EfficientNet, ConvNeXt, U-Net, YOLO
- **Self-Supervised Learning** — SimCLR, MoCo, DINO, MAE
- **Video Understanding** — Action Recognition, Temporal Modeling, Video-Transformer

### 🔥 State-of-the-Art
- **Transformer & Attention** — Self-Attention, ViT, Swin
- **Vision-Language Models** — CLIP, BLIP-2, LLaVA, Flamingo
- **Diffusion Models** — DDPM, Stable Diffusion, Flow Matching, Sora
- **Generative KI** — GANs, VAEs, Autoregressive — und der Vergleich
- **Multimodal & LLMs** — GPT-4o, Gemini, Claude, Sora, Frontier 2026
- **3D Computer Vision** — Kamera-Geometrie, Epipolar, SfM/SLAM, NeRF/3D Gaussian Splatting
- **RAG + Multimodal Agents** — Vision-RAG, Tool-Use, Prompting-Patterns, Guardrails

### 🚀 Praxis
- **Praxisprojekte** — 12 Projekte mit kompletten Code-Beispielen
- **Datasets & Tools** — wo du Daten findest und wie du sie nutzt
- **Deployment & MLOps** — vom Notebook zur Production-API
- **Evaluation & Robustness** — Calibration, OOD, Domain Shift, Error Analysis, Bias/Fairness

### 📰 Live
- **Live News** — RSS-Feeds aus arXiv, OpenAI, DeepMind, HuggingFace, Anthropic, Google AI
- **Paper-Bibliothek** — kuratierte Liste der Must-Read-Paper, sortiert nach Thema
- **Ressourcen & Tools** — Bücher, Kurse, Frameworks, Communities

---

## 🚀 Lokal starten

### Voraussetzungen
- Python 3.10+
- pip (oder uv, poetry, etc.)

### Installation
```bash
git clone https://github.com/hackbert301009/cv-ai-knowledge-lab.git
cd cv-ai-knowledge-lab
pip install -r requirements.txt
streamlit run app.py
```

Die App öffnet sich automatisch im Browser unter `http://localhost:8501`.

---

## ☁️ Kostenlos hosten — Streamlit Community Cloud

Die einfachste Variante. **Komplett kostenlos.** In 3 Schritten online:

### 1. Repository auf GitHub
Sicherstellen, dass dein Code (mit `app.py` und `requirements.txt` im Root) auf GitHub liegt.

### 2. Account bei Streamlit
Auf [share.streamlit.io](https://share.streamlit.io) mit GitHub einloggen.

### 3. App deployen
- "New app" klicken
- Repository auswählen (`hackbert301009/cv-ai-knowledge-lab`)
- Branch: `main`
- Main file path: `app.py`
- "Deploy!" klicken

Nach 1–2 Minuten ist deine App live unter `https://<dein-repo>-<hash>.streamlit.app`.

### Alternative kostenlose Hosting-Optionen
- **[Hugging Face Spaces](https://huggingface.co/spaces)** — kostenlos, perfekt für ML-Demos, einfach
- **[Render](https://render.com)** — Free Tier für Web-Services
- **[Railway](https://railway.app)** — gratis Starter Credits

---

## 📂 Projektstruktur

```
cv-ai-knowledge-lab/
├── app.py                      # Haupt-Entry-Point mit Sidebar-Navigation
├── requirements.txt            # Python-Dependencies
├── README.md                   # Du bist hier
├── LICENSE                     # MIT
├── .gitignore
├── .streamlit/
│   └── config.toml             # Dunkles Custom-Theme
├── assets/
│   └── styles.css              # Globales Custom-CSS
└── src/
    ├── __init__.py
    ├── registry.py             # Modul-Registry (Single Source of Truth: MODULES → CATEGORIES)
    ├── components/
    │   ├── __init__.py
    │   └── ui.py               # Wiederverwendbare UI-Helpers
    └── modules/                # 43 Lernmodule, eine Datei pro Modul
        ├── __init__.py
        ├── home.py · roadmap.py
        ├── math_crashcourse.py · linalg.py · calculus.py · probability.py · tensor_playground.py
        ├── image_basics.py · camera_pipeline.py · filters.py · edges.py · features.py
        ├── morphology.py · segmentation_classic.py · optical_flow.py · object_tracking.py
        ├── nn_basics.py · cnn.py · training.py · modern_archs.py · self_supervised.py · video_understanding.py
        ├── transformers_mod.py · vlm.py · diffusion.py · gen_ai.py · multimodal.py · vision_foundation.py
        ├── pose_estimation.py · three_d_vision.py · rag_multimodal_agents.py
        ├── learning_studio.py · projects.py · datasets.py · evaluation_robustness.py · compression.py · edge_ai.py · deployment.py
        ├── news.py · papers.py · paper_of_month.py · resources.py
        └── glossar.py
```

> **Modul-IDs sind stabil**, aber Reihenfolge & Kategorie werden ausschließlich in
> `registry.py` gepflegt. `CATEGORIES` (Sidebar) wird daraus automatisch abgeleitet —
> keine doppelte Liste mehr.

---

## 🧩 Eigenes Modul hinzufügen

1. Erstelle `src/modules/dein_modul.py` mit einer `render()`-Funktion
2. Trage **einen** `Module(...)`-Eintrag in `src/registry.py` → `MODULES` ein
   (Position in der Liste = Reihenfolge; `category` muss in `CATEGORY_ORDER` stehen)
3. Mappe es in `app.py` → `MODULE_FILES` zu seiner Datei
4. Fertig — Sidebar-Navigation **und** `CATEGORIES` aktualisieren sich automatisch

### Didaktik-Standard (empfohlener Modul-Aufbau)
Ein Lehrmodul folgt idealerweise diesem Bogen:

`hero()` → **Theorie** (Tabs) → **interaktives Lab** (`lab_header` + Sliders/Upload)
→ `render_learning_block(...)` (Mischformat-Übung) → `render_quiz_checkpoint(...)` (Checkpoint).

Nimmt ein Modul einen `render_quiz_checkpoint` mit `module_id="<id>"` auf, taucht es
automatisch im **Personal-Hub-Fortschritt** auf (siehe `CHECKPOINT_MODULES` in `app.py`).

Beispiel `src/modules/dein_modul.py`:
```python
import streamlit as st
from src.components import hero, section_header

def render():
    hero(
        eyebrow="Mein Bereich · Modul N",
        title="Mein Thema",
        sub="Kurze Beschreibung worum es geht."
    )
    section_header("Erste Sektion")
    st.markdown("Inhalt hier ...")
```

---

## 🎨 Design-System

- **Schriften**: Inter (Body), JetBrains Mono (Code)
- **Farben**: Lila/Pink/Orange-Gradient (`#7C3AED → #EC4899 → #F59E0B`)
- **Komponenten**: `hero`, `section_header`, `card`, `info_box`, `stat_tile`, `level_badge`
- **Boxen**: `info`, `success`, `warn`, `tip`

Siehe `src/components/ui.py` und `assets/styles.css`.

---

## 🤝 Mitwirken

Pull Requests willkommen! Besonders interessant:
- Neue Module zu spezifischen Topics (3D-CV, NeRF, Gaussian Splatting, Robotics, ...)
- Mehr interaktive Demos
- Übersetzungen weiterer Module ins Englische
- Bug-Fixes und Verbesserungen am Design

---

## 📜 Lizenz

MIT — siehe [LICENSE](LICENSE).

---

## 💜 Credits & Inspiration

- Design-Inspiration: Linear, Vercel, Tailwind UI, Radix
- Inhalte basieren auf öffentlich zugänglichen Papern und Tutorials
- Alle erwähnten Modelle, Bibliotheken und Tools gehören ihren jeweiligen Eigentümern

---

> **Konsistenz schlägt Intensität.** Lieber jeden Tag 30 Minuten als einmal pro Woche 5 Stunden.
> Code mit. Erkläre einer anderen Person, was du gelernt hast — das ist der ultimative Test. 🚀
