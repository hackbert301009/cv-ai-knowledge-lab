"""Paper des Monats — ein ausführlich erklärtes Paper pro Monat."""
import streamlit as st
from datetime import date
from src.components import (
    hero, section_header, divider, info_box,
    video_embed, lab_header, key_concept, step_list, card, render_card_grid,
)

# ---------------------------------------------------------------------------
# Datenbank der monatlichen Papers
# ---------------------------------------------------------------------------
PAPERS = [
    {
        "month": "Mai 2026",
        "title": "Segment Anything Model (SAM)",
        "authors": "Kirillov et al. — Meta AI Research",
        "year": 2023,
        "venue": "ICCV 2023",
        "arxiv": "2304.02643",
        "tldr": "Ein Foundation Model für Bildsegmentierung: ein einziger Klick, Box oder Textprompt genügt, um beliebige Objekte in beliebigen Bildern zu segmentieren — ohne spezifisches Fine-Tuning.",
        "tags": ["Segmentierung", "Foundation Model", "Zero-Shot", "Interactive"],
        "impact": "⭐⭐⭐⭐⭐",
        "difficulty": "Fortgeschritten",
        "youtube_id": "eKQScbQQUZ8",
        "youtube_caption": "SAM erklärt von Yannic Kilcher — Paper Walkthrough",
        "contributions": [
            ("Promptable Segmentation Task", "SAM kann auf Punkte, Boxen, Masken oder Text reagieren — ein universelles Interface für Segmentierung."),
            ("SA-1B Dataset", "11 Millionen Bilder mit 1,1 Milliarden Masken — der größte öffentliche Segmentierungs-Datensatz aller Zeiten. Generiert durch SAM selbst (Data Engine)."),
            ("Drei-Komponenten-Architektur", "Image Encoder (ViT-H), Prompt Encoder, Mask Decoder. Jede Komponente ist austauschbar."),
            ("Amortized Inference", "Image Encoding einmal, dann viele Prompts ultra-schnell (50ms pro Maske). Interaktiv in Echtzeit möglich."),
            ("Zero-Shot Generalisierung", "SAM segmentiert Objekte, die es nie gesehen hat — medizinische Bilder, Satelliten, Unterwasser — ohne Finetuning."),
        ],
        "architecture": r"""
#### Architektur

```
Bild (1024×1024)
     ↓
Image Encoder (MAE-pretrained ViT-H)
     ↓  image embedding [256×64×64]
     ↓
Prompt Encoder ←── Punkte / Boxen / Masken / Text
     ↓  prompt embeddings
     ↓
Mask Decoder (2× Transformer-Layer + MLP)
     ↓
3 Masken + Confidence Scores
```

**Warum 3 Masken?** Ambiguität — ein Klick auf ein Rad kann das Rad, das Auto oder das Bild meinen.
SAM gibt immer 3 Optionen aus, sortiert nach Confidence.

#### Image Encoder (ViT-H)
- 632M Parameter, ViT-H/16 Backbone
- Pretrained mit MAE (Masked Autoencoding) auf SA-1B
- Produziert 256-dimensionale Features bei 64×64 Auflösung

#### Prompt Encoder
- **Punkte**: Positionsembedding + Vordergrund/Hintergrund-Label
- **Boxen**: Als Eckpunkte mit Labels kodiert
- **Masken**: Durch Convolutions auf die richtige Auflösung gebracht
- **Text**: CLIP Text-Encoder (noch in Entwicklung zum Release-Zeitpunkt)

#### Mask Decoder
- Zwei Transformer-Layer mit Cross-Attention zwischen Prompts und Image-Embeddings
- MLP-Kopf für jede der 3 Masken
- Sehr klein (wenige MB) — deshalb 50ms Inference!
""",
        "code": """
# SAM mit HuggingFace Transformers (empfohlen)
# pip install transformers torch Pillow

from transformers import SamModel, SamProcessor
from PIL import Image
import torch
import numpy as np

# Modell laden (ViT-H = beste Qualität, ViT-B = schnell)
model = SamModel.from_pretrained("facebook/sam-vit-base")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
model.eval()

# Bild und Punkt-Prompt
image = Image.open("your_image.jpg")
input_point = [[500, 375]]   # x, y Koordinate
input_label = [1]            # 1 = Vordergrund, 0 = Hintergrund

# Verarbeitung
inputs = processor(
    image,
    input_points=[input_point],
    input_labels=[input_label],
    return_tensors="pt"
)

# Inferenz
with torch.no_grad():
    outputs = model(**inputs)

# Masken extrahieren
masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(),
    inputs["original_sizes"].cpu(),
    inputs["reshaped_input_sizes"].cpu()
)
scores = outputs.iou_scores  # Confidence pro Maske

# Beste Maske
best_mask = masks[0][0][scores[0].argmax()].numpy()
print(f"Maske Größe: {best_mask.shape}, Coverage: {best_mask.mean()*100:.1f}%")

# ----- Automatische Segmentierung (kein Prompt nötig!) -----
from transformers import pipeline

generator = pipeline("mask-generation",
                     model="facebook/sam-vit-base",
                     device="cuda" if torch.cuda.is_available() else "cpu")

outputs = generator(image, points_per_batch=64)
masks_auto = outputs["masks"]
print(f"Automatisch gefundene Masken: {len(masks_auto)}")
""",
        "why_matters": """
SAM ist ein **Paradigmenwechsel** in der Segmentierung:

- **Vorher**: Für jede neue Klasse (Tumoren, Satelliten, Industrieteile) → neues Fine-Tuning nötig
- **Nachher**: Ein Modell für alles. Zero-Shot. Kein Fine-Tuning.

**Heute eingesetzt bei:**
- Medizinischen Bildgebungssystemen (SAM-Med2D, MedSAM)
- Robotik (als universeller "Objekte identifizieren" Baustein)
- Foto-Editoren (Photoshop Generative Fill nutzt ähnliche Konzepte)
- Datenbeschriftung (10× schneller mit SAM als Vorschlag)
- SAM 2 (2024): Auch für Videos — Objekte über Frames tracken
""",
    },
    {
        "month": "April 2026",
        "title": "Attention Is All You Need",
        "authors": "Vaswani et al. — Google Brain / Google Research",
        "year": 2017,
        "venue": "NeurIPS 2017",
        "arxiv": "1706.03762",
        "tldr": "Transformers ohne Recurrenz oder Convolution — reines Self-Attention. Das Paper, das die gesamte moderne KI verändert hat. Heute Basis von GPT, BERT, ViT, Stable Diffusion, CLIP und praktisch allem.",
        "tags": ["Transformer", "Attention", "NLP", "Foundational"],
        "impact": "⭐⭐⭐⭐⭐",
        "difficulty": "Experte",
        "youtube_id": "iDulhoQ2pro",
        "youtube_caption": "Yannic Kilcher liest 'Attention is All You Need' — detaillierter Paper Walkthrough",
        "contributions": [
            ("Self-Attention Mechanismus", "Jede Position in einer Sequenz kann direkt jede andere 'sehen' — in O(1) Schritten statt O(N) bei RNNs."),
            ("Multi-Head Attention", "Mehrere parallele Attention-Köpfe lernen verschiedene Arten von Beziehungen gleichzeitig."),
            ("Positional Encoding", "Sinusoidale Positionscodierung ohne Recurrenz — elegant und generalisiert auf ungesehene Längen."),
            ("Encoder-Decoder Architektur", "Skalierbar und modular. Encoder + Decoder getrennt verwendbar (BERT = nur Encoder, GPT = nur Decoder)."),
            ("Parallelisierbarkeit", "Kein sequenzielles RNN — alles parallel auf GPU. Training 10× schneller. Die eigentliche Revolution."),
        ],
        "architecture": r"""
#### Transformer Architektur

```
Input Tokens
     ↓ + Positional Encoding
     ↓
[Encoder Block × N]
  ├── Multi-Head Self-Attention
  ├── Add & LayerNorm
  ├── Feed-Forward Network
  └── Add & LayerNorm
     ↓ Memory
[Decoder Block × N]
  ├── Masked Multi-Head Self-Attention
  ├── Add & LayerNorm
  ├── Cross-Attention (→ Memory)
  ├── Add & LayerNorm
  ├── Feed-Forward Network
  └── Add & LayerNorm
     ↓
Linear + Softmax → Output Tokens
```

**Skalierung im Original:** 6 Encoder + 6 Decoder, d_model=512, h=8 Heads → 65M Parameter.
Heute: GPT-4 ~1.8 Billionen Parameter, selbe Grundidee.
""",
        "code": """
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k    = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        B, T, _ = q.shape
        # Projizieren und in Köpfe aufteilen
        Q = self.W_q(q).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        # Scaled Dot-Product Attention
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = scores.softmax(-1)
        out  = (attn @ V).transpose(1, 2).reshape(B, T, -1)
        return self.W_o(out), attn

class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.attn(x, x, x, mask)
        x = self.norm1(x + self.drop(attn_out))   # Add & Norm
        x = self.norm2(x + self.drop(self.ff(x))) # Add & Norm
        return x

# Testen
block = TransformerBlock(d_model=512, n_heads=8)
x = torch.randn(2, 50, 512)  # Batch=2, Seq=50, d=512
out = block(x)
print(out.shape)  # [2, 50, 512]
""",
        "why_matters": """
**Dieses Paper hat buchstäblich die Welt verändert:**

- 2018: **BERT** (Google) — Bidirektionaler Transformer für NLP. Basis aller Suchmaschinen heute.
- 2019: **GPT-2** — Erste Ahnung, dass große Transformer generieren können.
- 2020: **GPT-3** — 175B Parameter. Few-Shot Learning. Sprach-KI wird ernst genommen.
- 2020: **ViT** — Transformer für Bilder. CNNs werden herausgefordert.
- 2021: **CLIP, DALL-E** — Multimodale Transformer.
- 2022: **ChatGPT, Stable Diffusion** — Massenmarkt-KI.
- 2023–2026: **GPT-4, Gemini, Claude, Sora** — Alle auf Transformer-Basis.

Zitiert: **100.000+ Mal**. Das meistzitierte KI-Paper der Geschichte.
""",
    },
    {
        "month": "März 2026",
        "title": "DINOv2: Learning Robust Visual Features without Supervision",
        "authors": "Oquab et al. — Meta AI",
        "year": 2023,
        "venue": "TMLR 2024",
        "arxiv": "2304.07193",
        "tldr": "Self-supervised ViT Features, die ohne Labels auskommen und trotzdem besser als supervised Modelle sind. Ein universeller visueller Feature-Extraktor für fast alle CV-Aufgaben.",
        "tags": ["Self-Supervised", "ViT", "Foundation Model", "Features"],
        "impact": "⭐⭐⭐⭐⭐",
        "difficulty": "Experte",
        "youtube_id": "h3ij3F3cPIk",
        "youtube_caption": "DINOv2 erklärt — Architektur und Self-Supervised Learning",
        "contributions": [
            ("Curated LVD-142M Dataset", "142M sorgfältig kuratierte Bilder ohne Labels — Qualität über Quantität."),
            ("Self-Distillation with Knowledge", "Teacher-Student Framework: Teacher ist EMA des Students. Student lernt von Teacher ohne Labels."),
            ("Kombination DINO + iBOT", "Simultane Optimierung: globale Bildrepräsentation (DINO) + lokale Patch-Features (iBOT)."),
            ("Universelle Features", "Depth Estimation, Segmentation, Classification, Retrieval — alles mit demselben frozen Backbone."),
            ("Effiziente Implementierung", "Flash Attention, xFormers, verteiltes Training — reproduzierbar mit modernen Tools."),
        ],
        "architecture": r"""
#### DINOv2 Training Framework

```
Bild  ──→  Teacher Encoder (EMA-Update ←─)
           ↓                              ↑
Crops ──→  Student Encoder  ──→  Loss  ──┘
           (DINO + iBOT kombiniert)
```

**DINO Loss**: Student und Teacher sollen gleiche globale Repräsentation für dasselbe Bild haben
(aber unterschiedliche Crops → Modell lernt Invarianz).

**iBOT Loss**: Einige Patches werden maskiert → Student muss sie rekonstruieren
(ähnlich MAE, aber im Feature-Raum statt Pixel).

**Ergebnis**: Features die gleichzeitig global bedeutungsvoll UND lokal präzise sind.
""",
        "code": """
# DINOv2 Features extrahieren (HuggingFace)
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model     = AutoModel.from_pretrained('facebook/dinov2-base')
model.eval()

image = Image.open("image.jpg")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# [CLS] Token = globale Bildrepräsentation (768-dim bei ViT-B)
cls_feature = outputs.last_hidden_state[:, 0, :]     # [1, 768]
# Patch Tokens = lokale Features
patch_features = outputs.last_hidden_state[:, 1:, :] # [1, 256, 768]

print(f"Global feature: {cls_feature.shape}")
print(f"Patch features: {patch_features.shape}")

# Für Ähnlichkeitssuche (Image Retrieval)
import torch.nn.functional as F

img_a_feat = cls_feature  # Dein Bild
img_b_feat = ...          # Datenbank-Bild
similarity = F.cosine_similarity(img_a_feat, img_b_feat)
print(f"Ähnlichkeit: {similarity.item():.4f}")
""",
        "why_matters": """
**DINOv2 Features sind vielseitig einsetzbar ohne Fine-Tuning:**
- **Tiefenschätzung**: Besser als spezialisierte supervised Modelle auf NYU Depth
- **Segmentierung**: k-NN auf Patch-Features → gute Segmentierung
- **Bildretrieval**: Beste Features für Ähnlichkeitssuche
- **Classification**: Linear Probe auf ImageNet: 86.5% Top-1

In der Praxis: Du nimmst DINOv2, frierst alle Gewichte ein,
und trainierst nur einen kleinen Kopf für deine spezifische Aufgabe.
Das spart 90%+ der Trainingszeit und braucht viel weniger Daten.
""",
    },
]


def _render_paper(p: dict):
    """Rendert ein einzelnes Paper vollständig."""
    st.markdown(
        f"""<div style="background:rgba(124,58,237,0.06);border:1px solid rgba(124,58,237,0.25);
            border-radius:14px;padding:1.25rem 1.5rem;margin-bottom:1.5rem;">
          <div style="font-size:0.75rem;color:#A78BFA;font-weight:700;text-transform:uppercase;
               letter-spacing:0.12em;margin-bottom:0.5rem;">{p['venue']} · {p['year']}</div>
          <div style="font-size:1.5rem;font-weight:800;color:#F3F4F6;margin-bottom:0.3rem;">{p['title']}</div>
          <div style="color:#9CA3AF;font-size:0.9rem;margin-bottom:0.75rem;">{p['authors']}</div>
          <div style="color:#E5E7EB;font-size:0.95rem;line-height:1.6;margin-bottom:0.75rem;">{p['tldr']}</div>
          <div>{''.join(f'<span style="display:inline-block;font-size:0.7rem;font-weight:700;padding:0.15rem 0.55rem;border-radius:6px;background:rgba(59,130,246,0.15);color:#93C5FD;margin:0.2rem;">{t}</span>' for t in p['tags'])}</div>
          <div style="margin-top:0.75rem;font-size:1.1rem;">Impact: {p['impact']}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    paper_tabs = st.tabs([
        "🎯 Kernergebnisse", "🏗️ Architektur", "💻 Code", "🤔 Warum wichtig?", "🎬 Video",
    ])

    with paper_tabs[0]:
        section_header("Die wichtigsten Beiträge")
        step_list(p["contributions"])

    with paper_tabs[1]:
        section_header("Architektur erklärt")
        st.markdown(p["architecture"])

    with paper_tabs[2]:
        section_header("Code-Walkthrough")
        st.code(p["code"], language="python")
        info_box(
            f"Das Paper: **arxiv.org/abs/{p['arxiv']}** — Originalpaper direkt lesen. "
            "Die meisten Papers sind sehr gut lesbar, wenn man die Architektur schon versteht.",
            kind="tip",
        )

    with paper_tabs[3]:
        section_header("Warum dieses Paper wichtig ist")
        st.markdown(p["why_matters"])

    with paper_tabs[4]:
        section_header("Erklärvideo")
        video_embed(p["youtube_id"], p["title"], p["youtube_caption"])


def render():
    hero(
        eyebrow="Live · Paper des Monats",
        title="Paper des Monats",
        sub="Jeden Monat ein zentrales Paper aus Computer Vision und KI — ausführlich erklärt, "
            "mit Code-Walkthrough, Architektur-Diagramm und Video."
    )

    section_header(
        "Wie du diesen Bereich nutzt",
        "Wähle ein Paper, lies die Zusammenfassung, dann Architektur, dann Code — und schau das Video zuletzt.",
    )
    info_box(
        "💡 **Tipp für effektives Paper-Lesen:** 1) Abstract + Conclusion zuerst. "
        "2) Figures durchschauen. 3) Introduction. 4) Related Work überfliegen. "
        "5) Methode. 6) Experimente. — Niemals von vorn nach hinten!",
        kind="tip",
    )

    divider()

    # Automatisch aktuellen Monat vorauswählen
    _DE_MONTHS = {
        1: "Januar", 2: "Februar", 3: "März", 4: "April",
        5: "Mai", 6: "Juni", 7: "Juli", 8: "August",
        9: "September", 10: "Oktober", 11: "November", 12: "Dezember",
    }
    today = date.today()
    current_month_str = f"{_DE_MONTHS[today.month]} {today.year}"
    default_idx = next(
        (i for i, p in enumerate(PAPERS) if p["month"] == current_month_str),
        0,  # Fallback: neuestes Paper
    )

    paper_titles = [f"{p['month']} — {p['title']}" for p in PAPERS]
    selected_idx = st.selectbox(
        "Paper auswählen",
        range(len(PAPERS)),
        format_func=lambda i: paper_titles[i],
        index=default_idx,
    )

    divider()
    _render_paper(PAPERS[selected_idx])

    divider()

    # Übersicht aller Papers
    section_header("Alle Papers im Überblick")
    ov_cols = st.columns(3)
    for i, p in enumerate(PAPERS):
        with ov_cols[i % 3]:
            st.markdown(
                card("📄", p["title"], f"{p['authors'][:40]}… · {p['year']}",
                     [p["venue"], p["difficulty"]], ["pink", "amber"]),
                unsafe_allow_html=True,
            )
