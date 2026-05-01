"""Vision-Language Models — CLIP, BLIP-2, LLaVA, Flamingo."""
import streamlit as st
from src.components import (
    hero, section_header, divider, info_box, card, render_card_grid,
    video_embed, lab_header, key_concept, step_list,
)


def render():
    hero(
        eyebrow="State-of-the-Art · Modul 16",
        title="Vision-Language Models",
        sub="Modelle, die Bild und Text in einem gemeinsamen Raum verstehen. "
            "CLIP hat alles verändert — und die Modelle danach haben Multimodal-KI definiert."
    )

    tabs = st.tabs(["🔗 CLIP", "🌉 BLIP-2", "🦙 LLaVA", "🦩 Flamingo", "🎯 Anwendungen", "🎬 Lernvideos"])

    with tabs[0]:
        section_header("CLIP — Contrastive Language-Image Pretraining (2021)")
        st.markdown(r"""
**CLIP** (OpenAI, Radford et al.) hat 400M Bild-Text-Paare aus dem Web gesammelt und gelernt,
sie in einem **gemeinsamen Embedding-Raum** anzuordnen.

#### Das Training
Pro Batch: $N$ Bilder, $N$ Texte. Berechne alle $N \times N$ Cosine-Similarities.
- **Diagonale** soll hoch sein (richtige Paare)
- **Rest** soll niedrig sein (falsche Paare)

Das ist **Contrastive Loss** — symmetrisch über Bilder und Texte.

$$\mathcal{L} = -\frac{1}{2N} \sum_i \left[ \log \frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ij}/\tau}} + \log \frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ji}/\tau}} \right]$$

#### Was kann CLIP?
- **Zero-Shot Klassifikation**: Gib einfach die Klassennamen als Text ein
- **Bildersuche** über Text-Queries
- **Stable Diffusion** benutzt CLIP für Text-Conditioning
- **Foundation Model** für viele Downstream-Tasks
        """)
        info_box(
            "CLIP installieren: `pip install git+https://github.com/openai/CLIP.git` "
            "(separates Paket, nicht auf PyPI). Alternativ: `transformers` von HuggingFace nutzen.",
            kind="warn",
        )
        st.code("""
# Option 1: OpenAI CLIP (Original)
# pip install git+https://github.com/openai/CLIP.git
import torch
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(image_pil).unsqueeze(0).to(device)
text  = clip.tokenize(["ein Hund", "eine Katze", "ein Auto"]).to(device)

with torch.no_grad():
    img_feat  = model.encode_image(image)
    txt_feat  = model.encode_text(text)
    logits, _ = model(image, text)
    probs     = logits.softmax(dim=-1)

# Option 2: HuggingFace Transformers (einfacher zu installieren)
# pip install transformers
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

inputs = processor(
    text=["ein Hund", "eine Katze", "ein Auto"],
    images=image_pil,
    return_tensors="pt", padding=True
)
with torch.no_grad():
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
        """, language="python")

    with tabs[1]:
        section_header("BLIP-2 — die Brücke zu LLMs (2023)")
        st.markdown(r"""
**BLIP-2** (Salesforce) hat den cleveren Trick: Verwende einen vortrainierten Bild-Encoder
**und** einen vortrainierten LLM, friere beide ein, und lerne nur eine kleine **Brücke** dazwischen.

#### Q-Former — die Brücke
Ein kleiner Transformer mit lernbaren **Query-Tokens**, der:
1. Aus den Bild-Features die wichtigsten Informationen extrahiert.
2. Sie als Tokens an den LLM weitergibt.

So bekommst du einen **multimodalen LLM** für Bruchteil der Trainingskosten.
        """)

    with tabs[2]:
        section_header("LLaVA — visuell instruktionsgesteuert (2023)")
        st.markdown(r"""
**LLaVA** (Liu et al.) hat das Visual-Instruction-Tuning populär gemacht.

#### Architektur
- CLIP ViT-L/14 als Bild-Encoder
- Vicuna / LLaMA als LLM
- Eine **lineare Projektion** (später MLP) als Brücke

#### Training in 2 Stufen
1. **Pretraining**: Nur die Projektion trainieren mit Bild-Caption-Paaren.
2. **Finetuning**: LLM und Projektion auf Visual-Instruction-Daten (von GPT-4 generiert).

LLaVA-1.5/1.6 sind heute starke offene Alternativen zu GPT-4V.
        """)

    with tabs[3]:
        section_header("Flamingo — wenige-Schuss-Bildverständnis (2022)")
        st.markdown(r"""
**Flamingo** (DeepMind) konnte aus **wenigen Beispielen** im Prompt neue Aufgaben lernen
(In-Context Learning für Bilder).

#### Innovationen
- **Perceiver Resampler**: variable Anzahl Bild-Tokens → fixe Anzahl
- **Cross-Attention** vom LLM zu den Bild-Tokens, in **eingefrorenen LLM-Layern** eingefügt
- Verarbeitet **interleaved Bild-Text-Sequenzen** (also auch Videos, Mehrere Bilder)

Flamingo war historisch wichtig — die Architektur lebt in vielen heutigen Multimodal-LLMs weiter.
        """)

    with tabs[4]:
        section_header("Was VLMs heute können")
        cards = [
            card("🔍", "Visual Question Answering", "Fragen über Bildinhalt beantworten — Objekte, Aktionen, Beziehungen.", ["VQA"], ["pink"]),
            card("📝", "Image Captioning", "Bilder in natürliche Sprache übersetzen — auch mit Stil-Vorgaben.", ["Captioning"], ["pink"]),
            card("🎯", "Grounding", "Wo im Bild ist das beschriebene Objekt? Bounding-Box-Output.", ["Lokalisierung"], ["pink"]),
            card("📊", "Document Understanding", "Tabellen, Diagramme, Formulare lesen und verstehen.", ["OCR+"], ["pink"]),
            card("🎨", "Stil + Inhalt", "Stilanalyse, Komposition, Kunstrichtung erkennen.", ["Aesthetic"], ["pink"]),
            card("🤖", "Embodied AI", "Robotern Aufgaben in natürlicher Sprache geben — sie sehen die Welt.", ["Robotics"], ["pink"]),
        ]
        render_card_grid(cards, cols=3)

        info_box(
            "Heutige Top-VLMs (2026): GPT-4o, Claude 3.7 Sonnet, Gemini 2.0 Flash, LLaVA-NeXT, Qwen2-VL, "
            "InternVL2, Pixtral — die offene Szene holt rasant auf.",
            kind="tip",
        )

    with tabs[5]:
        section_header("Lernvideos", "Vision-Language Models verstehen.")

        st.markdown("#### CLIP erklärt — Yannic Kilcher")
        video_embed("T9XSU0pKX2E",
                    "CLIP: Connecting Text and Images — Yannic Kilcher",
                    "Yannic Kilcher liest das CLIP-Paper durch und erklärt jedes Detail. ~40 Minuten.")

        divider()

        st.markdown("#### Vision Transformers (ViT) — Computerphile")
        video_embed("TrdevFK_am4",
                    "Vision Transformers — Computerphile",
                    "Wie ViT und moderne VLMs intern funktionieren.")

        divider()

        info_box(
            "Für CLIP: Das Originalpaper 'Learning Transferable Visual Models From Natural Language Supervision' "
            "(Radford et al., 2021) ist sehr gut lesbar. Starte damit nach den Videos.",
            kind="tip",
        )
