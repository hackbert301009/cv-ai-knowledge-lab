"""Multimodal & LLMs — GPT-4o, Gemini, Sora."""
import streamlit as st
from src.components import hero, section_header, divider, info_box, card, render_card_grid


def render():
    hero(
        eyebrow="State-of-the-Art · Modul 19",
        title="Multimodal &amp; LLMs",
        sub="Wenn ein Modell gleichzeitig Bild, Text, Audio und Video versteht — und auch erzeugt. "
            "Die Frontier von 2024–2026."
    )

    tabs = st.tabs(["🌐 Was ist Multimodal?", "🚀 Frontier Models", "🏗️ Architekturen", "🎬 Sora & Video", "📈 Ausblick"])

    with tabs[0]:
        section_header("Multimodal — was heißt das?")
        st.markdown(r"""
Ein **Multimodal-Modell** verarbeitet mehrere Modalitäten:
- **Text** (Schrift)
- **Bild** (statische Visuals)
- **Audio** (Sprache, Geräusche, Musik)
- **Video** (Bild + Zeit + Audio)

Das Ziel: ein **einheitliches Modell**, das nahtlos zwischen Modalitäten übersetzt
und kombinierte Aufgaben löst — z.B. ein Video anschauen und in natürlicher Sprache zusammenfassen.

#### Warum so wichtig?
- Menschen denken multimodal — wir hören, sehen und lesen gleichzeitig.
- **Robotik** braucht es zwingend (Sehen + Sprache + Aktion).
- **Bessere Repräsentationen** durch komplementäre Signale.
        """)

    with tabs[1]:
        section_header("Frontier Multimodal Models 2025/26")
        cards = [
            card("🌟", "GPT-5 / GPT-4o", "OpenAI's Flaggschiff. Bild, Audio, Video Input, Text/Audio Output. End-to-end multimodal trainiert.", ["OpenAI"], ["pink"]),
            card("💎", "Gemini 2 Pro", "Google's Antwort. Massives Kontextfenster (1M+ Tokens), nativ multimodal von Anfang an.", ["Google"], ["pink"]),
            card("🎭", "Claude 3.5/4 Opus", "Anthropic — stark bei Reasoning, Bild- und Dokumentverständnis, Constitutional AI.", ["Anthropic"], ["pink"]),
            card("🦙", "LLaMA 4", "Meta's Open-Source-Familie, multimodal, von Forschern weltweit weiterentwickelt.", ["Meta"], ["green"]),
            card("🔮", "Qwen2.5-VL / InternVL", "Chinesische Open-Source-Modelle, oft state-of-the-art bei Vision-Tasks.", ["China"], ["green"]),
            card("🎬", "Sora 2 / Veo 3", "Video-Generierung — Sora & Google Veo dominieren die Spitze.", ["Generativ"], ["amber"]),
        ]
        render_card_grid(cards, cols=3)

    with tabs[2]:
        section_header("Wie multimodal trainiert wird")
        st.markdown(r"""
Es gibt grob drei Strategien:

#### 1. Late Fusion (CLIP-Stil)
Separate Encoder pro Modalität, dann erst am Ende kombiniert.
- ✅ Einfach, parallelisierbar, gute Embeddings
- ❌ Begrenzte Interaktion zwischen Modalitäten

#### 2. Mid Fusion (LLaVA-Stil)
Bild-Encoder vorab, dann **Tokens als Eingabe** an einen LLM.
- ✅ Nutzt vortrainiertes LLM-Wissen
- ❌ Modalitäten werden nicht gemeinsam optimiert

#### 3. Early Fusion (GPT-4o-Stil)
**Ein einziges Modell** wird von Anfang an mit allen Modalitäten gemeinsam trainiert.
- ✅ Beste Performance, natürlichste Interaktion
- ❌ Massiver Compute-Bedarf

#### Tokenisierung von Bildern und Audio
Heutige Multimodal-LLMs **diskretisieren** Bild und Audio zu Tokens (über VQ-VAE, RVQ, etc.),
sodass sie wie Text-Tokens behandelt werden können.
- Bild: ~256–1024 Tokens pro Bild
- Audio: ~50 Tokens pro Sekunde Sprache
        """)

    with tabs[3]:
        section_header("Sora und Video-Generierung")
        st.markdown(r"""
**Sora** (OpenAI, 2024) hat gezeigt, was möglich ist: minutenlange, kohärente Videos aus Text-Prompts.

#### Architektur (basiert auf öffentlichen Hinweisen)
- **DiT** (Diffusion Transformer) als Backbone
- Eingabe: **Spacetime-Patches** — 3D-Kuben aus Pixel- und Zeitachse
- Trainiert auf riesiger Menge Video, einschließlich gerenderter und realer Szenen

#### Was Sora besonders macht
- **Welt-Modell-artiges Verhalten**: Objektkonsistenz, Physik (mit Schwächen)
- **Kameraführung**: dreht und bewegt sich kohärent
- **Schnitt-Komposition**: kann mehrere Shots in einem Clip
- **Charaktere**: bleiben über Zeit konsistent

#### 2025/26 Stand
- **Sora 2** (OpenAI), **Veo 3** (Google), **Kling 2** (Kuaishou), **Hunyuan Video** (Tencent)
- Open-Source: **CogVideoX**, **HunyuanVideo**, **Mochi-1**, **LTX-Video**
- Audio-synchronisierte Videos werden Standard
- 30s–60s in hoher Qualität sind Realität
        """)

    with tabs[4]:
        section_header("Wo geht es hin?")
        st.markdown(r"""
#### Trends, die du im Auge behalten solltest

**1. Action-Output-Modelle**
Modelle, die nicht nur Text/Bild/Audio erzeugen, sondern **handeln**:
- Computer-Use (Anthropic, OpenAI Operator)
- Robotik (Gemini Robotics, RT-X)
- Ein einziges Modell für Wahrnehmung + Aktion

**2. Long-Context Multimodal**
- Stundenlange Videos verstehen
- Ganze Bibliotheken im Kontext
- Persistentes Gedächtnis

**3. Effiziente Modelle**
- Mixture-of-Experts (MoE) für riesige Kapazität bei wenig Compute
- Distillation: kleine, schnelle, fast gleich gute Modelle
- On-Device Multimodal (Apple Intelligence, Gemini Nano)

**4. Scientific & Specialized AI**
- AlphaFold-Stil-Modelle für Chemie, Materialwissenschaft, Medizin
- Multimodal-LLMs als Werkzeuge in der Forschung

**5. Self-Improvement**
- Modelle generieren ihre eigenen Trainingsdaten
- Reinforcement Learning auf eigene Outputs (RLHF, RLAIF, RLVR)
        """)
        info_box(
            "Die Geschwindigkeit ist atemberaubend. Ein Tipp: Nicht jedem Modell hinterherrennen — "
            "verstehe die **Konzepte**. Architekturen kommen und gehen, aber Attention, Diffusion und "
            "Contrastive Learning bleiben für die nächsten Jahre relevant.",
            kind="tip",
        )
