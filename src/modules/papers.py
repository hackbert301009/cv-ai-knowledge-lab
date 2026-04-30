"""Paper-Bibliothek — Must-Read Papers nach Thema."""
import streamlit as st
from src.components import hero, section_header, divider, info_box


PAPERS = {
    "🏛️ CV Klassiker": [
        ("Gradient-Based Learning Applied to Document Recognition", "LeCun et al., 1998",
         "LeNet-5. Das erste richtige CNN. Pflichtlektüre.",
         "http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf"),
        ("ImageNet Classification with Deep Convolutional Neural Networks", "Krizhevsky, Sutskever, Hinton, 2012",
         "AlexNet — der Knall, der Deep Learning auslöste.",
         "https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks"),
        ("Very Deep Convolutional Networks for Large-Scale Image Recognition", "Simonyan & Zisserman, 2014",
         "VGG. Klar, einfach, stapelbar.",
         "https://arxiv.org/abs/1409.1556"),
        ("Deep Residual Learning for Image Recognition", "He et al., 2015",
         "ResNet. Skip Connections = sehr tiefe Netze trainierbar.",
         "https://arxiv.org/abs/1512.03385"),
    ],
    "⚡ Transformers": [
        ("Attention Is All You Need", "Vaswani et al., 2017",
         "Der Original-Transformer. Liest man wenigstens einmal.",
         "https://arxiv.org/abs/1706.03762"),
        ("An Image is Worth 16x16 Words", "Dosovitskiy et al., 2020",
         "ViT — Bilder als Sequenz von Patches.",
         "https://arxiv.org/abs/2010.11929"),
        ("Swin Transformer", "Liu et al., 2021",
         "Hierarchischer Transformer mit Shifted Windows.",
         "https://arxiv.org/abs/2103.14030"),
        ("DINOv2: Learning Robust Visual Features without Supervision", "Oquab et al., 2023",
         "Self-Supervised ViT, das CLIP in vielen Tasks schlägt.",
         "https://arxiv.org/abs/2304.07193"),
    ],
    "🌊 Diffusion": [
        ("Denoising Diffusion Probabilistic Models", "Ho, Jain, Abbeel, 2020",
         "DDPM. Die Grundlage moderner Diffusion.",
         "https://arxiv.org/abs/2006.11239"),
        ("High-Resolution Image Synthesis with Latent Diffusion Models", "Rombach et al., 2022",
         "Stable Diffusion. Latent space + Diffusion + Text.",
         "https://arxiv.org/abs/2112.10752"),
        ("Classifier-Free Diffusion Guidance", "Ho & Salimans, 2022",
         "CFG — wie man Diffusion auf einen Prompt zwingt.",
         "https://arxiv.org/abs/2207.12598"),
        ("Flow Matching for Generative Modeling", "Lipman et al., 2022",
         "Die elegante Formulierung, die jetzt SD3 antreibt.",
         "https://arxiv.org/abs/2210.02747"),
        ("Scalable Diffusion Models with Transformers", "Peebles & Xie, 2022",
         "DiT — Transformer als Backbone für Diffusion. Basis von Sora.",
         "https://arxiv.org/abs/2212.09748"),
    ],
    "👁️ VLM & Multimodal": [
        ("Learning Transferable Visual Models From Natural Language Supervision", "Radford et al., 2021",
         "CLIP. Das wichtigste VLM-Paper der letzten Jahre.",
         "https://arxiv.org/abs/2103.00020"),
        ("BLIP-2: Bootstrapping Language-Image Pre-training", "Li et al., 2023",
         "Q-Former — wie man LLM und Bild-Encoder verbindet.",
         "https://arxiv.org/abs/2301.12597"),
        ("Visual Instruction Tuning (LLaVA)", "Liu et al., 2023",
         "Wie man einen multimodalen Assistenten günstig baut.",
         "https://arxiv.org/abs/2304.08485"),
        ("Flamingo: a Visual Language Model for Few-Shot Learning", "Alayrac et al., 2022",
         "Erste echte Few-Shot-Multimodal-Architektur.",
         "https://arxiv.org/abs/2204.14198"),
    ],
    "🎯 Detection & Segmentation": [
        ("You Only Look Once: Unified Real-Time Object Detection", "Redmon et al., 2016",
         "Das Original-YOLO. Echtzeit-Detection.",
         "https://arxiv.org/abs/1506.02640"),
        ("Faster R-CNN", "Ren et al., 2015",
         "Two-Stage Detector — wichtig für Detection-Theorie.",
         "https://arxiv.org/abs/1506.01497"),
        ("U-Net: Convolutional Networks for Biomedical Image Segmentation", "Ronneberger et al., 2015",
         "U-Net. Die Architektur lebt heute in Stable Diffusion weiter.",
         "https://arxiv.org/abs/1505.04597"),
        ("Segment Anything", "Kirillov et al., 2023",
         "SAM. Prompt-basiertes Segmentierungs-Foundation-Model.",
         "https://arxiv.org/abs/2304.02643"),
        ("DETR: End-to-End Object Detection with Transformers", "Carion et al., 2020",
         "Detection ohne Anker oder NMS.",
         "https://arxiv.org/abs/2005.12872"),
    ],
    "🎨 Generative (Pre-Diffusion)": [
        ("Generative Adversarial Nets", "Goodfellow et al., 2014",
         "Das GAN-Original.",
         "https://arxiv.org/abs/1406.2661"),
        ("Auto-Encoding Variational Bayes", "Kingma & Welling, 2013",
         "VAE — die theoretische Eleganz.",
         "https://arxiv.org/abs/1312.6114"),
        ("A Style-Based Generator Architecture for GANs", "Karras et al., 2018",
         "StyleGAN — hyperrealistische Gesichter.",
         "https://arxiv.org/abs/1812.04948"),
    ],
}


def render():
    hero(
        eyebrow="Live · Modul 24",
        title="Paper-Bibliothek",
        sub="Eine kuratierte Liste der wichtigsten Paper, sortiert nach Thema. "
            "Die hier sind die, die du wenigstens einmal angefasst haben solltest."
    )

    info_box(
        "Tipps zum Paper-Lesen: Drei-Pass-Methode. "
        "Pass 1: Abstract, Intro, Conclusion (5 Min). "
        "Pass 2: Figures und Methodik (30 Min). "
        "Pass 3: Komplett, mit Mathe (mehrere Stunden). "
        "Die meisten Paper musst du nur Pass 1 lesen.",
        kind="tip",
    )

    for category, papers in PAPERS.items():
        section_header(category)
        for title, authors, summary, url in papers:
            with st.container():
                st.markdown(f"#### [{title}]({url})")
                st.caption(f"📝 {authors}")
                st.markdown(summary)
                st.markdown("")
        divider()

    info_box(
        "Wo du noch suchen kannst: "
        "**Papers with Code** für Code-Linker, "
        "**alphaXiv** für Diskussionen, "
        "**Hugging Face Daily Papers** für tägliche Highlights, "
        "**Twitter/X** für News-Geschwindigkeit.",
        kind="info",
    )
