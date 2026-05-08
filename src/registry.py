"""
Modul-Registry — zentrale Konfiguration aller Lernmodule.
Alle Module mit Metadaten für Navigation, Kategorisierung und Rendering.
"""

from dataclasses import dataclass, field
from typing import Callable, List


@dataclass
class Module:
    id: str
    icon: str
    title: str
    short: str
    category: str
    level: str          # "Anfänger" | "Fortgeschritten" | "Experte"
    duration: str       # geschätzte Lernzeit
    tags: List[str] = field(default_factory=list)


# --------------------------------------------------------------------
# Kategorien
# --------------------------------------------------------------------
CATEGORIES = {
    "🏠 Übersicht":      ["home", "roadmap"],
    "🧮 Grundlagen":     ["math", "linalg", "calculus", "probability", "tensor_playground"],
    "🖼️ Bildverarbeitung": ["image_basics", "camera_pipeline", "filters", "edges", "features", "morphology", "segmentation_classic"],
    "🤖 Deep Learning":   ["nn_basics", "cnn", "training", "modern_archs"],
    "🔥 State-of-the-Art": ["transformers", "vlm", "diffusion", "gen_ai", "multimodal", "pose_estimation"],
    "🚀 Praxis":          ["learning_studio", "projects", "datasets", "deployment", "compression"],
    "📰 Live":            ["news", "papers", "paper_of_month", "resources"],
    "📚 Referenz":        ["glossar"],
}


# --------------------------------------------------------------------
# Alle Module
# --------------------------------------------------------------------
MODULES: List[Module] = [
    # --- Übersicht ---
    Module("home",        "🏠", "Startseite",                    "Willkommen & Überblick",                      "Übersicht",       "Anfänger",       "5 min",  ["intro"]),
    Module("roadmap",     "🗺️", "Roadmap & Lernpfad",            "Strukturierter Weg vom Anfänger zum Experten", "Übersicht",       "Anfänger",       "10 min", ["guide"]),

    # --- Grundlagen ---
    Module("math",        "📐", "Mathe-Crashkurs",                "Notation, Vektoren, Matrizen — kompakt",       "Grundlagen",      "Anfänger",       "30 min", ["math"]),
    Module("linalg",      "🧮", "Lineare Algebra",                "Tensoren, Matrixoperationen, Eigenwerte",      "Grundlagen",      "Anfänger",       "45 min", ["math", "core"]),
    Module("calculus",    "∂",  "Analysis & Gradienten",          "Ableitungen, Backprop, Optimierung",           "Grundlagen",      "Anfänger",       "40 min", ["math"]),
    Module("probability", "🎲", "Wahrscheinlichkeit & Statistik", "Bayes, Verteilungen, Information Theory",     "Grundlagen",      "Anfänger",       "35 min", ["math"]),

    # --- Bildverarbeitung ---
    Module("image_basics",        "🖼️", "Bildgrundlagen & Pixel",          "Was ist ein Bild? Farbräume, Sampling",      "Bildverarbeitung", "Anfänger",       "20 min", ["cv"]),
    Module("camera_pipeline",     "📷", "Wie eine Kamera ein Bild macht",  "Photonen, Sensor, Bayer, ISP und Grundpipeline", "Bildverarbeitung", "Anfänger",   "35 min", ["cv", "camera", "sensor", "isp"]),
    Module("filters",             "🌫️", "Filter & Faltung",                 "Convolution, Gauß, Median, Bilateral",       "Bildverarbeitung", "Anfänger",       "30 min", ["cv", "core"]),
    Module("edges",               "📏", "Kantendetektion",                 "Sobel, Canny, Laplace — interaktiv",         "Bildverarbeitung", "Fortgeschritten", "25 min", ["cv"]),
    Module("features",            "🔑", "Feature Detection & Matching",    "SIFT, ORB, Harris, Keypoints",               "Bildverarbeitung", "Fortgeschritten", "40 min", ["cv"]),
    Module("morphology",          "🧱", "Morphologie",                     "Erosion, Dilatation, Opening, Closing",      "Bildverarbeitung", "Fortgeschritten", "20 min", ["cv"]),
    Module("segmentation_classic","✂️", "Klassische Segmentierung",        "Threshold, Watershed, GrabCut, K-Means",     "Bildverarbeitung", "Fortgeschritten", "30 min", ["cv"]),

    # --- Deep Learning ---
    Module("nn_basics",    "🧠", "Neuronale Netze von Grund auf", "Perzeptron, MLP, Aktivierungen",              "Deep Learning",   "Anfänger",        "45 min", ["dl", "core"]),
    Module("cnn",          "🔲", "Convolutional Neural Networks",  "Von LeNet bis ResNet",                        "Deep Learning",   "Fortgeschritten", "60 min", ["dl", "core"]),
    Module("training",     "🎯", "Training, Loss & Optimizer",     "SGD, Adam, Regularisierung, Learning Rate",   "Deep Learning",   "Fortgeschritten", "45 min", ["dl"]),
    Module("modern_archs", "🏛️", "Moderne Architekturen",          "ResNet, EfficientNet, ConvNeXt, U-Net",        "Deep Learning",   "Experte",          "60 min", ["dl"]),

    # --- State-of-the-Art ---
    Module("transformers", "⚡", "Transformer & Attention",        "Self-Attention, ViT, Swin Transformer",        "State-of-the-Art", "Experte", "60 min", ["sota"]),
    Module("vlm",          "👁️‍🗨️", "Vision-Language Models",       "CLIP, BLIP-2, LLaVA, Flamingo",                "State-of-the-Art", "Experte", "50 min", ["sota"]),
    Module("diffusion",    "🌊", "Diffusion Models",                "DDPM, Stable Diffusion, Flow Matching",        "State-of-the-Art", "Experte", "55 min", ["sota"]),
    Module("gen_ai",       "🎨", "Generative KI",                   "GANs, VAEs, Autoregressive Modelle",           "State-of-the-Art", "Experte", "45 min", ["sota"]),
    Module("multimodal",   "🌐", "Multimodal & LLMs",               "GPT-4o, Gemini, Sora — Bild+Text+Video",       "State-of-the-Art", "Experte", "40 min", ["sota"]),

    # --- Praxis ---
    Module("learning_studio", "🧪", "Lernstudio: Labs & Uebungen", "Progressive Labs, Mischformat und Community", "Praxis", "Anfänger", "35 min", ["praxis", "labs", "community"]),
    Module("projects",   "💻", "Praxisprojekte",          "10+ Hands-on Projekte mit Code",          "Praxis", "Fortgeschritten", "varies", ["praxis"]),
    Module("datasets",   "📦", "Datasets & Tools",        "ImageNet, COCO, HuggingFace, Roboflow",   "Praxis", "Anfänger",        "20 min", ["praxis"]),
    Module("deployment", "🚀", "Deployment & MLOps",      "ONNX, TensorRT, FastAPI, Docker",         "Praxis", "Experte",          "50 min", ["praxis"]),

    # --- Live ---
    Module("news",          "📰", "Live News",                "Aktuelle Forschung & Releases",                "Live",    "Anfänger",        "live",    ["live"]),
    Module("papers",        "📄", "Paper-Bibliothek",         "Must-Read Papers nach Thema",                  "Live",    "Experte",          "varies",  ["live"]),
    Module("paper_of_month","🗞️", "Paper des Monats",         "Kuratiertes Paper mit Deep-Dive & Video",      "Live",    "Fortgeschritten",  "30 min",  ["live", "paper"]),
    Module("resources",     "🔗", "Ressourcen & Tools",       "Bücher, Kurse, Frameworks, Communities",       "Live",    "Anfänger",         "varies",  ["live"]),

    # --- Referenz ---
    Module("glossar",       "📖", "Glossar & Wörterbuch",     "500+ CV & AI Begriffe durchsuchen",            "Referenz", "Anfänger",        "varies",  ["reference", "glossar"]),

    # --- Neue Inhalte ---
    Module("tensor_playground", "🎮", "Tensor Playground",    "NumPy interaktiv: Reshape, Einsum, Broadcasting", "Grundlagen",    "Anfänger",    "30 min", ["math", "numpy", "core"]),
    Module("compression",       "🗜️", "Model Compression",    "Quantisierung, Pruning, KD, ONNX, TensorRT",      "Praxis",        "Experte",      "45 min", ["praxis", "edge", "deployment"]),
    Module("pose_estimation",   "🧍", "Pose Estimation",      "Human Pose, 6DoF, Skeleton-Visualizer",           "State-of-the-Art", "Experte",  "45 min", ["sota", "cv", "pose"]),
]


def get_module(mod_id: str) -> Module | None:
    """Modul nach ID finden."""
    return next((m for m in MODULES if m.id == mod_id), None)


def modules_by_category() -> dict:
    """Module gruppiert nach Kategorie zurückgeben."""
    grouped: dict = {}
    for m in MODULES:
        grouped.setdefault(m.category, []).append(m)
    return grouped
