"""
Modul-Registry — zentrale Konfiguration aller Lernmodule.

Single Source of Truth: Die MODULES-Liste definiert Reihenfolge, Kategorie und
Metadaten jedes Moduls. CATEGORIES (Sidebar-Navigation) wird daraus automatisch
abgeleitet — es gibt keine zweite, manuell gepflegte ID-Liste mehr.
"""

from dataclasses import dataclass, field
from typing import List


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
# Kategorie-Reihenfolge + Icons (die einzige Stelle, an der Kategorien
# definiert werden). Module.category muss auf einen dieser Namen zeigen.
# --------------------------------------------------------------------
CATEGORY_ORDER: List[str] = [
    "Übersicht",
    "Grundlagen",
    "Bildverarbeitung",
    "Deep Learning",
    "State-of-the-Art",
    "Praxis",
    "Live",
    "Referenz",
]

CATEGORY_ICONS = {
    "Übersicht":        "🏠",
    "Grundlagen":       "🧮",
    "Bildverarbeitung": "🖼️",
    "Deep Learning":    "🤖",
    "State-of-the-Art": "🔥",
    "Praxis":           "🚀",
    "Live":             "📰",
    "Referenz":         "📚",
}


# --------------------------------------------------------------------
# Alle Module — Reihenfolge = Lernreihenfolge innerhalb der Kategorie
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
    Module("tensor_playground", "🎮", "Tensor Playground",    "NumPy interaktiv: Reshape, Einsum, Broadcasting", "Grundlagen",    "Anfänger",    "30 min", ["math", "numpy", "core"]),

    # --- Bildverarbeitung ---
    Module("image_basics",        "🖼️", "Bildgrundlagen & Pixel",          "Was ist ein Bild? Farbräume, Sampling",      "Bildverarbeitung", "Anfänger",       "20 min", ["cv"]),
    Module("camera_pipeline",     "📷", "Wie eine Kamera ein Bild macht",  "Photonen, Sensor, Bayer, ISP und Grundpipeline", "Bildverarbeitung", "Anfänger",   "35 min", ["cv", "camera", "sensor", "isp"]),
    Module("filters",             "🌫️", "Filter & Faltung",                 "Convolution, Gauß, Median, Bilateral",       "Bildverarbeitung", "Anfänger",       "30 min", ["cv", "core"]),
    Module("edges",               "📏", "Kantendetektion",                 "Sobel, Canny, Laplace — interaktiv",         "Bildverarbeitung", "Fortgeschritten", "25 min", ["cv"]),
    Module("features",            "🔑", "Feature Detection & Matching",    "SIFT, ORB, Harris, Keypoints",               "Bildverarbeitung", "Fortgeschritten", "40 min", ["cv"]),
    Module("morphology",          "🧱", "Morphologie",                     "Erosion, Dilatation, Opening, Closing",      "Bildverarbeitung", "Fortgeschritten", "20 min", ["cv"]),
    Module("segmentation_classic","✂️", "Klassische Segmentierung",        "Threshold, Watershed, GrabCut, K-Means",     "Bildverarbeitung", "Fortgeschritten", "30 min", ["cv"]),
    Module("optical_flow",        "🌬️", "Optical Flow & Motion",           "Lucas-Kanade, Farnebäck, RAFT, Motion",      "Bildverarbeitung", "Fortgeschritten", "35 min", ["cv", "motion", "flow"]),
    Module("object_tracking",      "🎯", "Objekterkennung & Tracking",      "YOLO, DETR, NMS, mAP, MOT, ByteTrack",       "Bildverarbeitung", "Fortgeschritten", "55 min", ["cv", "detection", "tracking"]),

    # --- Deep Learning ---
    Module("nn_basics",    "🧠", "Neuronale Netze von Grund auf", "Perzeptron, MLP, Aktivierungen",              "Deep Learning",   "Anfänger",        "45 min", ["dl", "core"]),
    Module("cnn",          "🔲", "Convolutional Neural Networks",  "Von LeNet bis ResNet",                        "Deep Learning",   "Fortgeschritten", "60 min", ["dl", "core"]),
    Module("training",     "🎯", "Training, Loss & Optimizer",     "SGD, Adam, Regularisierung, Learning Rate",   "Deep Learning",   "Fortgeschritten", "45 min", ["dl"]),
    Module("modern_archs", "🏛️", "Moderne Architekturen",          "ResNet, EfficientNet, ConvNeXt, U-Net",        "Deep Learning",   "Experte",          "60 min", ["dl"]),
    Module("self_supervised", "🧪", "Self-Supervised Learning",      "SimCLR, MoCo, DINO, MAE und Label-Effizienz",  "Deep Learning",   "Experte",          "50 min", ["dl", "ssl"]),
    Module("video_understanding", "🎬", "Video Understanding",        "Action Recognition, Temporal Modeling, Video Transformer", "Deep Learning", "Experte", "55 min", ["dl", "video"]),

    # --- State-of-the-Art ---
    Module("transformers", "⚡", "Transformer & Attention",        "Self-Attention, ViT, Swin Transformer",        "State-of-the-Art", "Experte", "60 min", ["sota"]),
    Module("vlm",          "👁️‍🗨️", "Vision-Language Models",       "CLIP, BLIP-2, LLaVA, Flamingo",                "State-of-the-Art", "Experte", "50 min", ["sota"]),
    Module("diffusion",    "🌊", "Diffusion Models",                "DDPM, Stable Diffusion, Flow Matching",        "State-of-the-Art", "Experte", "55 min", ["sota"]),
    Module("gen_ai",       "🎨", "Generative KI",                   "GANs, VAEs, Autoregressive Modelle",           "State-of-the-Art", "Experte", "45 min", ["sota"]),
    Module("multimodal",   "🌐", "Multimodal & LLMs",               "GPT-5, Gemini, Sora — Bild+Text+Video",        "State-of-the-Art", "Experte", "40 min", ["sota"]),
    Module("vision_foundation", "🧭", "Vision Foundation Models",   "SAM 2, DINOv3, Depth Anything, CLIP, Grounding", "State-of-the-Art", "Experte", "45 min", ["sota", "foundation"]),
    Module("pose_estimation",   "🧍", "Pose Estimation",      "Human Pose, 6DoF, Skeleton-Visualizer",           "State-of-the-Art", "Experte",  "45 min", ["sota", "cv", "pose"]),
    Module("three_d_vision", "🧊", "3D Computer Vision",            "Kamera-Geometrie, Epipolar, SfM/SLAM, NeRF",   "State-of-the-Art", "Experte", "60 min", ["sota", "3d", "cv"]),
    Module("rag_multimodal_agents", "🛰️", "RAG + Multimodal Agents", "Vision-RAG, Tool-Use, Prompting, Guardrails",  "State-of-the-Art", "Experte", "50 min", ["sota", "rag", "agents"]),

    # --- Praxis ---
    Module("learning_studio", "🧪", "Lernstudio: Labs & Übungen", "Progressive Labs, Mischformat und Community", "Praxis", "Anfänger", "35 min", ["praxis", "labs", "community"]),
    Module("projects",   "💻", "Praxisprojekte",          "10+ Hands-on Projekte mit Code",          "Praxis", "Fortgeschritten", "varies", ["praxis"]),
    Module("datasets",   "📦", "Datasets & Tools",        "ImageNet, COCO, HuggingFace, Roboflow",   "Praxis", "Anfänger",        "20 min", ["praxis"]),
    Module("evaluation_robustness", "🛡️", "Evaluation & Robustness", "Calibration, OOD, Domain Shift, Bias/Fairness", "Praxis", "Fortgeschritten", "45 min", ["praxis", "evaluation", "robustness"]),
    Module("compression",       "🗜️", "Model Compression",    "Quantisierung, Pruning, KD, ONNX, TensorRT",      "Praxis",        "Experte",      "45 min", ["praxis", "edge", "deployment"]),
    Module("edge_ai",           "🔌", "Edge & Embedded CV",   "Jetson, TFLite, Core ML, On-Device-Inferenz",     "Praxis",        "Experte",      "40 min", ["praxis", "edge", "embedded"]),
    Module("deployment", "🚀", "Deployment & MLOps",      "ONNX, TensorRT, FastAPI, Docker",         "Praxis", "Experte",          "50 min", ["praxis"]),

    # --- Live ---
    Module("news",          "📰", "Live News",                "Aktuelle Forschung & Releases",                "Live",    "Anfänger",        "live",    ["live"]),
    Module("papers",        "📄", "Paper-Bibliothek",         "Must-Read Papers nach Thema",                  "Live",    "Experte",          "varies",  ["live"]),
    Module("paper_of_month","🗞️", "Paper des Monats",         "Kuratiertes Paper mit Deep-Dive & Video",      "Live",    "Fortgeschritten",  "30 min",  ["live", "paper"]),
    Module("resources",     "🔗", "Ressourcen & Tools",       "Bücher, Kurse, Frameworks, Communities",       "Live",    "Anfänger",         "varies",  ["live"]),

    # --- Referenz ---
    Module("glossar",       "📖", "Glossar & Wörterbuch",     "CV & AI Begriffe durchsuchen",                 "Referenz", "Anfänger",        "varies",  ["reference", "glossar"]),
]


# --------------------------------------------------------------------
# Abgeleitete Strukturen — automatisch aus MODULES gebaut
# --------------------------------------------------------------------
def _build_categories() -> dict:
    """CATEGORIES (Emoji-Key → [mod_id]) aus MODULES ableiten — keine Doppelpflege."""
    grouped: dict = {}
    for cat in CATEGORY_ORDER:
        key = f"{CATEGORY_ICONS[cat]} {cat}"
        ids = [m.id for m in MODULES if m.category == cat]
        if ids:
            grouped[key] = ids
    return grouped


CATEGORIES = _build_categories()

# Konsistenz-Check: jede Modul-Kategorie muss bekannt sein.
_unknown = sorted({m.category for m in MODULES} - set(CATEGORY_ORDER))
assert not _unknown, f"Unbekannte Kategorie(n) in MODULES: {_unknown}"


def get_module(mod_id: str) -> Module | None:
    """Modul nach ID finden."""
    return next((m for m in MODULES if m.id == mod_id), None)


def modules_by_category() -> dict:
    """Module gruppiert nach Kategorie (in CATEGORY_ORDER-Reihenfolge)."""
    grouped: dict = {cat: [] for cat in CATEGORY_ORDER}
    for m in MODULES:
        grouped[m.category].append(m)
    return {cat: mods for cat, mods in grouped.items() if mods}


def module_position(mod_id: str) -> tuple[int, int]:
    """1-basierte Position eines Moduls unter allen Nicht-Home-Modulen + Gesamtzahl."""
    trackable = [m.id for m in MODULES if m.id != "home"]
    total = len(trackable)
    if mod_id in trackable:
        return trackable.index(mod_id) + 1, total
    return 0, total
