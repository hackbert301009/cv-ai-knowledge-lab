"""Datasets & Tools."""
import streamlit as st
from src.components import hero, section_header, divider, info_box, card, render_card_grid


def render():
    hero(
        eyebrow="Praxis · Modul 21",
        title="Datasets &amp; Tools",
        sub="Saubere Daten sind 80% deines Erfolgs. Hier sind die wichtigsten öffentlichen Datasets, "
            "Tools zur Annotation und Best Practices."
    )

    section_header("Klassiker — die du kennen solltest")
    st.markdown("""
| Dataset | Größe | Wofür? | Notiz |
|---|---|---|---|
| **MNIST** | 70k | Klassifikation 0–9 | Hello World von CV |
| **CIFAR-10/100** | 60k | 10/100 Klassen, 32×32 | Schnell zu trainieren |
| **ImageNet-1k** | 1.3M | 1000-Klassen Klassifikation | Standard-Pretraining-Set |
| **ImageNet-21k** | 14M | 21k Klassen | Größere Pretraining-Variante |
| **COCO** | 330k Bilder | Detection, Segmentation, Captioning | 80 Objektklassen |
| **Cityscapes** | 25k Frames | Semantic Segmentation, Urban | Fahrzeug-relevant |
| **Pascal VOC** | 11k | Detection (klassisch) | Älter, aber Benchmark-Standard |
| **OpenImages** | 9M | Detection, Segmentation | Sehr groß, von Google |
| **LAION-5B** | 5.85B | Bild-Text-Paare | Trainiert Stable Diffusion, CLIP |
| **WebLI** | 12B | Multilingual Bild-Text | Google, intern |
""")

    divider()

    section_header("Spezial-Datasets nach Domäne")
    cards = [
        card("👤", "Faces", "CelebA-HQ, FFHQ — Gesichter für GANs, FaceNet's VGGFace2 für Recognition.", ["Faces"], ["pink"]),
        card("🩺", "Medizin", "ChestX-ray14, ISIC, MedMNIST, BraTS für Tumor-Segmentation.", ["Medical"], ["green"]),
        card("🚗", "Autonomous Driving", "KITTI, Waymo Open Dataset, nuScenes, BDD100K.", ["Driving"], ["blue"]),
        card("🛰️", "Satellite", "BigEarthNet, EuroSAT, SpaceNet — Erdbeobachtung.", ["Remote Sensing"], ["amber"]),
        card("🦴", "3D / Pose", "Human3.6M, MPII, COCO Keypoints, ShapeNet.", ["3D"], ["pink"]),
        card("🎬", "Video", "Kinetics-700, UCF101, ActivityNet, Something-Something.", ["Video"], ["pink"]),
    ]
    render_card_grid(cards, cols=3)

    divider()

    section_header("Tools für Annotation & Datenmanagement")
    st.markdown("""
- **[Roboflow](https://roboflow.com)** — Web-basiert, super UX, exportiert in alle gängigen Formate
- **[Label Studio](https://labelstud.io)** — Open Source, sehr flexibel, viele Modalitäten
- **[CVAT](https://cvat.ai)** — auch Open Source, fokussiert auf Video und Bild
- **[FiftyOne](https://voxel51.com/fiftyone)** — Dataset-Exploration und -Curation, sehr mächtig
- **[Encord](https://encord.com)** — kommerziell, fortgeschrittene Workflows
""")

    divider()

    section_header("Best Practices")
    st.markdown("""
1. **Bevor du trainierst**: 100 Bilder visuell durchschauen. Sind die Labels korrekt? Sind die Klassen ausgewogen?
2. **Train/Val/Test splitten** — vor jeder anderen Verarbeitung. Niemals durch's Augmentieren mischen.
3. **Stratified Splitting** — bei unbalancierten Klassen unbedingt sorgen, dass jede Klasse in jedem Split vertreten ist.
4. **Augmentation testen** — visualisiere ein Batch nach Augmentation. Sind die Bilder noch sinnvoll?
5. **Klassenungleichgewicht behandeln** — `WeightedRandomSampler`, Klassengewichte im Loss, Focal Loss
6. **Datenversion** — Versioniere deine Datensätze (DVC, Roboflow, FiftyOne)
""")

    info_box(
        "Faustregel: Wenn dein Modell schlecht performt, ist die Wahrscheinlichkeit am größten, dass das Problem in den Daten liegt. "
        "Nicht im Modell, nicht in den Hyperparametern. **Schau dir die Daten an.**",
        kind="warn",
    )
