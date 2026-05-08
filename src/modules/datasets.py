"""Datasets & Tools."""
import streamlit as st
from src.components import hero, section_header, divider, info_box, card, render_card_grid, render_learning_block


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

    divider()
    render_learning_block(
        key_prefix="datasets",
        progression=[
            ("🟢", "Guided Lab", "Datensatz auditieren: Klassen, Labels, Qualität, Splits.", "Beginner", "green"),
            ("🟠", "Challenge Lab", "Unbalancierte Klassen mit sauberen Gegenmaßnahmen stabilisieren.", "Intermediate", "amber"),
            ("🔴", "Debug Lab", "Data Leakage und Label-Noise identifizieren und beheben.", "Advanced", "pink"),
            ("🏁", "Mini-Projekt", "Dataset Card + reproduzierbare Data-Pipeline erstellen.", "Abschluss", "blue"),
        ],
        mcq_question="Was sollte vor Augmentation und Training immer zuerst passieren?",
        mcq_options=["Hyperparameter-Suche", "Train/Val/Test-Split", "Model Compression", "Deployment-Setup"],
        mcq_correct_option="Train/Val/Test-Split",
        mcq_success_message="Richtig. Sonst riskierst du Data Leakage.",
        mcq_retry_message="Nicht korrekt. Prüfe die Best-Practices.",
        open_question="Offene Frage: Welche zwei Checks würdest du für Label-Qualität einführen?",
        code_task="""# Code-Aufgabe: Stratified Split
from sklearn.model_selection import train_test_split
X, y = ...
# TODO: erst train/temp, dann val/test mit stratify=y aufteilen
""",
        community_rows=[
            {"Format": "Diskussion", "Fokus": "Wo entstehen bei euch die meisten Datenfehler?", "Output": "Kurzbeispiel"},
            {"Format": "Peer-Feedback", "Fokus": "Sind Splits und Label-Checks reproduzierbar?", "Output": "2 Pluspunkte + 1 Risiko"},
            {"Format": "Challenge", "Fokus": "Bestes Ergebnis ohne Data-Leakage", "Output": "Data-Checklist"},
        ],
        cheat_sheet=[
            "Daten erst prüfen, dann trainieren.",
            "Splits reproduzierbar machen.",
            "Klassenbalance und Labelqualität aktiv monitoren.",
        ],
        key_takeaways=[
            "Datenqualität limitiert Modellqualität.",
            "Gute Doku (Dataset Card) spart später massiv Zeit.",
        ],
        common_errors=[
            "Data Leakage.",
            "Kein stratified split.",
            "Label-Noise ignoriert.",
            "Augmentation nicht validiert.",
            "Keine Datenversionierung.",
        ],
    )
