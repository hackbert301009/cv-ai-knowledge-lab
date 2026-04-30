"""Praxisprojekte — 12 Projekte mit Code."""
import streamlit as st
from src.components import hero, section_header, divider, info_box, card, render_card_grid


def render():
    hero(
        eyebrow="Praxis · Modul 20",
        title="Praxisprojekte",
        sub="Theorie ist gut. Code ist besser. Hier sind 12 hands-on Projekte vom Anfänger- bis "
            "zum Fortgeschrittenen-Niveau — jeweils mit Setup, Code und Erweiterungen."
    )

    section_header("Übersicht: 12 Projekte für jedes Level")
    cards = [
        card("🎨", "1. Kanten-Detektor App", "OpenCV + Streamlit: Bild hochladen, Schwellwerte einstellen, Canny-Edges anzeigen.", ["Anfänger", "1–2h"], ["green", "blue"]),
        card("🐶", "2. Cat vs. Dog Klassifikator", "Klassiker. PyTorch + ResNet18 Transfer Learning auf Kaggle-Datensatz.", ["Anfänger", "3h"], ["green", "blue"]),
        card("🔢", "3. MNIST von Grund auf", "MLP und CNN selber bauen, ohne fertige Modelle. Lehrreich.", ["Anfänger", "4h"], ["green", "blue"]),
        card("🧐", "4. Object Detection mit YOLO", "Ultralytics YOLOv8 — pretrained und eigenes Dataset. Webcam-Inferenz.", ["Fortgeschritten", "1 Tag"], ["amber", "blue"]),
        card("🎭", "5. Style Transfer", "Neural Style Transfer (Gatys et al.). Künstlerische Bilder erzeugen.", ["Fortgeschritten", "1 Tag"], ["amber", "blue"]),
        card("👤", "6. Gesichtserkennung", "MTCNN + FaceNet Embeddings. Wer ist das? Cosine Similarity.", ["Fortgeschritten", "1 Tag"], ["amber", "blue"]),
        card("✂️", "7. Semantic Segmentation", "U-Net auf einem kleinen Dataset (z.B. Cityscapes-Subset).", ["Fortgeschritten", "2 Tage"], ["amber", "blue"]),
        card("🔮", "8. Image Captioning", "CLIP + GPT-Decoder. Bild → Beschreibung in natürlicher Sprache.", ["Experte", "2 Tage"], ["pink", "blue"]),
        card("🎨", "9. DCGAN-Pokémon", "GAN trainieren, Pokémon-artige Bilder erzeugen.", ["Experte", "2 Tage"], ["pink", "blue"]),
        card("🌊", "10. Mini-Diffusion-Model", "DDPM auf MNIST/CIFAR-10 von Grund auf. Verstehe das Innenleben.", ["Experte", "3 Tage"], ["pink", "blue"]),
        card("🔍", "11. Visual Search Engine", "CLIP-Embeddings + FAISS für 'Bilder suchen mit Worten'.", ["Experte", "1 Tag"], ["pink", "blue"]),
        card("🤖", "12. VLM-Anwendung", "LLaVA oder Qwen-VL — eigene Anwendung mit lokalem VLM bauen.", ["Experte", "2 Tage"], ["pink", "blue"]),
    ]
    render_card_grid(cards, cols=3)

    divider()

    section_header("Detailliertes Beispiel: Projekt #1 — Cat vs. Dog")
    st.markdown("""
**Setup**:
- Python 3.10+, PyTorch 2.x, torchvision
- Dataset: [Kaggle Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data)
- GPU empfohlen (Colab Free reicht)

**Was du lernst**: Transfer Learning, DataLoader, Training-Loop, Validation, Augmentation.
""")

    st.code("""
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Daten + Augmentation
train_tf = transforms.Compose([
    transforms.Resize(232),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder("data/train", transform=train_tf)
val_ds   = datasets.ImageFolder("data/val",   transform=val_tf)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=4)

# 2. Modell — ResNet18 vortrainiert, letzte Layer ersetzen
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# 3. Loss + Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# 4. Training Loop
for epoch in range(10):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    print(f"Epoch {epoch+1}: Val Acc = {correct/total:.3f}")
    scheduler.step()
    """, language="python")

    info_box(
        "**Erweiterungen**: Test-Time Augmentation, Ensemble mehrerer Modelle, Grad-CAM für Visualisierung "
        "der wichtigen Bildregionen, Modell mit ONNX exportieren.",
        kind="tip",
    )

    divider()

    section_header("Datasets & Tools für deine Projekte")
    st.markdown("""
- **[Kaggle](https://www.kaggle.com/datasets)** — riesige Bibliothek, viele Wettbewerbe
- **[HuggingFace Datasets](https://huggingface.co/datasets)** — schnell ladbar mit `datasets`-Library
- **[Roboflow Universe](https://universe.roboflow.com/)** — speziell für Detection/Segmentation
- **[Papers with Code Datasets](https://paperswithcode.com/datasets)** — verlinkt zu Papern
- **Tools**: PyTorch, torchvision, timm, transformers, ultralytics, opencv
""")
