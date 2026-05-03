"""Glossar — durchsuchbares Wörterbuch aller CV & KI Fachbegriffe."""
import streamlit as st
from src.components import hero, section_header, divider, info_box

# ---------------------------------------------------------------------------
# Glossar-Datenbank
# Struktur: (Begriff, Kurzerklärung, Kategorie, optionaler Modul-Link)
# ---------------------------------------------------------------------------
TERMS = [
    # Math
    ("Gradient", "Vektor aller partiellen Ableitungen einer Funktion — zeigt in die Richtung des steilsten Anstiegs.", "Mathematik", "calculus"),
    ("Backpropagation", "Algorithmus zur Berechnung aller Gradienten in einem Netz via Kettenregel — rückwärts durch den Graphen.", "Mathematik", "nn_basics"),
    ("Kettenregel", "d/dx[f(g(x))] = f'(g(x)) · g'(x). Basis von Backpropagation.", "Mathematik", "calculus"),
    ("Eigenwert", "Skalare λ mit Av = λv. Eigenwerte beschreiben wie stark eine Matrix in Richtung v streckt.", "Mathematik", "linalg"),
    ("Eigenvektoren", "Vektoren v, deren Richtung unter der Matrixtransformation A erhalten bleibt (nur Länge ändert sich).", "Mathematik", "linalg"),
    ("Matrixmultiplikation", "C[i,j] = Σ_k A[i,k] · B[k,j]. Die fundamentale Operation in allen neuronalen Netzen.", "Mathematik", "linalg"),
    ("Dot Product / Skalarprodukt", "a·b = Σ aᵢbᵢ = |a||b|cos(θ). Misst Ähnlichkeit und ist Basis von Attention.", "Mathematik", "linalg"),
    ("Kosinusähnlichkeit", "cos(θ) = a·b / (|a|·|b|). Normiertes Skalarprodukt für Ähnlichkeitsvergleiche unabhängig von der Norm.", "Mathematik", "linalg"),
    ("Gradient Descent", "Iterative Optimierung: θ ← θ - η·∇L. Parameter in Richtung des negativen Gradienten aktualisieren.", "Mathematik", "calculus"),
    ("Stochastic Gradient Descent (SGD)", "Gradient nur auf einem Mini-Batch berechnen. Schneller, stochastische Regularisierung.", "Training", "training"),
    ("Bayes-Theorem", "P(A|B) = P(B|A)·P(A) / P(B). Fundamentales Theorem für probabilistische Inferenz.", "Mathematik", "probability"),
    ("Prior", "Vorwissen über einen Parameter vor der Beobachtung von Daten.", "Mathematik", "probability"),
    ("Posterior", "Aktualisierte Wahrscheinlichkeit nach Beobachtung von Daten.", "Mathematik", "probability"),
    ("KL-Divergenz", "D_KL(P||Q) = ΣP(x)log(P(x)/Q(x)). Misst Unterschied zweier Verteilungen.", "Mathematik", "probability"),
    ("Cross-Entropy", "H(y,ŷ) = -Σ y log(ŷ). Standard-Verlustfunktion für Klassifikation.", "Mathematik", "training"),
    ("Konvexe Funktion", "Funktion, bei der jedes Chord-Segment über der Kurve liegt. Gradient Descent findet globales Minimum.", "Mathematik", "calculus"),
    ("Regularisierung (L1/L2)", "Strafterm auf Gewichte: L1 erzeugt Sparsität, L2 hält Gewichte klein.", "Training", "training"),
    ("Tensor", "Verallgemeinerung von Skalar (0D), Vektor (1D), Matrix (2D) auf beliebige Dimensionen.", "Mathematik", "linalg"),
    ("Einsum", "Einstein-Summations-Notation: 'ij,jk->ik' für Matmul. Universelle Notation für Tensor-Ops.", "Mathematik", "tensor_playground"),

    # Computer Vision
    ("Pixel", "Picture Element. Kleinste Einheit eines digitalen Bildes mit einem Intensitätswert pro Kanal.", "Bildverarbeitung", "image_basics"),
    ("Bittiefe", "Anzahl Bits pro Pixelwert. 8 bit = 256 Werte. Bestimmt Helligkeitsauflösung.", "Bildverarbeitung", "image_basics"),
    ("RGB", "Rot, Grün, Blau — additives Farbmodell. Standard für Displays und Kameras.", "Bildverarbeitung", "image_basics"),
    ("HSV", "Hue, Saturation, Value — Farbmodell für einfache Farbsegmentierung.", "Bildverarbeitung", "image_basics"),
    ("LAB", "Wahrnehmungslinearer Farbraum (L*a*b*). Euklidische Distanz ≈ wahrgenommener Farbunterschied.", "Bildverarbeitung", "image_basics"),
    ("Faltung / Convolution", "Kernel gleitet über Bild: (I★K)(y,x) = ΣΣ I(y+i,x+j)·K(i,j). Basis von CNNs.", "Bildverarbeitung", "filters"),
    ("Kernel / Filter", "Kleine Matrix (z.B. 3×3), die durch Faltung angewendet wird. Bestimmt den Effekt.", "Bildverarbeitung", "filters"),
    ("Gauß-Filter", "Gewichteter Blur-Kernel nach Gaußverteilung. Verhindert Aliasing, reduziert Rauschen.", "Bildverarbeitung", "filters"),
    ("Sobel-Operator", "3×3 Kernel für horizontale/vertikale Kantendetektion. Diskrete Ableitung.", "Bildverarbeitung", "edges"),
    ("Canny Edge Detector", "5-Schritt Kantendetektion: Gauß → Sobel → NMS → Doppel-Schwellwert → Hysterese.", "Bildverarbeitung", "edges"),
    ("Non-Maximum Suppression (NMS)", "Behält nur lokale Maxima. Bei Kanten: dünnt Kanten auf 1 Pixel. Bei Detection: entfernt doppelte Boxen.", "Bildverarbeitung", "edges"),
    ("Laplace-Operator", "Zweite Ableitung des Bildes. Kanten erscheinen als Nulldurchgänge.", "Bildverarbeitung", "edges"),
    ("SIFT", "Scale-Invariant Feature Transform. Robuste Keypoints + Deskriptoren — invariant gegen Skalierung, Rotation.", "Bildverarbeitung", "features"),
    ("ORB", "Oriented FAST and Rotated BRIEF. Schnelle, kostenlose Alternative zu SIFT.", "Bildverarbeitung", "features"),
    ("Bounding Box", "Achsenparalleles Rechteck um ein Objekt — definiert durch (x, y, w, h) oder (x1, y1, x2, y2).", "Object Detection", None),
    ("IoU (Intersection over Union)", "Überlappung zweier Bounding Boxes: |A∩B| / |A∪B|. Standard-Metrik für Detection/Segmentation.", "Object Detection", None),
    ("Morphologie", "Bildoperationen auf binären Bildern: Erosion (schrumpfen), Dilatation (wachsen).", "Bildverarbeitung", "morphology"),
    ("Erosion", "Morphologische Op: Pixel bleibt nur wenn der ganze Kernel-Bereich 1 ist. Schrumpft Objekte.", "Bildverarbeitung", "morphology"),
    ("Dilatation", "Morphologische Op: Pixel wird 1 wenn mindestens ein Kernel-Pixel 1 ist. Vergrößert Objekte.", "Bildverarbeitung", "morphology"),
    ("Padding", "Rand um Bild hinzufügen (Nullen, Reflect…) um Ausgabegröße nach Faltung zu erhalten.", "Bildverarbeitung", "filters"),
    ("Stride", "Schrittweite des Kernels. Stride=2 → Output halb so groß wie Input.", "Bildverarbeitung", "cnn"),
    ("Feature Map", "Ausgabe eines Conv-Layers — 2D-Aktivierungskarte für ein bestimmtes Merkmal.", "Deep Learning", "cnn"),
    ("Receptive Field", "Bereich im Eingabebild, der einen Output-Pixel beeinflusst. Wächst mit Tiefe des Netzes.", "Deep Learning", "cnn"),
    ("Aliasing / Moiré", "Artefakt bei zu niedrigem Sampling. Nyquist: sampeln mit ≥ 2× der Maximalfrequenz.", "Bildverarbeitung", "image_basics"),
    ("Optical Flow", "Scheinbare Bewegung von Bildpunkten zwischen Frames. Grundlage für Video-Analyse.", "Bildverarbeitung", None),
    ("Depth Estimation", "Schätzung der 3D-Tiefe aus 2D-Bildern. Monocular (1 Bild) oder Stereo (2 Kameras).", "Computer Vision", None),

    # Deep Learning
    ("Perzeptron", "Einfachstes künstliches Neuron: y = σ(w·x + b). Kann nur linear trennbare Probleme lösen.", "Deep Learning", "nn_basics"),
    ("MLP (Multi-Layer Perceptron)", "Mehrere Schichten von Neuronen. Kann beliebige Funktionen approximieren (Universal Approximation).", "Deep Learning", "nn_basics"),
    ("Aktivierungsfunktion", "Nichtlinearität zwischen Layern. Ohne: komplettes MLP = eine lineare Funktion.", "Deep Learning", "nn_basics"),
    ("ReLU", "Rectified Linear Unit: max(0, x). Schnell, kein Vanishing Gradient für x>0. Standard in CNNs.", "Deep Learning", "nn_basics"),
    ("GELU", "Gaussian Error Linear Unit: x·Φ(x). Glatter als ReLU. Standard in BERT, GPT, ViT.", "Deep Learning", "nn_basics"),
    ("Softmax", "Normiert Logits zu Wahrscheinlichkeiten: exp(xᵢ)/Σexp(xⱼ). Immer im Output-Layer für Klassifikation.", "Deep Learning", "nn_basics"),
    ("Vanishing Gradient", "Gradienten werden mit der Tiefe exponentiell kleiner. Problem bei Sigmoid/Tanh in tiefen Netzen.", "Deep Learning", "nn_basics"),
    ("Batch Normalization (BN)", "Normalisiert Aktivierungen per Mini-Batch. Stabileres Training, höhere LR möglich.", "Deep Learning", "cnn"),
    ("Layer Normalization (LN)", "Normalisiert entlang Feature-Dim statt Batch. Standard in Transformers.", "Deep Learning", "transformers"),
    ("Dropout", "Zufällig Aktivierungen auf 0 setzen während Training. Regularisierung, reduziert Overfitting.", "Training", "training"),
    ("Overfitting", "Modell lernt Trainingsdaten auswendig, generalisiert schlecht. Train-Loss sinkt, Val-Loss steigt.", "Training", "training"),
    ("Underfitting", "Modell zu einfach/LR zu klein. Sowohl Train- als auch Val-Loss hoch.", "Training", "training"),
    ("Transfer Learning", "Vortrainiertes Modell (z.B. ImageNet) als Startpunkt für neue Aufgabe. Viel effizienter.", "Training", "cnn"),
    ("Fine-Tuning", "Vortrainiertes Modell auf neuen Daten weitertrainieren — oft nur letzte Layer oder alle mit kleiner LR.", "Training", "cnn"),
    ("Learning Rate (LR)", "Schrittgröße bei Gradient Descent. Zu groß: divergiert. Zu klein: sehr langsam.", "Training", "training"),
    ("Weight Decay", "L2-Regularisierung auf Gewichte: L_total = L + λ||w||². Verhindert zu große Gewichte.", "Training", "training"),
    ("Adam", "Adaptiver Optimizer: Momentum + RMS-Prop. Standard für Deep Learning. AdamW entkoppelt Weight Decay.", "Training", "training"),
    ("Momentum", "Akkumuliert vorherige Gradienten für stabiler Update. v ← μv + ∇L, dann θ ← θ - η·v.", "Training", "training"),
    ("Mini-Batch", "Teilmenge des Datensatzes für einen Gradient-Update-Schritt. Typisch 32–512.", "Training", "training"),
    ("Epoch", "Ein kompletter Durchgang durch den gesamten Trainingsdatensatz.", "Training", "training"),
    ("Early Stopping", "Training abbrechen wenn Validation-Loss nicht mehr sinkt. Einfachste Regularisierung.", "Training", "training"),
    ("Data Augmentation", "Künstliche Erweiterung des Datasets durch Transformationen (Flip, Crop, Rotation, Color Jitter).", "Training", "training"),
    ("Label Smoothing", "Statt harten One-Hot Labels: [0.05, 0.9, 0.05]. Verhindert overconfident Predictions.", "Training", "training"),
    ("Mixed Precision (FP16)", "Training mit 16-bit Float statt 32-bit. 2× schneller, 2× weniger Speicher.", "Training", None),
    ("Gradient Clipping", "Gradient-Norm auf Maximalwert kappen. Verhindert explodierende Gradienten.", "Training", "training"),
    ("He-Initialisierung", "Gewichte mit std=√(2/n_in) initialisieren. Optimal für ReLU-Netze.", "Deep Learning", "nn_basics"),
    ("Xavier-Initialisierung", "Gewichte mit std=√(2/(n_in+n_out)). Für Sigmoid/Tanh-Netze.", "Deep Learning", "nn_basics"),

    # CNN
    ("Convolutional Layer", "Lernbare Kernel gleiten über Bild. Weight Sharing, Lokalität, Translationsäquivarianz.", "CNN", "cnn"),
    ("Pooling", "Reduktion der räumlichen Auflösung. Max-Pooling nimmt Maximum im Fenster.", "CNN", "cnn"),
    ("Global Average Pooling (GAP)", "Mittelt jeden Feature-Map auf einen Wert. Ersetzt Flatten+Dense — robuster.", "CNN", "cnn"),
    ("Residual Connection / Skip Connection", "F(x) + x. Gradient fließt direkt durch. Ermöglicht sehr tiefe Netze (ResNet).", "CNN", "cnn"),
    ("Bottleneck Block", "1×1 Conv zum Verringern, 3×3 Conv, 1×1 zum Vergrößern. Effizient (ResNet-50+).", "CNN", "cnn"),
    ("Depthwise Separable Convolution", "Räumliche + kanalweise Faltung getrennt. 8-9× effizienter. Basis von MobileNet.", "CNN", "cnn"),
    ("Receptive Field (effective)", "Effektiver Sichtbereich eines Output-Pixels im Eingabebild. Wächst nicht-linear mit Tiefe.", "CNN", "cnn"),
    ("ResNet", "Residual Network (He et al., 2015). Skip Connections ermöglichen 50-152+ Layer. Standard-Backbone.", "CNN", "cnn"),
    ("VGG", "Very Deep CNN (Simonyan 2014). Nur 3×3 Conv, aber sehr viele Layer. Klar und modular.", "CNN", "cnn"),
    ("AlexNet", "Erster großer CNN-Gewinner (ImageNet 2012). ReLU, Dropout, GPU-Training. Startschuss für Deep Learning.", "CNN", "cnn"),
    ("EfficientNet", "Compound Scaling (Tiefe+Breite+Auflösung). State-of-the-Art 2019-2021.", "CNN", "cnn"),
    ("ConvNeXt", "Modernisiertes CNN (2022). Nimmt Ideen von ViT (LayerNorm, GELU, Patchify Stem).", "CNN", "cnn"),

    # Transformer & Attention
    ("Attention", "Mechanismus: Query sucht Key, bekommt Value. QKᵀ/√d_k → Softmax → Gewichtet Values.", "Transformer", "transformers"),
    ("Self-Attention", "Q=K=V kommen aus derselben Sequenz. Jede Position sieht jede andere direkt.", "Transformer", "transformers"),
    ("Multi-Head Attention (MHA)", "H parallele Attention-Köpfe. Jeder lernt andere Beziehungstypen.", "Transformer", "transformers"),
    ("Cross-Attention", "Q kommt aus Decoder, K und V aus Encoder. Verbindet zwei Sequenzen.", "Transformer", "transformers"),
    ("Positional Encoding", "Addiert Positionsinformation zu Tokens, da Attention permutationsinvariant ist.", "Transformer", "transformers"),
    ("RoPE (Rotary Position Embedding)", "Rotiert Q und K abhängig von Position. Standard in LLaMA, GPT-NeoX.", "Transformer", "transformers"),
    ("ViT (Vision Transformer)", "Bild in 16×16 Patches → Tokens → Standard Transformer. (Dosovitskiy et al., 2020).", "Transformer", "transformers"),
    ("Swin Transformer", "Hierarchischer ViT mit lokaler Window-Attention. Linear statt quadratisch.", "Transformer", "transformers"),
    ("CLS-Token", "Spezielles Klassifikations-Token, das am Anfang angehängt wird. Repräsentiert das gesamte Bild/Satz.", "Transformer", "transformers"),
    ("Feed-Forward Network (FFN)", "Zwei Dense-Layer mit GELU in jedem Transformer-Block. d_model → 4×d_model → d_model.", "Transformer", "transformers"),
    ("Layer Norm (Pre-LN)", "Normalisierung vor MHA und FFN. Stabiler als Post-LN für tiefe Transformer.", "Transformer", "transformers"),
    ("KV-Cache", "Gespeicherte K und V für Auto-Regressive Generierung. Vermeidet Neuberechnung alter Tokens.", "Transformer", None),
    ("Flash Attention", "IO-bewusste Attention-Implementierung. 3-5× schneller, kein quadratischer Speicher.", "Transformer", None),
    ("Attention Entropy", "Maß für Fokussierung der Attention. Niedrig = fokussiert auf wenige Tokens. Hoch = gleichmäßig.", "Transformer", "transformers"),

    # Generative KI
    ("GAN", "Generative Adversarial Network. Generator vs Diskriminator. Training instabil. Basis von vielen Bildgeneratoren.", "Generative KI", "gen_ai"),
    ("VAE", "Variational Autoencoder. Kodiert in latenten Raum, sampelt und dekodiert. Basis von Stable Diffusion.", "Generative KI", "gen_ai"),
    ("Diffusion Model", "Forward: Rauschen addieren. Reverse: Rauschen schrittweise vorhersagen. Basis von SD, DALL-E.", "Generative KI", "diffusion"),
    ("DDPM", "Denoising Diffusion Probabilistic Models (Ho et al. 2020). Das grundlegende Diffusion-Framework.", "Generative KI", "diffusion"),
    ("DDIM", "Deterministische Variante von DDPM. 50 statt 1000 Schritte ohne Qualitätsverlust.", "Generative KI", "diffusion"),
    ("Latent Diffusion", "Diffusion im komprimierten VAE-Latentspace statt Pixel. Basis von Stable Diffusion.", "Generative KI", "diffusion"),
    ("CFG (Classifier-Free Guidance)", "Guidance-Scale w steuert Prompt-Treue bei Diffusion. Höher=treuer, aber weniger divers.", "Generative KI", "diffusion"),
    ("Flow Matching", "Lerne Vektorfeld das Rausch- zu Bild-Verteilung transportiert. Geradliniger als DDPM.", "Generative KI", "diffusion"),
    ("U-Net", "Encoder-Decoder mit Skip-Connections. Standard-Backbone für Diffusion-Modelle und Segmentierung.", "Architektur", "diffusion"),
    ("DiT (Diffusion Transformer)", "Transformer statt U-Net als Diffusion-Backbone. Sora, SD3, FLUX nutzen DiT.", "Generative KI", "diffusion"),

    # VLMs & Multimodal
    ("CLIP", "Contrastive Language-Image Pretraining (OpenAI). Verbindet Bild und Text in gemeinsamem Raum.", "VLM", "vlm"),
    ("Contrastive Learning", "Ähnliche Paare nah, unähnliche weit im Embedding-Raum. Basis von CLIP, SimCLR.", "VLM", "vlm"),
    ("Zero-Shot Learning", "Auf Klassen generalisieren, die im Training nicht gesehen wurden — nur über Textbeschreibung.", "VLM", "vlm"),
    ("Prompt Engineering", "Optimierung von Text-Eingaben für bessere Modell-Ausgaben ohne Gewichts-Update.", "VLM", None),
    ("LLaVA", "Large Language and Vision Assistant. CLIP Vision + LLM mit MLP-Bridge. Effizientes VLM.", "VLM", "vlm"),
    ("BLIP-2", "Q-Former Brücke zwischen frozen Bild-Encoder und frozen LLM. Sehr effizienter Ansatz.", "VLM", "vlm"),
    ("Instruction Tuning", "Fine-Tuning auf Instruktions-Folge-Daten. Macht Modelle hilfsbereit und 'chatty'.", "VLM", None),
    ("Multimodal LLM", "LLM der mehrere Modalitäten verarbeitet: Text, Bild, Audio, Video.", "VLM", "multimodal"),
    ("Grounding", "Verknüpfung von Text-Beschreibungen mit konkreten Bildregionen (Bounding Boxes).", "VLM", "vlm"),

    # Self-Supervised
    ("Self-Supervised Learning", "Training ohne Labels. Das Modell erstellt seine eigene Supervision aus Daten.", "Self-Supervised", None),
    ("Contrastive Learning", "Positive Paare (Augmentierungen desselben Bilds) nah, negative (andere Bilder) weit.", "Self-Supervised", None),
    ("MAE (Masked Autoencoder)", "Maskiere zufällige Patches, rekonstruiere sie. ViT Pretraining. He et al., 2021.", "Self-Supervised", None),
    ("DINO / DINOv2", "Self-Distillation: Student lernt von EMA-Teacher ohne Labels. Starke universelle Features.", "Self-Supervised", None),
    ("SimCLR", "Simple Contrastive Learning: zwei Augmentierungen → Ähnlichkeit maximieren. Chen et al., 2020.", "Self-Supervised", None),
    ("Knowledge Distillation", "Kleines Schüler-Modell lernt von großem Lehrer-Modell (Soft Labels, Feature Matching).", "Compression", "compression"),

    # Deployment & MLOps
    ("ONNX", "Open Neural Network Exchange. Plattformunabhängiges Modellformat für Deployment.", "Deployment", "deployment"),
    ("TensorRT", "NVIDIA Inference-Optimierung. 3-5× schneller auf GPU durch Quantisierung, Kernel-Fusion.", "Deployment", "deployment"),
    ("Quantisierung (INT8)", "Gewichte von FP32 auf 8-bit Integer reduzieren. 4× kleiner, 2-4× schneller.", "Compression", "compression"),
    ("Pruning", "Unwichtige Gewichte entfernen (auf 0 setzen). Reduziert Modellgröße.", "Compression", "compression"),
    ("Model Compression", "Oberbegriff für Quantisierung, Pruning, Destillation, Architecture Search.", "Compression", "compression"),
    ("Latenz", "Zeit für eine Inferenz-Anfrage. Entscheidend für Real-Time-Anwendungen.", "Deployment", "deployment"),
    ("Throughput", "Anfragen pro Sekunde (RPS). Entscheidend für Server-Deployment.", "Deployment", "deployment"),
    ("Edge AI", "KI direkt auf Endgerät (Smartphone, Kamera, MCU) statt in der Cloud.", "Deployment", "compression"),
    ("FLOP", "Floating Point Operation. Maß für Rechenaufwand. GFLOPs = 10⁹ FLOPs.", "Deployment", None),
    ("Latency vs Throughput", "Latenz: Zeit pro Request. Throughput: Requests/Sekunde. Trade-off bei Batch-Größe.", "Deployment", None),

    # Evaluation
    ("Accuracy", "Korrekt klassifizierte Bilder / Gesamtbilder. Irreführend bei unbalancierten Klassen.", "Evaluation", None),
    ("Precision", "Wie viele der als positiv vorhergesagten wirklich positiv sind. TP/(TP+FP).", "Evaluation", None),
    ("Recall / Sensitivity", "Wie viele der echten Positiven erkannt werden. TP/(TP+FN).", "Evaluation", None),
    ("F1-Score", "Harmonisches Mittel aus Precision und Recall. Gut für unbalancierte Datasets.", "Evaluation", None),
    ("mAP (mean Average Precision)", "Standard-Metrik für Object Detection. Mittelt AP über alle Klassen und IoU-Schwellen.", "Evaluation", None),
    ("FID (Fréchet Inception Distance)", "Qualitätsmetrik für generierte Bilder. Vergleicht Feature-Statistiken mit echten Bildern.", "Evaluation", None),
    ("Top-1/Top-5 Accuracy", "ImageNet Standard: Richtig wenn echte Klasse unter Top-1/Top-5 Vorhersagen.", "Evaluation", None),
    ("Confusion Matrix", "Tabelle: Zeilen=echte Klasse, Spalten=vorhergesagte Klasse. Zeigt alle Fehlerarten.", "Evaluation", None),

    # Pose Estimation
    ("Human Pose Estimation", "Lokalisierung von Körper-Keypoints (Gelenke) in Bildern oder Videos.", "Pose", "pose_estimation"),
    ("Keypoints", "Charakteristische Punkte eines Objekts (Schulter, Ellbogen, Knie…) als (x,y) Koordinaten.", "Pose", "pose_estimation"),
    ("Heatmap Regression", "Keypoints als 2D-Heatmaps vorhersagen, nicht direkt als Koordinaten. Robuster.", "Pose", "pose_estimation"),
    ("6DoF Pose", "6 Freiheitsgrade: 3D Position (x,y,z) + 3D Orientierung (Rotation). Für Robotik.", "Pose", "pose_estimation"),
    ("Skeleton", "Graph aus Keypoints verbunden durch Knochen-Kanten. Repräsentiert menschliche Körperstruktur.", "Pose", "pose_estimation"),

    # Misc
    ("Foundation Model", "Großes pretrained Modell als universelle Basis für viele Downstream-Tasks.", "Allgemein", None),
    ("Fine-Tuning", "Foundation Model auf spezifische Aufgabe anpassen — wenig Daten, wenig Compute nötig.", "Allgemein", "cnn"),
    ("Inference", "Modell auf neuen Daten anwenden (nach Training). Auch: Deployment.", "Allgemein", None),
    ("Hyperparameter", "Trainings-Konfiguration (LR, Batch Size, Modellgröße) — nicht durch Training gelernt.", "Training", "training"),
    ("Ablation Study", "Experiment das zeigt wie jede Komponente zum Gesamtergebnis beiträgt (eine nach der anderen entfernen).", "Allgemein", None),
    ("State-of-the-Art (SOTA)", "Bester bekannter Ansatz für eine spezifische Aufgabe auf einem Benchmark.", "Allgemein", None),
    ("Benchmark", "Standardisierter Test-Datensatz + Metrik zum Vergleich verschiedener Methoden.", "Allgemein", None),
    ("Autograd", "Automatische Differenzierung — berechnet Gradienten ohne manuelle Ableitung. Basis von PyTorch.", "Deep Learning", "nn_basics"),
    ("Tokenizer", "Wandelt Text in numerische Token-IDs. Basis aller LLMs. Unterschiedliche Strategien (BPE, WordPiece).", "NLP/LLM", None),
    ("Embedding", "Dichte Vektorrepräsentation von diskreten Objekten (Wörter, Patches, Klassen).", "Deep Learning", None),
    ("Logits", "Unnormalisierte Modell-Ausgaben vor Softmax. Können negative Werte sein.", "Deep Learning", None),
    ("Temperature Sampling", "Skalierung von Logits vor Softmax: höhere Temp = diversere Outputs, niedrige = greedy.", "LLM", None),
    ("LoRA", "Low-Rank Adaptation. Trainiere nur kleine Rang-Matrix statt ganzer Gewichte. 10-100× effizienter.", "Fine-Tuning", None),
    ("RLHF", "Reinforcement Learning from Human Feedback. Macht LLMs hilfsbereit (ChatGPT, Claude).", "LLM", None),
]

CATEGORIES = sorted(set(t[2] for t in TERMS))
MODULE_LINKS = {
    "image_basics": "Bildgrundlagen",
    "filters": "Filter & Faltung",
    "edges": "Kantendetektion",
    "features": "Feature Detection",
    "morphology": "Morphologie",
    "nn_basics": "Neuronale Netze",
    "cnn": "CNNs",
    "training": "Training",
    "transformers": "Transformer",
    "vlm": "VLMs",
    "diffusion": "Diffusion",
    "gen_ai": "Generative KI",
    "multimodal": "Multimodal",
    "linalg": "Lineare Algebra",
    "calculus": "Analysis",
    "probability": "Wahrscheinlichkeit",
    "deployment": "Deployment",
    "compression": "Model Compression",
    "pose_estimation": "Pose Estimation",
    "tensor_playground": "Tensor Playground",
}


def render():
    hero(
        eyebrow="Referenz · Wörterbuch",
        title="Glossar — CV &amp; KI Fachbegriffe",
        sub=f"Über {len(TERMS)} Fachbegriffe aus Computer Vision und KI — "
            "durchsuchbar, kategorisiert, mit direkten Links zu den Lernmodulen."
    )

    # Suche + Filter
    col_s, col_c = st.columns([3, 1])
    search = col_s.text_input("🔍 Begriff suchen", placeholder="z.B. Attention, ReLU, Diffusion...")
    cat_filter = col_c.selectbox("Kategorie", ["Alle"] + CATEGORIES)

    # Filtern
    filtered = TERMS
    if search:
        s = search.lower()
        filtered = [(t, e, c, m) for t, e, c, m in filtered
                    if s in t.lower() or s in e.lower()]
    if cat_filter != "Alle":
        filtered = [(t, e, c, m) for t, e, c, m in filtered if c == cat_filter]

    # Stats
    st.markdown(
        f"<div style='color:#9CA3AF;font-size:0.85rem;margin:0.5rem 0 1rem 0;'>"
        f"<b>{len(filtered)}</b> von <b>{len(TERMS)}</b> Begriffen angezeigt</div>",
        unsafe_allow_html=True,
    )

    if not filtered:
        info_box(f"Kein Begriff gefunden für '{search}'. Versuche einen anderen Suchbegriff.", kind="warn")
        return

    # Gruppiert nach Kategorie anzeigen
    from collections import defaultdict
    grouped = defaultdict(list)
    for term, expl, cat, mod in filtered:
        grouped[cat].append((term, expl, mod))

    for cat in sorted(grouped.keys()):
        terms_in_cat = grouped[cat]
        with st.expander(f"**{cat}** — {len(terms_in_cat)} Begriff{'e' if len(terms_in_cat) != 1 else ''}", expanded=bool(search)):
            for term, expl, mod in sorted(terms_in_cat, key=lambda x: x[0]):
                mod_badge = ""
                if mod and mod in MODULE_LINKS:
                    mod_badge = (f'<a style="display:inline-block;font-size:0.65rem;font-weight:700;'
                                 f'padding:0.1rem 0.45rem;border-radius:5px;background:rgba(59,130,246,0.15);'
                                 f'color:#93C5FD;text-decoration:none;margin-left:0.4rem;" '
                                 f'href="?">→ {MODULE_LINKS[mod]}</a>')
                st.markdown(
                    f"""<div style="padding:0.6rem 0;border-bottom:1px solid rgba(255,255,255,0.05);">
                      <span style="font-weight:700;color:#F3F4F6;">{term}</span>{mod_badge}
                      <div style="color:#9CA3AF;font-size:0.875rem;margin-top:0.2rem;line-height:1.5;">{expl}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

    divider()
    info_box(
        f"Das Glossar enthält {len(TERMS)} Begriffe und wächst regelmäßig. "
        "Fehlender Begriff? Schreibe einen Issue auf GitHub.",
        kind="tip",
    )
