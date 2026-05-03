"""Model Compression & Edge AI — Quantisierung, Pruning, Knowledge Distillation, ONNX, TensorRT."""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from src.components import hero, section_header, divider, info_box, video_embed, lab_header, key_concept, step_list


def render():
    hero(
        eyebrow="Praxis · Modul Compression",
        title="Model Compression & Edge AI",
        sub="Vom Labor zum Produkt: Modelle kleiner, schneller, effizienter machen — "
            "ohne die Genauigkeit zu opfern. Quantisierung, Pruning, Knowledge Distillation, ONNX, TensorRT.",
    )

    tabs = st.tabs([
        "🎯 Überblick",
        "🔢 Quantisierung",
        "✂️ Pruning",
        "👨‍🏫 Knowledge Distillation",
        "📦 ONNX & Export",
        "⚡ TensorRT & Edge",
        "🧪 Live-Demo",
        "🎬 Lernvideos",
    ])

    # ── Tab 0: Überblick ──────────────────────────────────────────────────────
    with tabs[0]:
        section_header("Warum Model Compression?")
        st.markdown("""
Ein ResNet-50 hat **~25 Mio. Parameter** und braucht **~4 GB GPU RAM** für FP32-Training.
Auf einem Smartphone oder Raspberry Pi ist das unmöglich.

**Ziel**: Gleiche (oder fast gleiche) Accuracy, aber:
- 🪶 Weniger Parameter & Speicher
- ⚡ Schnellere Inferenz
- 🔋 Geringerer Energieverbrauch
- 📱 Deployment auf Edge-Devices
        """)

        cols = st.columns(4)
        metrics = [
            ("Quantisierung INT8", "4×", "weniger Speicher"),
            ("Pruning 90%", "10×", "weniger Gewichte"),
            ("Knowledge Distillation", "10×", "kleineres Modell"),
            ("TensorRT", "5×", "schnellere Inferenz"),
        ]
        for col, (name, val, unit) in zip(cols, metrics):
            col.metric(name, val, unit)

        divider()

        key_concept("🔢", "Quantisierung",
                    "Gewichte von FP32 (4 Bytes) auf INT8 (1 Byte) reduzieren — 4× Speicher, 2–4× schneller.")
        key_concept("✂️", "Pruning",
                    "Unwichtige Neuronen/Verbindungen auf Null setzen oder entfernen. Sparsity bis 90% möglich.")
        key_concept("👨‍🏫", "Knowledge Distillation",
                    "Ein kleines 'Student'-Modell lernt vom großen 'Teacher'-Modell — Soft Targets statt Hard Labels.")
        key_concept("📦", "ONNX",
                    "Open Neural Network Exchange — portables Format, das PyTorch, TensorFlow und TensorRT verbindet.")
        key_concept("⚡", "TensorRT",
                    "NVIDIAs Inference-Optimierer: Layer Fusion, Kernel Auto-Tuning, INT8/FP16 — bis 5× Speedup.")

        divider()
        st.markdown("""
#### Compression-Techniken im Vergleich

| Methode | Speicher ↓ | Latenz ↓ | Accuracy-Verlust | Training nötig? |
|---------|-----------|----------|-----------------|-----------------|
| INT8 Quantisierung | ~4× | ~2–4× | <1% | Nein (PTQ) / Wenig (QAT) |
| FP16 / BF16 | ~2× | ~1.5–2× | minimal | Nein |
| Pruning (unstrukturiert) | ~2–10× | variabel | 1–3% | Fine-tuning |
| Pruning (strukturiert) | ~2–5× | ~2–5× | 1–5% | Fine-tuning |
| Knowledge Distillation | ~5–50× | ~5–50× | 2–10% | Vollständig |
| Low-Rank Approximation | ~2–10× | ~2–5× | 1–5% | Fine-tuning |
        """)

    # ── Tab 1: Quantisierung ─────────────────────────────────────────────────
    with tabs[1]:
        section_header("Quantisierung — FP32 → INT8")
        st.markdown(r"""
**Quantisierung** mappt Fließkommazahlen auf Integer mit begrenztem Wertebereich:

$$q = \text{round}\left(\frac{x}{S}\right) + Z$$

- $S$ = Scale Factor (Skalierung)
- $Z$ = Zero Point (Verschiebung)
- Rückrechnung: $\hat{x} = S \cdot (q - Z)$

#### Post-Training Quantization (PTQ)
Kein Re-Training — Gewichte werden nach dem Training quantisiert.
Braucht **Kalibrierungsdaten** (kleines Dataset), um den Wertebereich zu schätzen.

#### Quantization-Aware Training (QAT)
Quantisierungsfehler wird **während des Trainings** simuliert → bessere Accuracy,
aber teurer.
        """)
        st.code("""
# PyTorch PTQ (Post-Training Static Quantization)
import torch
from torch.quantization import quantize_dynamic

model = MyModel().eval()

# Dynamische Quantisierung — einfachste Variante
quant_model = quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8,
)
print(f"Original: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

# Static Quantization (genauer)
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
# Kalibrierung mit repräsentativen Daten:
with torch.no_grad():
    for batch in calibration_loader:
        model(batch)
torch.quantization.convert(model, inplace=True)
        """, language="python")

        divider()
        section_header("INT4 — noch kleiner")
        st.markdown("""
Moderne LLMs nutzen **GPTQ, AWQ, bitsandbytes** für INT4/NF4-Quantisierung:
- LLaMA-2 70B FP16: **140 GB** → INT4: **~35 GB**
- Läuft auf einer Consumer-GPU (RTX 4090, 24 GB) mit Offloading
        """)
        st.code("""
# HuggingFace bitsandbytes INT4
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",        # NormalFloat4 — besser als uniform INT4
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,   # Nested quantization
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=bnb_config,
)
        """, language="python")

        info_box("FP16 = 2 Bytes · INT8 = 1 Byte · INT4 = 0.5 Bytes · FP32 = 4 Bytes", kind="info")

    # ── Tab 2: Pruning ───────────────────────────────────────────────────────
    with tabs[2]:
        section_header("Pruning — Verbindungen kappen")
        st.markdown(r"""
**Pruning** entfernt unwichtige Gewichte. Kriterium für "unwichtig": kleiner Absolutwert.

#### Unstrukturiertes Pruning
Einzelne Gewichte → $w_{ij} = 0$. Erzeugt **sparse** Gewichtsmatrizen.
Gut für Speicher, aber schwer auf Standard-Hardware zu beschleunigen.

#### Strukturiertes Pruning
Ganze **Filter, Channel, Layer** entfernen → echte Speedups auf jeder Hardware.
- **L1-Filter-Pruning**: Filter mit kleinstem L1-Norm entfernen
- **Channel Pruning**: Ganze Channels streichen

#### Lottery Ticket Hypothesis (Frankle & Carlin 2019)
> Ein großes Netz enthält ein kleines "Gewinner"-Teilnetz, das alleine genauso gut trainierbar ist.

Iteratives Pruning + Reset findet dieses Teilnetz.
        """)
        st.code("""
# PyTorch Pruning — Magnitude-based
import torch.nn.utils.prune as prune

# 30% der kleinsten Gewichte auf 0 setzen
prune.l1_unstructured(model.conv1, name='weight', amount=0.3)

# Strukturiertes Filter-Pruning
prune.ln_structured(model.conv1, name='weight', amount=0.3, n=2, dim=0)

# Pruning permanent machen
prune.remove(model.conv1, 'weight')

# Sparsity messen
total = sum(t.numel() for t in model.parameters())
zeros = sum((t == 0).sum() for t in model.parameters())
print(f"Sparsity: {100 * zeros / total:.1f}%")
        """, language="python")

        divider()
        section_header("Iteratives Magnitude Pruning (IMP)")
        step_list([
            ("Train", "Modell normal trainieren bis zur Ziel-Accuracy."),
            ("Prune", "X% der kleinsten Gewichte (nach Betrag) auf 0 setzen."),
            ("Fine-tune", "Kurzes Fine-tuning, um Accuracy zu erholen."),
            ("Wiederholen", "Bis zur gewünschten Sparsity. 90% möglich mit <1% Accuracy-Verlust."),
            ("Deploy", "Sparse-Modell exportieren. NVIDIA CUDA unterstützt 2:4 structured sparsity nativ."),
        ])

    # ── Tab 3: Knowledge Distillation ────────────────────────────────────────
    with tabs[3]:
        section_header("Knowledge Distillation")
        st.markdown(r"""
**Idee (Hinton et al. 2015)**: Statt einen Student mit One-Hot-Labels zu trainieren,
nutze die **Soft Predictions des Teachers** — sie enthalten mehr Information.

$$\mathcal{L}_{KD} = \alpha \cdot \mathcal{L}_{CE}(y, \hat{y}_s) + (1-\alpha) \cdot T^2 \cdot \mathcal{L}_{KL}\!\left(\sigma\!\left(\frac{z_t}{T}\right),\, \sigma\!\left(\frac{z_s}{T}\right)\right)$$

- $T$ = Temperatur (> 1 macht Verteilung weicher)
- $\alpha$ = Balance zwischen Hard und Soft Loss
- $z_t, z_s$ = Logits von Teacher und Student

#### Warum "soft" Labels besser sind
Ein Cat-Classifier gibt vielleicht `[cat: 0.9, lynx: 0.08, dog: 0.02]` aus.
Das sagt dem Student: "Katzen und Luchse sehen ähnlich aus."
One-Hot `[1, 0, 0]` sagt das nicht.

#### Varianten
- **Feature Distillation**: Student lernt interne Repräsentationen des Teachers
- **Attention Transfer**: Attention-Maps des Teachers nachahmen
- **Self-Distillation**: Tiefere Schichten lernen von früheren (Born Again Networks)
        """)
        st.code("""
# PyTorch Knowledge Distillation
import torch
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels,
                       temperature=4.0, alpha=0.7):
    # Hard Loss (Student vs. Ground Truth)
    hard_loss = F.cross_entropy(student_logits, labels)

    # Soft Loss (Student vs. Teacher — weiche Verteilungen)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')

    return alpha * hard_loss + (1 - alpha) * temperature**2 * soft_loss

# Training Loop
teacher.eval()
for images, labels in loader:
    with torch.no_grad():
        teacher_logits = teacher(images)
    student_logits = student(images)
    loss = distillation_loss(student_logits, teacher_logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
        """, language="python")

        info_box(
            "BERT-base (110M) → DistilBERT (66M): 40% kleiner, 60% schneller, 97% der Accuracy erhalten. "
            "Das ist Knowledge Distillation in Aktion.",
            kind="success",
        )

    # ── Tab 4: ONNX ──────────────────────────────────────────────────────────
    with tabs[4]:
        section_header("ONNX — Open Neural Network Exchange")
        st.markdown("""
**ONNX** ist ein offenes Format für ML-Modelle — eine gemeinsame Sprache zwischen Frameworks.

```
PyTorch  ──┐
TensorFlow ─┤ → ONNX Graph → TensorRT / CoreML / OpenVINO / ONNX Runtime
JAX       ──┘
```

#### Vorteile
- **Framework-agnostisch**: Einmal exportieren, überall deployen
- **Optimierungen**: ONNX Runtime macht automatische Graph-Optimierungen
- **Hardware-spezifisch**: TensorRT (NVIDIA), CoreML (Apple), OpenVINO (Intel)
        """)
        st.code("""
# PyTorch → ONNX Export
import torch
import torch.onnx

model = MyModel().eval()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input":  {0: "batch_size"},   # batch dim dynamisch
        "output": {0: "batch_size"},
    },
)
print("✅ ONNX Export fertig!")

# Verifikation
import onnx
model_onnx = onnx.load("model.onnx")
onnx.checker.check_model(model_onnx)

# Inferenz mit ONNX Runtime
import onnxruntime as ort
session = ort.InferenceSession("model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
outputs = session.run(None, {"input": dummy_input.numpy()})
        """, language="python")

        divider()
        section_header("ONNX Graph Optimierungen")
        step_list([
            ("Constant Folding", "Konstante Teilgraphen werden zur Compile-Zeit berechnet."),
            ("Layer Fusion", "BatchNorm + ReLU → ein Operator. Weniger Kernel-Launches."),
            ("Dead Node Elimination", "Ungenutzte Berechnungen entfernen."),
            ("Graph Partitioning", "Teile auf GPU, Teile auf CPU ausführen."),
        ])

    # ── Tab 5: TensorRT & Edge ───────────────────────────────────────────────
    with tabs[5]:
        section_header("TensorRT — NVIDIAs Inference-Engine")
        st.markdown("""
**TensorRT** optimiert ONNX-Modelle für NVIDIA-GPUs:
- Layer Fusion (Conv + BN + ReLU → 1 Op)
- Kernel Auto-Tuning (beste CUDA-Implementierung für GPU + Input-Shape)
- INT8/FP16-Präzision
- Dynamic Shapes
        """)
        st.code("""
# TensorRT aus ONNX bauen
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open("model.onnx", "rb") as f:
    parser.parse(f.read())

config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)          # FP16 aktivieren
config.set_flag(trt.BuilderFlag.INT8)          # INT8 aktivieren

# INT8 Kalibrierung benötigt Calibrator-Klasse
# config.int8_calibrator = MyCalibrator(calibration_loader)

engine = builder.build_serialized_network(network, config)
with open("model.trt", "wb") as f:
    f.write(engine)

# Inferenz
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(open("model.trt", "rb").read())
context = engine.create_execution_context()
        """, language="python")

        divider()
        section_header("Edge AI — Deployment Targets")
        st.markdown("""
| Platform | Framework | Precision | Typische Latenz |
|----------|-----------|-----------|-----------------|
| NVIDIA Jetson | TensorRT | FP16/INT8 | 5–50ms |
| Raspberry Pi | ONNX Runtime / TFLite | FP32/INT8 | 100–500ms |
| Apple Silicon | CoreML | FP16/ANE | 1–10ms |
| Android | TFLite | INT8 | 10–100ms |
| Intel (CPU) | OpenVINO | INT8 | 10–100ms |
| Coral (Google) | TFLite + EdgeTPU | INT8 | <1ms |

#### Tipps für Edge Deployment
- Profiling zuerst: Wo sind die Bottlenecks?
- Batch-Size 1 für Echtzeit-Inferenz
- Quantisierung bringt den größten Gewinn
- Modellgröße vs. Accuracy: Pareto-Front kennen
        """)
        info_box(
            "MobileNet (4.2M params, 569KB INT8) schlägt VGG-16 (138M params, 528MB) "
            "auf einem Smartphone in Latenz um den Faktor 100 — bei nur 3% weniger Accuracy.",
            kind="info",
        )

    # ── Tab 6: Live-Demo ─────────────────────────────────────────────────────
    with tabs[6]:
        lab_header("Quantisierungs-Fehler Simulator", "Wie viel Genauigkeit verliert man?")

        c1, c2 = st.columns([1, 2])
        with c1:
            bits = st.select_slider("Bit-Breite", options=[2, 3, 4, 6, 8, 16, 32], value=8)
            signal_type = st.selectbox("Signal", ["Sinus", "Rauschen", "Bild-Gewichte (Normalverteilung)"])
            n_samples = st.slider("Samples", 100, 2000, 500)

        rng = np.random.default_rng(42)
        x = np.arange(n_samples) / n_samples
        if signal_type == "Sinus":
            signal = np.sin(2 * np.pi * 3 * x) + 0.1 * rng.standard_normal(n_samples)
        elif signal_type == "Rauschen":
            signal = rng.standard_normal(n_samples)
        else:
            signal = rng.standard_normal(n_samples) * 0.5  # Typische Gewichtsverteilung

        # Quantisierung simulieren
        levels = 2 ** bits
        vmin, vmax = signal.min(), signal.max()
        scale = (vmax - vmin) / (levels - 1)
        quantized = np.round((signal - vmin) / scale) * scale + vmin
        error = signal - quantized
        snr = 10 * np.log10(np.var(signal) / np.var(error)) if np.var(error) > 0 else float('inf')
        mse = np.mean(error ** 2)

        with c2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=signal[:200], name="Original (FP32)", line=dict(color="#7C3AED")))
            fig.add_trace(go.Scatter(y=quantized[:200], name=f"Quantisiert (INT{bits})", line=dict(color="#EC4899", dash="dash")))
            fig.update_layout(
                template="plotly_dark", height=300,
                title=f"Quantisierungsfehler · {bits}-bit · SNR={snr:.1f} dB · MSE={mse:.5f}",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        c3.metric("SNR", f"{snr:.1f} dB", delta=f"+{snr - 6.0:.1f} dB vs. 1-bit baseline")
        c4.metric("MSE", f"{mse:.6f}")

        st.markdown(f"""
**{levels} Quantisierungslevels** bei **{bits} bit**:
- Speicher: `{32/bits:.1f}×` weniger als FP32
- SNR-Faustregel: ~6 dB pro Bit → bei {bits} bit: ~{bits*6} dB
        """)

        divider()
        lab_header("Modell-Größen-Rechner", "Parameter → Bytes → MB")
        col_a, col_b, col_c = st.columns(3)
        n_params = col_a.number_input("Parameter (Mio.)", min_value=0.1, max_value=1000.0, value=25.0, step=0.1)
        precision = col_b.selectbox("Präzision", ["FP32 (4 Byte)", "FP16 (2 Byte)", "INT8 (1 Byte)", "INT4 (0.5 Byte)"])
        bytes_per = {"FP32 (4 Byte)": 4, "FP16 (2 Byte)": 2, "INT8 (1 Byte)": 1, "INT4 (0.5 Byte)": 0.5}[precision]
        size_mb = n_params * 1e6 * bytes_per / 1e6
        col_c.metric("Modellgröße", f"{size_mb:.0f} MB", f"{size_mb/1024:.2f} GB" if size_mb > 1024 else None)

    # ── Tab 7: Videos ─────────────────────────────────────────────────────────
    with tabs[7]:
        section_header("Lernvideos")
        video_embed("DsGd2e9JNH4", "Quantization Explained — Confident Learning",
                    "Quantisierung und INT8 erklärt — warum 8 Bits oft genug sind")
        divider()
        video_embed("mBVjJ1ORQZY", "Knowledge Distillation — Yannic Kilcher",
                    "Hinton's Knowledge Distillation Paper erklärt")
        divider()
        video_embed("v_4KWmkwmsU", "TensorRT — NVIDIA Developer",
                    "TensorRT Workflow: ONNX Export → Optimierung → Deployment")
        divider()
        video_embed("W0HdmFCq8mc", "Pruning Deep Neural Networks",
                    "Iteratives Pruning, Sparsity und die Lottery Ticket Hypothesis")
