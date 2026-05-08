"""Deployment & MLOps."""
import streamlit as st
from src.components import hero, section_header, divider, info_box, step_list, render_learning_block


def render():
    hero(
        eyebrow="Praxis · Modul 22",
        title="Deployment &amp; MLOps",
        sub="Ein Modell, das nur in Jupyter läuft, hilft niemandem. Hier lernst du, wie du Modelle "
            "in Produktion bringst — schnell, robust und skalierbar."
    )

    tabs = st.tabs(["📦 Export", "⚡ Optimierung", "🌐 Serving", "🐳 Docker", "📊 Monitoring", "🧭 Lernpfad & Übungen"])

    with tabs[0]:
        section_header("Export: Vom Notebook zur deploybaren Datei")
        st.markdown(r"""
| Format | Wofür? | Stärken |
|---|---|---|
| **TorchScript** | PyTorch-Inferenz auf C++ | Direkt aus PyTorch, einfach |
| **ONNX** | Universalformat | Läuft auf vielen Runtimes (ONNX Runtime, TensorRT) |
| **TensorRT** | NVIDIA GPUs | Schnellste Inferenz auf NVIDIA |
| **Core ML** | Apple-Geräte | iPhone, iPad, Mac native |
| **TFLite** | Mobile / Edge | Android, Mikrocontroller |
| **OpenVINO** | Intel CPUs/GPUs | Sehr schnell auf Intel-Hardware |
        """)
        st.code("""
import torch

model.eval()
example_input = torch.randn(1, 3, 224, 224)

# 1. TorchScript
traced = torch.jit.trace(model, example_input)
traced.save("model.pt")

# 2. ONNX
torch.onnx.export(
    model, example_input, "model.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=17,
)
        """, language="python")

    with tabs[1]:
        section_header("Optimierung: schnell und schlank")
        st.markdown(r"""
#### Quantisierung
Statt 32-bit Floats benutze 8-bit Integers (oder INT4 für LLMs):
- **Post-Training Quantization (PTQ)**: Modell trainieren, dann quantisieren. Schnell, evtl. kleiner Genauigkeitsverlust.
- **Quantization-Aware Training (QAT)**: während des Trainings quantisieren. Höhere Genauigkeit.
- Speicher: ~4× weniger; Geschwindigkeit: 2–4× schneller auf passender Hardware.

#### Pruning
Entferne unwichtige Gewichte (oder ganze Channels).
- Strukturiertes Pruning (ganze Filter weg) ist hardwarefreundlich.
- Kann 50–90% der Parameter entfernen mit kleinem Genauigkeitsverlust.

#### Knowledge Distillation
Trainiere ein **kleines Student-Modell**, das die Outputs eines großen Teacher-Modells nachahmt.
- Beispiel: DistilBERT, TinyViT, MobileSAM.

#### Compilation
- **`torch.compile`** (PyTorch 2.x) — JIT-Optimiert, oft 1.5–2× schneller
- **TensorRT** — bestmögliche GPU-Performance auf NVIDIA
- **ONNX Runtime** mit Provider TensorRT/OpenVINO
        """)

    with tabs[2]:
        section_header("Serving: Modelle als API")
        st.markdown(r"""
#### Einfacher Stack: FastAPI
```python
from fastapi import FastAPI, UploadFile
import torch
from PIL import Image
from io import BytesIO

app = FastAPI()
model = torch.jit.load("model.pt")
model.eval()

@app.post("/predict")
async def predict(file: UploadFile):
    img = Image.open(BytesIO(await file.read())).convert("RGB")
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
    return {"class": int(out.argmax().item()),
            "probs": out.softmax(1).tolist()[0]}
```

Starten: `uvicorn app:app --host 0.0.0.0 --port 8000`

#### Production-grade Server
- **TorchServe** — Model Management, Batching, Versionierung
- **NVIDIA Triton** — Multi-Modell, dynamic batching, GPU-optimiert
- **BentoML** — modernes Framework, Modelle als "Bentos"
- **Ray Serve** — skalierbar, gut für komplexe Pipelines
- **vLLM / TGI** — speziell für LLMs (wenn relevant)

#### Edge-Deployment
- **OpenCV DNN Module** — direkt aus C++ oder Python
- **ONNX Runtime Mobile** — Android und iOS
- **CoreML** — iOS/macOS native
        """)

    with tabs[3]:
        section_header("Containerisierung mit Docker")
        st.code("""
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
        """, language="dockerfile")
        st.markdown(r"""
Bauen und starten:
```bash
docker build -t my-cv-api .
docker run -p 8000:8000 my-cv-api
```

Mit GPU-Support (NVIDIA):
```bash
docker run --gpus all -p 8000:8000 my-cv-api
```

#### Best Practices
- Verwende Multi-Stage-Builds (kleinere Images)
- Nutze offizielle PyTorch/CUDA-Images für GPU-Workloads
- `.dockerignore` setzen, sonst landet alles im Image
- Health Check Endpoint bauen (`/health`)
- Logging strukturiert (JSON), nicht print
        """)

    with tabs[4]:
        section_header("Monitoring: was passiert nach dem Deploy?")
        st.markdown(r"""
#### Was du tracken musst
- **Latenz** (p50, p95, p99) — wie schnell antwortet das Modell?
- **Throughput** — Requests pro Sekunde
- **Fehlerrate** — was geht schief?
- **Eingabeverteilung** — verschiebt sich die Datenverteilung über Zeit? (**Data Drift**)
- **Outputverteilung** — vorhersagen sich plötzlich anders? (**Concept Drift**)
- **Modell-Konfidenz** — fallen die Konfidenzen über Zeit?

#### Tools
- **Prometheus + Grafana** — klassische Metriken-Stack
- **Weights & Biases** — Modell- und Experiment-Tracking
- **MLflow** — Open-Source-Alternative zu W&B
- **Evidently AI** — Drift-Detection out of the box
- **Arize AI, Fiddler** — kommerzielle Observability für ML

#### Retraining-Strategie
1. Drift erkennen
2. Falsch klassifizierte Beispiele sammeln
3. Manuell labeln (oder mit Active Learning automatisieren)
4. Modell retrainen
5. A/B Test gegen altes Modell
6. Wenn besser: deployen
        """)
        info_box(
            "Modelle 'verfaulen' in Produktion. Die Welt ändert sich, deine Trainings-Daten nicht. "
            "Plane Retraining-Zyklen schon beim ersten Deployment ein.",
            kind="warn",
        )

    with tabs[5]:
        render_learning_block(
            key_prefix="deployment",
            section_title="Lernpfad für Deployment",
            progression=[
                ("🟢", "Guided Lab", "Ein Modell nach ONNX exportieren und lokal serven.", "Beginner", "green"),
                ("🟠", "Challenge Lab", "Latenz unter Zielwert bringen (Quantisierung/Scheduler/Batching).", "Intermediate", "amber"),
                ("🔴", "Debug Lab", "API-Fehler, Shape-Mismatch und Inferenz-Timeouts beheben.", "Advanced", "pink"),
                ("🏁", "Mini-Projekt", "Deployment-Pipeline mit Monitoring und Health-Checks.", "Abschluss", "blue"),
            ],
            mcq_question="Welche Metrik zeigt am besten Tail-Latenz in Produktion?",
            mcq_options=["p50", "p95/p99", "Mean Accuracy", "GPU Temperature only"],
            mcq_correct_option="p95/p99",
            mcq_success_message="Richtig. Tail-Latenz entscheidet oft über echte Nutzererfahrung.",
            mcq_retry_message="Noch nicht korrekt. Monitoring-Sektion prüfen.",
            open_question="Offene Frage: Welche Deploy-Risiken würdest du vor Go-Live aktiv absichern?",
            code_task="""# Code-Aufgabe: einfacher Health-Endpoint in FastAPI
from fastapi import FastAPI

app = FastAPI()

# TODO: /health Endpoint ergänzen, der Status und Modellversion zurückgibt
""",
            community_rows=[
                {"Format": "Diskussion", "Fokus": "Welcher Bottleneck war bei dir am größten?", "Output": "Kurzbeitrag"},
                {"Format": "Peer-Feedback", "Fokus": "Ist das Deployment reproduzierbar dokumentiert?", "Output": "2 Pluspunkte + 1 Risiko"},
                {"Format": "Challenge", "Fokus": "Niedrigste p95 bei gleicher Qualität", "Output": "Benchmarks"},
            ],
            cheat_sheet=[
                "Export-Format früh festlegen.",
                "Health, logging, tracing von Anfang an einbauen.",
                "p95/p99 und Fehlerquote kontinuierlich überwachen.",
            ],
            key_takeaways=[
                "Deployment ist ein Produktproblem, nicht nur ein Modellproblem.",
                "Observability ist Pflicht, sonst werden Fehler zu spät sichtbar.",
            ],
            common_errors=[
                "Kein Health-Endpoint.",
                "Keine Lasttests vor Go-Live.",
                "Model/Input-Versionen nicht versioniert.",
                "Nur Mean-Latenz statt p95/p99.",
                "Kein Retraining-Plan.",
            ],
        )
