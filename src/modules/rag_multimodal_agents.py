"""RAG und multimodale Agents fuer Vision + Sprache."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.components import (
    divider,
    hero,
    info_box,
    key_concept,
    lab_header,
    section_header,
    step_list,
    render_quiz_checkpoint,
    video_embed,
)


def _retrieval_quality(top_k: int, chunk_size: int, rerank: bool) -> float:
    base = 0.52 + 0.18 * np.tanh((top_k - 3) / 6)
    chunk_effect = 0.10 * np.exp(-((chunk_size - 420) ** 2) / (2 * (220 ** 2)))
    rerank_bonus = 0.08 if rerank else 0.0
    return float(np.clip(base + chunk_effect + rerank_bonus, 0.0, 0.97))


def render():
    hero(
        eyebrow="State-of-the-Art · Agents",
        title="RAG + Multimodal Agents",
        sub="Vision-RAG Pipelines, VLM Tool-Use, Prompting-Patterns und Guardrails in der Praxis.",
    )

    tabs = st.tabs(
        [
            "🔎 RAG Grundlagen",
            "🧰 Tool-Use",
            "🛡 Guardrails",
            "🧪 Pipeline Lab",
            "💻 Code",
            "📋 Patterns",
            "✅ Checkpoint",
            "🎬 Videos",
        ]
    )

    with tabs[0]:
        section_header("Wie Vision-RAG funktioniert")
        step_list(
            [
                ("Ingest", "Dokumente, Bilder, OCR-Text und Metadaten extrahieren."),
                ("Index", "Embeddings in Vector Store speichern."),
                ("Retrieve", "Top-k relevante Chunks fuer Anfrage abrufen."),
                ("Grounded Generation", "VLM/LLM antwortet mit Quellenbezug."),
            ]
        )
        key_concept("🧩", "Chunking", "Gute Chunk-Groesse erhoeht Recall und reduziert Halluzinationen.")
        key_concept("🖼️", "Multimodal Retrieval", "Suche ueber Text + Bildfeatures statt nur Text.")

    with tabs[1]:
        section_header("VLM Tool-Use")
        st.markdown(
            """
Ein Agent kombiniert Modellantworten mit externen Tools:
- OCR fuer Text in Bildern
- Detection/Segmentation APIs
- Tabellenabfragen, Knowledge Base, Search
- Deterministische Checker fuer kritische Aussagen
            """
        )
        info_box("Toolformer-Verhalten braucht strikte Tool-Schemas und Timeouts.", kind="tip")

    with tabs[2]:
        section_header("Guardrails und Sicherheit")
        st.markdown(
            """
| Layer | Beispiel |
|---|---|
| Input Guardrails | Prompt Injection Filter, Dateityp-Checks |
| Retrieval Guardrails | Source Whitelisting, Sensitivity Labels |
| Output Guardrails | Citation Pflicht, Policy/PII Filter |
| Runtime Guardrails | Tool Allowlist, Budget Limits, Audit Logs |
            """
        )
        key_concept("🧾", "Citations", "Antworten sollten auf konkrete Quellen-IDs verweisen.")
        key_concept("⛔", "Tool Allowlist", "Agent darf nur explizit freigegebene Aktionen ausfuehren.")

    with tabs[3]:
        lab_header("RAG Tuning", "Top-k, Chunking und Reranking beeinflussen Antwortqualitaet.")
        top_k = st.slider("Top-k Retrieval", 1, 25, 8, 1)
        chunk = st.slider("Chunk Size (tokens)", 120, 1200, 420, 20)
        rerank = st.checkbox("Reranking aktivieren", value=True)
        quality = _retrieval_quality(top_k, chunk, rerank)
        latency = 180 + top_k * 24 + (65 if rerank else 0)
        halluc = max(0.02, 0.32 - quality * 0.27)
        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated Answer Quality", f"{quality * 100:.1f}%")
        c2.metric("Estimated Latency", f"{latency} ms")
        c3.metric("Estimated Hallucination Risk", f"{halluc * 100:.1f}%")

        x = np.arange(1, 26)
        y = [_retrieval_quality(int(k), chunk, rerank) for k in x]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", line=dict(color="#10B981", width=3)))
        fig.update_layout(
            template="plotly_dark",
            height=360,
            xaxis_title="Top-k",
            yaxis_title="Estimated quality",
            yaxis=dict(range=[0.45, 1.0]),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        section_header("Praxis-Code")
        st.code(
            """# Minimaler multimodaler RAG Ablauf (Pseudo)
query = user_input()
query_vec = embedder.encode(query)
docs = vector_db.search(query_vec, top_k=8)
reranked = reranker.rank(query, docs)[:5]

answer = vlm.generate(
    prompt=build_prompt(query, reranked),
    images=[d.image for d in reranked if d.image is not None],
)
print(answer)
            """,
            language="python",
        )
        divider()
        st.code(
            """# Tool schema + allowlist (Pseudo)
TOOLS = {
    "ocr": ocr_tool,
    "detect_objects": detect_tool,
    "kb_search": kb_tool,
}

def safe_call(tool_name, args):
    if tool_name not in TOOLS:
        raise ValueError("Tool not allowed")
    return TOOLS[tool_name](**args)
            """,
            language="python",
        )

    with tabs[5]:
        section_header("Prompting Patterns")
        st.markdown(
            """
- **Ground-then-answer:** Erst Quellen ausgeben, dann Antwort.
- **Tool-first for uncertainty:** Bei unklarer Evidenz zuerst Retrieval/Tool aufrufen.
- **Structured output:** JSON Schema fuer downstream Verarbeitung.
- **Self-check:** Modell validiert eigene Antwort gegen Quellenliste.
            """
        )

    with tabs[6]:
        render_quiz_checkpoint(
            key_prefix="rag_multimodal_agents",
            question="Welcher Guardrail gehoert zur Runtime-Ebene?",
            options=[
                "Tool Allowlist",
                "Chunking",
                "Temperature Scaling",
                "Data Augmentation",
            ],
            correct_option="Tool Allowlist",
            checklist=[
                "Ich verstehe die Schritte Ingest -> Index -> Retrieve -> Generate.",
                "Ich weiss, wann Reranking sinnvoll ist.",
                "Ich kann mindestens drei Guardrail-Layer benennen.",
            ],
            capstone_prompt="Skizziere einen Vision-RAG-Agenten fuer technische Dokumente "
            "inklusive Tool-Schema, Citation-Policy und Fehlerbehandlung.",
        )

    with tabs[7]:
        section_header("Lernvideos")
        video_embed("T-D1OfcDW1M", "RAG Basics", "Retriever + Generator sauber aufbauen.")
        divider()
        video_embed("a8QvnIAGjPA", "Multimodal RAG", "Text und Bildquellen gemeinsam nutzen.")
        divider()
        video_embed("4nUjYp8fN0Q", "Agent Guardrails", "Sicherheitsmuster fuer produktive Agents.")
