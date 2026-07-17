"""Lernstudio — strukturierte Labs, Übungen und Community-Formate."""
import streamlit as st

from src.components import card, divider, hero, info_box, render_card_grid, section_header


def _render_lab_tracks():
    section_header(
        "Labore mit klarer Progression",
        "Vom geführten Einstieg bis zur realen Challenge in drei Stufen.",
    )
    cards = [
        card(
            "🟢",
            "Beginner Labs",
            "Geführte Basics mit Schritt-für-Schritt Hinweisen, Checkpoints und sofortigem Feedback.",
            ["Geführt", "30-45 min"],
            ["green", "blue"],
        ),
        card(
            "🟠",
            "Intermediate Labs",
            "Weniger Anleitung, mehr Transfer. Fokus auf Parametertuning, Datenqualität und Evaluation.",
            ["Angeleitet", "45-75 min"],
            ["amber", "blue"],
        ),
        card(
            "🔴",
            "Advanced Labs",
            "Reale Problemstellungen mit unvollständiger Spezifikation und begrenzten Ressourcen.",
            ["Challenge", "75-120 min"],
            ["pink", "blue"],
        ),
    ]
    render_card_grid(cards, cols=3)

    st.markdown("### Lab-Typen pro Thema")
    st.markdown(
        """
        - **Geführte Labs:** klarer Ablauf, Hilfekarten, Lösungswege in Etappen.
        - **Challenge Labs:** kaum Hilfen, Fokus auf eigenständiges Problemlösen.
        - **Debug Labs:** absichtlich fehlerhafte Pipelines, die systematisch repariert werden.
        - **Mini-Projekte:** Modulabschluss mit kurzer Projektbeschreibung und Abnahmekriterien.
        """
    )
    st.graphviz_chart(
        """
        digraph G {
            rankdir=LR;
            node [shape=box, style=rounded];
            A [label="Theorie-Kern"];
            B [label="Geführtes Lab"];
            C [label="Challenge Lab"];
            D [label="Debug Lab"];
            E [label="Mini-Projekt"];
            A -> B -> C -> D -> E;
        }
        """
    )


def _render_exercises():
    section_header(
        "Mischformat-Übungen",
        "Multiple Choice, offene Reflexion und Code-Aufgabe in einer kompakten Einheit.",
    )
    with st.container(border=True):
        st.markdown("#### 1) Multiple Choice")
        answer = st.radio(
            "Welche Metrik ist bei unausgeglichenen Klassen oft aussagekräftiger als Accuracy?",
            ["Accuracy", "F1-Score", "Top-1 Error", "Mean Pixel Value"],
            key="mcq_metric",
        )
        if st.button("Antwort prüfen", key="check_mcq"):
            if answer == "F1-Score":
                st.success("Richtig. Der F1-Score balanciert Precision und Recall.")
            else:
                st.warning("Noch nicht ganz. Schau dir Precision/Recall und F1 an.")

        st.markdown("#### 2) Offene Frage")
        st.text_area(
            "Beschreibe in 3-5 Sätzen, warum Datenaugmentation bei kleinen Datensätzen hilft.",
            key="open_reflection",
            placeholder="Formuliere den Effekt auf Generalisierung und Overfitting...",
            height=120,
        )

        st.markdown("#### 3) Code-Aufgabe")
        st.code(
            """# TODO: val_accuracy aus logits und labels berechnen
import torch

def compute_val_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    # your code here
    pass
""",
            language="python",
        )
        show_solution = st.checkbox("Beispiel-Lösung anzeigen", key="show_code_solution")
        if show_solution:
            st.code(
                """import torch

def compute_val_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == labels).float().mean()
    return float(correct.item())
""",
                language="python",
            )


def _render_community():
    section_header(
        "Community-Elemente",
        "Diskussionen, Peer-Feedback und gemeinsame Challenges für konstantes Lernen.",
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Diskussionen")
        st.markdown(
            """
            - Wöchentliche Leitfrage pro Modul.
            - "Was war für dich der Knackpunkt?"-Threads.
            - Moderierte Zusammenfassung der besten Antworten.
            """
        )
        st.markdown("#### Peer-Feedback")
        st.markdown(
            """
            - Klare Rubrik: Korrektheit, Lesbarkeit, Effizienz, Erklärung.
            - 2 Pluspunkte + 1 Verbesserungsvorschlag als Standard.
            - Feedback innerhalb von 48 Stunden als Challenge-Regel.
            """
        )
    with c2:
        st.markdown("#### Gemeinsame Challenges")
        st.markdown(
            """
            - Monats-Challenge mit gemeinsamem Datensatz.
            - Team-Option (2-3 Personen) für Pair-Learning.
            - Demo-Day: kurze Präsentation + Lessons Learned.
            """
        )
        st.markdown("#### Vergleichstabelle")
        st.table(
            [
                {"Format": "Diskussion", "Ziel": "Verständnis", "Dauer": "10-20 min"},
                {"Format": "Peer-Feedback", "Ziel": "Qualität", "Dauer": "15-30 min"},
                {"Format": "Challenge", "Ziel": "Transfer", "Dauer": "1-5 Tage"},
            ]
        )


def _render_cheat_sheet():
    section_header("Cheat Sheet, Key Takeaways und häufige Fehler")
    tab1, tab2, tab3 = st.tabs(["Cheat Sheet", "Key Takeaways", "Häufige Fehler"])
    with tab1:
        st.markdown(
            """
            - **Train/Val/Test sauber trennen** vor jeder Modellierung.
            - **Baseline zuerst** (einfaches Modell, klare Metrik).
            - **Ablation dokumentieren** (was verbessert wirklich?).
            - **Reproduzierbarkeit**: Seeds, Versionen, Config sichern.
            """
        )
    with tab2:
        st.markdown(
            """
            - Jede Einheit endet mit einem konkreten Artefakt (Notebook, Report oder Demo).
            - Lernfortschritt steigt stark durch Wechsel aus Lesen, Anwenden und Erklären.
            - Debug-Labs trainieren dieselben Fähigkeiten, die im echten Projekt den Unterschied machen.
            """
        )
    with tab3:
        st.markdown(
            """
            1. Zu früh komplexe Modelle statt erst Baseline.
            2. Datenleck durch falsches Splitting.
            3. Metrik passt nicht zur eigentlichen Aufgabe.
            4. Ergebnisse ohne Fehlanalyse interpretieren.
            5. Hyperparameter ändern ohne sauberes Logging.
            """
        )
        info_box(
            "Diese Fehlerliste ist als Review-Checkliste gedacht. Nutze sie vor jeder Abgabe.",
            kind="warn",
        )


def render():
    hero(
        eyebrow="Praxis · Lernstudio",
        title="Lernstudio: Labs, Übungen & Community",
        sub=(
            "Hier baust du praxisnahes Lernen strukturiert auf: klare Lab-Progression, "
            "Mischformat-Übungen, kollaborative Challenges und saubere Lernzusammenfassungen."
        ),
    )

    _render_lab_tracks()
    divider()
    _render_exercises()
    divider()
    _render_community()
    divider()
    _render_cheat_sheet()
