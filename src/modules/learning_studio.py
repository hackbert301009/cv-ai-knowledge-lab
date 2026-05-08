"""Lernstudio — strukturierte Labs, Uebungen und Community-Formate."""
import streamlit as st

from src.components import card, divider, hero, info_box, render_card_grid, section_header, render_learning_block


def _render_lab_tracks():
    section_header(
        "Labore mit klarer Progression",
        "Vom gefuehrten Einstieg bis zur realen Challenge in drei Stufen.",
    )
    cards = [
        card(
            "🟢",
            "Beginner Labs",
            "Gefuehrte Basics mit Schritt-fuer-Schritt Hinweisen, Checkpoints und sofortigem Feedback.",
            ["Gefuehrt", "30-45 min"],
            ["green", "blue"],
        ),
        card(
            "🟠",
            "Intermediate Labs",
            "Weniger Anleitung, mehr Transfer. Fokus auf Parametertuning, Datenqualitaet und Evaluation.",
            ["Angeleitet", "45-75 min"],
            ["amber", "blue"],
        ),
        card(
            "🔴",
            "Advanced Labs",
            "Reale Problemstellungen mit unvollstaendiger Spezifikation und begrenzten Ressourcen.",
            ["Challenge", "75-120 min"],
            ["pink", "blue"],
        ),
    ]
    render_card_grid(cards, cols=3)

    st.markdown("### Lab-Typen pro Thema")
    st.markdown(
        """
        - **Gefuehrte Labs:** klarer Ablauf, Hilfekarten, Loesungswege in Etappen.
        - **Challenge Labs:** kaum Hilfen, Fokus auf eigenstaendiges Problemloesen.
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
            B [label="Gefuehrtes Lab"];
            C [label="Challenge Lab"];
            D [label="Debug Lab"];
            E [label="Mini-Projekt"];
            A -> B -> C -> D -> E;
        }
        """
    )


def _render_exercises():
    section_header(
        "Mischformat-Uebungen",
        "Multiple Choice, offene Reflexion und Code-Aufgabe in einer kompakten Einheit.",
    )
    with st.container(border=True):
        st.markdown("#### 1) Multiple Choice")
        answer = st.radio(
            "Welche Metrik ist bei unausgeglichenen Klassen oft aussagekraeftiger als Accuracy?",
            ["Accuracy", "F1-Score", "Top-1 Error", "Mean Pixel Value"],
            key="mcq_metric",
        )
        if st.button("Antwort pruefen", key="check_mcq"):
            if answer == "F1-Score":
                st.success("Richtig. Der F1-Score balanciert Precision und Recall.")
            else:
                st.warning("Noch nicht ganz. Schau dir Precision/Recall und F1 an.")

        st.markdown("#### 2) Offene Frage")
        st.text_area(
            "Beschreibe in 3-5 Saetzen, warum Datenaugmentation bei kleinen Datensaetzen hilft.",
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
        show_solution = st.checkbox("Beispiel-Loesung anzeigen", key="show_code_solution")
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
        "Diskussionen, Peer-Feedback und gemeinsame Challenges fuer konstantes Lernen.",
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Diskussionen")
        st.markdown(
            """
            - Woechentliche Leitfrage pro Modul.
            - "Was war fuer dich der Knackpunkt?"-Threads.
            - Moderierte Zusammenfassung der besten Antworten.
            """
        )
        st.markdown("#### Peer-Feedback")
        st.markdown(
            """
            - Klare Rubrik: Korrektheit, Lesbarkeit, Effizienz, Erklaerung.
            - 2 Pluspunkte + 1 Verbesserungsvorschlag als Standard.
            - Feedback innerhalb von 48 Stunden als Challenge-Regel.
            """
        )
    with c2:
        st.markdown("#### Gemeinsame Challenges")
        st.markdown(
            """
            - Monats-Challenge mit gemeinsamem Datensatz.
            - Team-Option (2-3 Personen) fuer Pair-Learning.
            - Demo-Day: kurze Praesentation + Lessons Learned.
            """
        )
        st.markdown("#### Vergleichstabelle")
        st.table(
            [
                {"Format": "Diskussion", "Ziel": "Verstaendnis", "Dauer": "10-20 min"},
                {"Format": "Peer-Feedback", "Ziel": "Qualitaet", "Dauer": "15-30 min"},
                {"Format": "Challenge", "Ziel": "Transfer", "Dauer": "1-5 Tage"},
            ]
        )


def _render_cheat_sheet():
    section_header("Cheat Sheet, Key Takeaways und haeufige Fehler")
    tab1, tab2, tab3 = st.tabs(["Cheat Sheet", "Key Takeaways", "Hauefige Fehler"])
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
            - Lernfortschritt steigt stark durch Wechsel aus Lesen, Anwenden und Erklaeren.
            - Debug-Labs trainieren dieselben Faehigkeiten, die im echten Projekt den Unterschied machen.
            """
        )
    with tab3:
        st.markdown(
            """
            1. Zu frueh komplexe Modelle statt erst Baseline.
            2. Datenleck durch falsches Splitting.
            3. Metrik passt nicht zur eigentlichen Aufgabe.
            4. Ergebnisse ohne Fehlanalyse interpretieren.
            5. Hyperparameter aendern ohne sauberes Logging.
            """
        )
        info_box(
            "Diese Fehlerliste ist als Review-Checkliste gedacht. Nutze sie vor jeder Abgabe.",
            kind="warn",
        )


def render():
    hero(
        eyebrow="Lernsystem · Neues Modul",
        title="Lernstudio: Labs, Uebungen & Community",
        sub=(
            "Hier baust du praxisnahes Lernen strukturiert auf: klare Lab-Progession, "
            "Mischformat-Uebungen, kollaborative Challenges und saubere Lernzusammenfassungen."
        ),
    )

    _render_lab_tracks()
    divider()
    render_learning_block(
        key_prefix="learning_studio",
        section_title="Mischformat, Community und Lernzusammenfassung",
        section_sub="Einheitlicher Lernblock für alle Module.",
        mcq_question="Welche Metrik ist bei unausgeglichenen Klassen oft aussagekraeftiger als Accuracy?",
        mcq_options=["Accuracy", "F1-Score", "Top-1 Error", "Mean Pixel Value"],
        mcq_correct_option="F1-Score",
        mcq_success_message="Richtig. Der F1-Score balanciert Precision und Recall.",
        mcq_retry_message="Noch nicht ganz. Schau dir Precision/Recall und F1 an.",
        open_question="Beschreibe in 3-5 Saetzen, warum Datenaugmentation bei kleinen Datensaetzen hilft.",
        code_task="""# TODO: val_accuracy aus logits und labels berechnen
import torch

def compute_val_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == labels).float().mean()
    return float(correct.item())
""",
        community_rows=[
            {"Format": "Diskussion", "Ziel": "Verstaendnis", "Dauer": "10-20 min"},
            {"Format": "Peer-Feedback", "Ziel": "Qualitaet", "Dauer": "15-30 min"},
            {"Format": "Challenge", "Ziel": "Transfer", "Dauer": "1-5 Tage"},
        ],
        cheat_sheet=[
            "Train/Val/Test sauber trennen vor jeder Modellierung.",
            "Baseline zuerst (einfaches Modell, klare Metrik).",
            "Ablation dokumentieren (was verbessert wirklich?).",
            "Reproduzierbarkeit: Seeds, Versionen, Config sichern.",
        ],
        key_takeaways=[
            "Jede Einheit endet mit einem konkreten Artefakt (Notebook, Report oder Demo).",
            "Lernfortschritt steigt stark durch Wechsel aus Lesen, Anwenden und Erklaeren.",
            "Debug-Labs trainieren dieselben Faehigkeiten, die im echten Projekt den Unterschied machen.",
        ],
        common_errors=[
            "Zu frueh komplexe Modelle statt erst Baseline.",
            "Datenleck durch falsches Splitting.",
            "Metrik passt nicht zur eigentlichen Aufgabe.",
            "Ergebnisse ohne Fehlanalyse interpretieren.",
            "Hyperparameter aendern ohne sauberes Logging.",
        ],
    )
    info_box(
        "Diese Fehlerliste ist als Review-Checkliste gedacht. Nutze sie vor jeder Abgabe.",
        kind="warn",
    )
