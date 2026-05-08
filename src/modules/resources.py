"""Ressourcen — Bücher, Kurse, Frameworks, Communities."""
import streamlit as st
from src.components import hero, section_header, divider, info_box, card, render_card_grid, render_learning_block


def render():
    hero(
        eyebrow="Live · Modul 25",
        title="Ressourcen &amp; Tools",
        sub="Kuratierte Sammlung der besten Bücher, Kurse, Frameworks, Newsletter "
            "und Communities — alles auf einer Seite."
    )

    section_header("📚 Bücher")
    cards = [
        card("📕", "Deep Learning", "Goodfellow, Bengio, Courville. Das Standardwerk. Kostenlos online verfügbar.", ["Klassiker"], ["amber"]),
        card("📘", "Dive into Deep Learning (d2l.ai)", "Aston Zhang et al. Open Source, mit Code in PyTorch/TF/MXNet. Hervorragend zum Lernen.", ["interaktiv"], ["green"]),
        card("📗", "Computer Vision: Algorithms and Applications", "Richard Szeliski. Das CV-Standardwerk. Kostenlos online.", ["CV-Bibel"], ["amber"]),
        card("📙", "Pattern Recognition and Machine Learning", "Christopher Bishop. Mathe-fundiert, klassisch.", ["theoretisch"], ["pink"]),
        card("📓", "Hands-On Machine Learning", "Aurélien Géron. Praxisnah, mit scikit-learn und TF.", ["praktisch"], ["green"]),
        card("📔", "Designing Machine Learning Systems", "Chip Huyen. Über Production ML — was die anderen Bücher nicht zeigen.", ["MLOps"], ["amber"]),
    ]
    render_card_grid(cards, cols=3)

    divider()

    section_header("🎓 Online-Kurse")
    cards = [
        card("🌟", "fast.ai — Practical Deep Learning", "Jeremy Howard. Top-down, sofort produktiv. Kostenlos.", ["Anfänger"], ["green"]),
        card("🎓", "Stanford CS231n", "Karpathy / Fei-Fei Li. Klassiker für CV mit Deep Learning. YouTube.", ["Standard"], ["amber"]),
        card("🤖", "Stanford CS224n", "NLP mit Deep Learning — wichtig auch für Multimodal-CV.", ["NLP/MM"], ["amber"]),
        card("🧠", "DeepLearning.AI Specializations", "Andrew Ng. Strukturiert, Coursera-Format.", ["umfassend"], ["green"]),
        card("🔧", "Hugging Face Courses", "Praktisch zu Transformers, Diffusion, RL. Kostenlos.", ["praktisch"], ["green"]),
        card("📐", "3Blue1Brown — Neural Networks", "Visuelle Intuition für Mathe und NN-Mechanismen.", ["Intuition"], ["blue"]),
    ]
    render_card_grid(cards, cols=3)

    divider()

    section_header("🛠️ Frameworks & Libraries")
    cards = [
        card("🔥", "PyTorch", "Der De-facto-Standard für Forschung. Flexibel, intuitiv.", ["Framework"], ["amber"]),
        card("🤗", "Transformers (HF)", "Pretrained-Modelle in 3 Zeilen. Auch für Vision.", ["Library"], ["green"]),
        card("📊", "timm", "Ross Wightman's PyTorch Image Models — alle SOTA-Modelle in einer Lib.", ["Vision"], ["green"]),
        card("🎯", "Ultralytics YOLO", "YOLO state-of-the-art. Detection/Segmentation/Pose in einer API.", ["Detection"], ["pink"]),
        card("🌊", "Diffusers (HF)", "Stable Diffusion, ControlNet, etc. Modular und gut dokumentiert.", ["Generativ"], ["pink"]),
        card("🔬", "OpenCV", "Klassische CV. Immer noch unverzichtbar.", ["CV-Klassik"], ["blue"]),
        card("⚡", "Lightning (PyTorch L.)", "Strukturiert Trainings-Code, scikit-artig.", ["Training"], ["amber"]),
        card("🧪", "Weights & Biases", "Experiment-Tracking. Der Industriestandard.", ["Tracking"], ["amber"]),
        card("🎨", "Roboflow", "Datasets annotieren, organisieren, augmentieren.", ["Daten"], ["blue"]),
    ]
    render_card_grid(cards, cols=3)

    divider()

    section_header("📡 Newsletter & Blogs")
    st.markdown("""
- **[Import AI](https://importai.substack.com/)** — Jack Clark (Anthropic). Wöchentlich, kuratiert, schnell.
- **[The Batch](https://www.deeplearning.ai/the-batch/)** — von DeepLearning.AI / Andrew Ng.
- **[The Algorithm](https://www.technologyreview.com/topic/artificial-intelligence/)** — MIT Tech Review.
- **[Last Week in AI](https://lastweekin.ai/)** — wöchentlich, breite Abdeckung.
- **[Lil'Log](https://lilianweng.github.io/)** — Lilian Weng's tiefe Posts. Pflichtlektüre für Diffusion, RL, Agents.
- **[Karpathy's Blog](https://karpathy.ai/)** — eigener Blog + neuronale Netz-Tutorials auf YouTube.
- **[Distill.pub](https://distill.pub/)** — visuelle Erklärungen (Archiv, nicht mehr aktiv, aber zeitlos).
- **[Hugging Face Blog](https://huggingface.co/blog)** — neue Modelle, Tutorials, Releases.
""")

    divider()

    section_header("👥 Communities")
    st.markdown("""
- **[r/MachineLearning](https://reddit.com/r/MachineLearning)** — Forschung, Diskussion
- **[r/computervision](https://reddit.com/r/computervision)** — CV-spezifisch
- **[r/StableDiffusion](https://reddit.com/r/StableDiffusion)** — Generative-Tipps und -Tricks
- **Hugging Face Discord** — sehr aktiv, Forscher und Engineers
- **[fast.ai Forums](https://forums.fast.ai/)** — gut für Anfängerfragen
- **Twitter/X** — Forscher posten oft direkt; folge @karpathy, @ylecun, @goodfellow_ian, @hardmaru, @_akhaliq
- **[Papers with Code Discord](https://discord.gg/paperswithcode)** — Discussions zu neuen Papers
- **[ML Collective](https://mlcollective.org/)** — Open Research-Community
""")

    divider()

    section_header("🎬 YouTube-Kanäle")
    st.markdown("""
- **3Blue1Brown** — Beste mathematische Visualisierungen
- **Two Minute Papers** — Schnelle Paper-Übersichten
- **Yannic Kilcher** — Tiefe Paper-Reviews
- **AI Coffee Break with Letitia** — verständlich, aktuell
- **Andrej Karpathy** — Neural Networks: Zero to Hero (Code-Along)
- **StatQuest** — Statistik und ML-Konzepte aus dem Boden heraus
""")

    info_box(
        "Mein bester Tipp: Such dir **eine** Hauptquelle (z.B. fast.ai oder CS231n) und arbeite sie durch. "
        "Der Versuch, alle 50 Ressourcen parallel zu konsumieren, endet meist in nichts. "
        "Tiefe schlägt Breite — wenigstens am Anfang.",
        kind="tip",
    )

    divider()
    render_learning_block(
        key_prefix="resources",
        progression=[
            ("🟢", "Guided Lab", "Eine Hauptressource wählen und 2-Wochen-Plan aufsetzen.", "Beginner", "green"),
            ("🟠", "Challenge Lab", "Theoriequelle + Praxisprojekt sinnvoll kombinieren.", "Intermediate", "amber"),
            ("🔴", "Debug Lab", "Lernstau erkennen und Ressourcenset fokussiert reduzieren.", "Advanced", "pink"),
            ("🏁", "Mini-Projekt", "Persönliches Learning-OS mit Weekly-Review bauen.", "Abschluss", "blue"),
        ],
        mcq_question="Was erhöht Lernerfolg typischerweise am stärksten?",
        mcq_options=[
            "Viele Quellen parallel",
            "Konsequentes Arbeiten mit einer Kernquelle + Praxis",
            "Nur Videos schauen",
            "Nur Paper lesen",
        ],
        mcq_correct_option="Konsequentes Arbeiten mit einer Kernquelle + Praxis",
        mcq_success_message="Richtig. Fokus + Anwendung schlägt reine Breite.",
        mcq_retry_message="Nicht optimal. Fokus ist entscheidend.",
        open_question="Offene Frage: Welche 3 Ressourcen setzt du in den nächsten 30 Tagen konkret ein?",
        code_task="""# Code-Aufgabe: Lernplan-Datenstruktur
learning_plan = {
    "week_1": [],
    "week_2": [],
    "week_3": [],
    "week_4": [],
}
# TODO: pro Woche Theorie, Praxis, Review ergänzen
""",
        community_rows=[
            {"Format": "Diskussion", "Fokus": "Welche Quelle hat dir zuletzt wirklich geholfen?", "Output": "Kurzbeitrag"},
            {"Format": "Peer-Feedback", "Fokus": "Ist dein Lernplan realistisch und ausgewogen?", "Output": "2 Stärken + 1 Verbesserung"},
            {"Format": "Challenge", "Fokus": "30-Tage-Lernstreak", "Output": "Weekly Check-in"},
        ],
        cheat_sheet=[
            "Eine Kernquelle + eine Praxisquelle reicht zum Start.",
            "Wöchentliches Review fest einplanen.",
            "Lernen sichtbar machen (Notizen, Demos, Repo).",
        ],
        key_takeaways=[
            "Konsistenz schlägt Intensität.",
            "Ressourcen wirken erst durch Umsetzung.",
        ],
        common_errors=[
            "Zu viele Quellen.",
            "Kein konkreter Wochenplan.",
            "Keine Wiederholung.",
            "Kein Praxisanteil.",
            "Fortschritt nicht dokumentiert.",
        ],
    )
