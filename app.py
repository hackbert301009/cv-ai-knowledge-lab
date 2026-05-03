"""
CV & AI Knowledge Lab — Master Edition
======================================
Streamlit-Hauptanwendung mit Sidebar-Navigation und modularer Architektur.
"""
import streamlit as st

# ----------------------------------------------------------------------
# Page-Config — MUSS als erstes Streamlit-Kommando aufgerufen werden
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="CV & AI Knowledge Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help":     "https://github.com/hackbert301009/cv-ai-knowledge-lab",
        "Report a bug": "https://github.com/hackbert301009/cv-ai-knowledge-lab/issues",
        "About":        "# CV & AI Knowledge Lab\nDeutsches Wissenshub für Computer Vision & KI."
    },
)

from src.components import inject_css
from src.registry import MODULES, CATEGORIES, get_module

# CSS einmalig injizieren
inject_css()

# ----------------------------------------------------------------------
# Modul-Loader Mapping (id → Render-Funktion)
# ----------------------------------------------------------------------
def _load_render(module_path: str, mod_id: str):
    """Lazy-Import um Startzeit zu reduzieren."""
    from importlib import import_module
    return import_module(module_path).render

# Mapping: jedes ID → Datei
MODULE_FILES = {
    "home":                "src.modules.home",
    "roadmap":             "src.modules.roadmap",
    "math":                "src.modules.math_crashcourse",
    "linalg":              "src.modules.linalg",
    "calculus":            "src.modules.calculus",
    "probability":         "src.modules.probability",
    "image_basics":        "src.modules.image_basics",
    "filters":             "src.modules.filters",
    "edges":               "src.modules.edges",
    "features":            "src.modules.features",
    "morphology":          "src.modules.morphology",
    "segmentation_classic":"src.modules.segmentation_classic",
    "nn_basics":           "src.modules.nn_basics",
    "cnn":                 "src.modules.cnn",
    "training":            "src.modules.training",
    "modern_archs":        "src.modules.modern_archs",
    "transformers":        "src.modules.transformers_mod",  # Datei umbenannt, um Konflikt mit HF-Lib zu vermeiden
    "vlm":                 "src.modules.vlm",
    "diffusion":           "src.modules.diffusion",
    "gen_ai":              "src.modules.gen_ai",
    "multimodal":          "src.modules.multimodal",
    "projects":            "src.modules.projects",
    "datasets":            "src.modules.datasets",
    "deployment":          "src.modules.deployment",
    "news":                "src.modules.news",
    "papers":              "src.modules.papers",
    "paper_of_month":      "src.modules.paper_of_month",
    "resources":           "src.modules.resources",
    "glossar":             "src.modules.glossar",
    "tensor_playground":   "src.modules.tensor_playground",
    "compression":         "src.modules.compression",
    "pose_estimation":     "src.modules.pose_estimation",
}

# ----------------------------------------------------------------------
# Sidebar — Navigation
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 0.5rem 0 1.5rem 0;">
          <div style="font-size: 2rem;">🧠</div>
          <div style="font-size: 1.05rem; font-weight: 700;
               background: linear-gradient(135deg, #7C3AED, #EC4899, #F59E0B);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;">
            CV &amp; AI Knowledge Lab
          </div>
          <div style="font-size: 0.7rem; color: #9CA3AF; letter-spacing: 0.1em;
               text-transform: uppercase; margin-top: 0.2rem;">
            Master Edition
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # State-basierte Navigation
    if "current_module" not in st.session_state:
        st.session_state.current_module = "home"

    # Suche
    search = st.text_input("🔍 Suche", placeholder="z.B. Diffusion, CNN, Sobel ...", label_visibility="collapsed")

    if search:
        results = [m for m in MODULES
                   if search.lower() in m.title.lower()
                   or search.lower() in m.short.lower()
                   or any(search.lower() in t for t in m.tags)]
        if results:
            st.caption(f"{len(results)} Treffer")
            for m in results:
                if st.button(f"{m.icon} {m.title}", key=f"search_{m.id}", use_container_width=True):
                    st.session_state.current_module = m.id
                    st.rerun()
        else:
            st.caption("Keine Treffer.")
        st.markdown("---")

    # Kategorien-Navigation
    for category, mod_ids in CATEGORIES.items():
        with st.expander(category, expanded=(category in ["🏠 Übersicht", "🧮 Grundlagen"])):
            for mod_id in mod_ids:
                m = get_module(mod_id)
                if m is None:
                    continue
                is_active = st.session_state.current_module == mod_id
                label = f"{m.icon} {m.title}"
                if st.button(
                    label,
                    key=f"nav_{mod_id}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state.current_module = mod_id
                    st.rerun()

    st.markdown("---")
    st.caption("🐙 [GitHub Repo](https://github.com/hackbert301009/cv-ai-knowledge-lab)")
    st.caption(f"📚 {len(MODULES)} Module · Made with Streamlit")

# ----------------------------------------------------------------------
# Main — aktuelles Modul rendern
# ----------------------------------------------------------------------
current_id = st.session_state.current_module
mod = get_module(current_id)

if mod is None or current_id not in MODULE_FILES:
    st.error(f"Modul **{current_id}** nicht gefunden.")
    st.stop()

try:
    render = _load_render(MODULE_FILES[current_id], current_id)
    render()
except Exception as exc:
    st.error(f"❌ Fehler beim Laden von Modul **{mod.title}**:\n\n```\n{exc}\n```")
    st.exception(exc)
