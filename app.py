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
    "camera_pipeline":     "src.modules.camera_pipeline",
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
    "learning_studio":     "src.modules.learning_studio",
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

TRACKABLE_MODULES = [m.id for m in MODULES if m.id != "home"]


def _init_hub_state():
    """Session-State für Personal-Hub initialisieren."""
    st.session_state.setdefault("current_module", "home")
    st.session_state.setdefault("visited_modules", [])
    st.session_state.setdefault("favorite_modules", [])
    st.session_state.setdefault("completed_modules", [])
    st.session_state.setdefault("last_module", "home")


def _mark_module_visit(mod_id: str):
    """Zuletzt besuchte Module in der Session pflegen."""
    if mod_id == "home":
        return
    visited = list(st.session_state.get("visited_modules", []))
    if mod_id in visited:
        visited.remove(mod_id)
    visited.insert(0, mod_id)
    st.session_state.visited_modules = visited[:8]
    st.session_state.last_module = mod_id


def _toggle_list_item(session_key: str, item: str):
    """Item in einer Session-Liste umschalten (add/remove)."""
    items = list(st.session_state.get(session_key, []))
    if item in items:
        items.remove(item)
    else:
        items.append(item)
    st.session_state[session_key] = items


def _completion_ratio() -> float:
    completed = set(st.session_state.get("completed_modules", []))
    total = max(1, len(TRACKABLE_MODULES))
    return len(completed.intersection(TRACKABLE_MODULES)) / total


def _render_personal_hub():
    """Sidebar-Block mit personalisiertem Lernstatus und Shortcuts."""
    st.markdown("### 🧭 Personal Hub")

    progress = _completion_ratio()
    completed_count = len(set(st.session_state.get("completed_modules", [])).intersection(TRACKABLE_MODULES))
    st.progress(progress)
    st.caption(f"Fortschritt: {completed_count}/{len(TRACKABLE_MODULES)} Module abgeschlossen")

    last_mod_id = st.session_state.get("last_module", "home")
    last_mod = get_module(last_mod_id)
    if last_mod is not None and last_mod.id != "home":
        if st.button(f"▶️ Weiter mit: {last_mod.title}", key="hub_resume", use_container_width=True):
            st.session_state.current_module = last_mod.id
            st.rerun()

    favorites = st.session_state.get("favorite_modules", [])
    if favorites:
        st.caption("⭐ Favoriten")
        for fav_id in favorites[:5]:
            mod = get_module(fav_id)
            if mod is None:
                continue
            if st.button(f"{mod.icon} {mod.title}", key=f"hub_fav_{fav_id}", use_container_width=True):
                st.session_state.current_module = fav_id
                st.rerun()

    recent = st.session_state.get("visited_modules", [])
    if recent:
        st.caption("🕘 Zuletzt besucht")
        for rec_id in recent[:5]:
            mod = get_module(rec_id)
            if mod is None:
                continue
            if st.button(f"{mod.icon} {mod.title}", key=f"hub_recent_{rec_id}", use_container_width=True):
                st.session_state.current_module = rec_id
                st.rerun()

    if st.button("🔄 Hub-Session zurücksetzen", key="hub_reset", use_container_width=True):
        st.session_state.visited_modules = []
        st.session_state.favorite_modules = []
        st.session_state.completed_modules = []
        st.session_state.last_module = "home"
        st.rerun()


_init_hub_state()

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
    _render_personal_hub()

    current_mod = get_module(st.session_state.current_module)
    if current_mod is not None and current_mod.id != "home":
        is_favorite = current_mod.id in st.session_state.favorite_modules
        is_completed = current_mod.id in st.session_state.completed_modules
        c1, c2 = st.columns(2)
        with c1:
            if st.button(
                "⭐ Unfave" if is_favorite else "⭐ Favorit",
                key=f"active_fav_{current_mod.id}",
                use_container_width=True,
            ):
                _toggle_list_item("favorite_modules", current_mod.id)
                st.rerun()
        with c2:
            if st.button(
                "✅ Offen" if is_completed else "✅ Fertig",
                key=f"active_done_{current_mod.id}",
                use_container_width=True,
            ):
                _toggle_list_item("completed_modules", current_mod.id)
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
    _mark_module_visit(current_id)
    render = _load_render(MODULE_FILES[current_id], current_id)
    render()
except Exception as exc:
    st.error(f"❌ Fehler beim Laden von Modul **{mod.title}**:\n\n```\n{exc}\n```")
    st.exception(exc)
