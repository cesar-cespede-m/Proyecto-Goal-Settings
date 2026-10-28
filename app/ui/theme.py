"""CSS generado desde tokens; no contiene estado ni reglas de dominio."""

import streamlit as st

from app.ui.accessibility import validate_contrast_pairs
from app.ui.tokens import (
    BORDER_WIDTH,
    CONTRAST_PAIRS,
    FONT_FAMILY,
    LAYOUT,
    LINE_HEIGHT,
    RADIUS,
    SIZES,
    color,
    spacing,
    type_size,
)


def build_theme_css() -> str:
    """Construye el tema sólo después de validar sus contrastes semánticos."""

    validate_contrast_pairs(CONTRAST_PAIRS)
    return f"""
    <style>
      :root {{
        --gsa-color-bg: {color("background")};
        --gsa-color-surface: {color("surface")};
        --gsa-color-surface-subtle: {color("surface-subtle")};
        --gsa-color-surface-muted: {color("surface-muted")};
        --gsa-color-border: {color("border")};
        --gsa-color-text: {color("text")};
        --gsa-color-text-muted: {color("text-muted")};
        --gsa-color-primary: {color("primary")};
        --gsa-color-primary-hover: {color("primary-hover")};
        --gsa-color-on-primary: {color("on-primary")};
        --gsa-space-inline-tight: {spacing("inline-tight")};
        --gsa-space-control-gap: {spacing("control-gap")};
        --gsa-space-component: {spacing("component-inset")};
        --gsa-space-card: {spacing("card-inset")};
        --gsa-space-section: {spacing("section-gap")};
        --gsa-space-label: {spacing("label-gap")};
        --gsa-space-panel: {spacing("panel-inset")};
        --gsa-space-stage: {spacing("stage-gap")};
        --gsa-space-page-inline: {spacing("page-inline")};
        --gsa-space-page-bottom: {spacing("page-bottom")};
        --gsa-space-page-bottom-compact: {spacing("page-bottom-compact")};
        --gsa-space-page-bottom-tablet: {spacing("page-bottom-tablet")};
        --gsa-radius-sm: {RADIUS["sm"]};
        --gsa-radius-md: {RADIUS["md"]};
        --gsa-radius-lg: {RADIUS["lg"]};
        --gsa-font-xs: {type_size("caption")};
        --gsa-font-sm: {type_size("supporting")};
        --gsa-font-base: {type_size("body")};
        --gsa-color-accent: {color("accent")};
        --gsa-color-accent-container: {color("accent-container")};
        --gsa-color-on-accent-container: {color("on-accent-container")};
        --gsa-color-primary-container: {color("primary-container")};
        --gsa-color-on-primary-container: {color("on-primary-container")};
        --gsa-color-selected: {color("surface-selected")};
        --gsa-color-on-selected: {color("on-surface-selected")};
        --gsa-color-border-selected: {color("border-selected")};
        --gsa-color-disabled: {color("surface-disabled")};
        --gsa-color-on-disabled: {color("on-surface-disabled")};
        --gsa-color-border-disabled: {color("border-disabled")};
        --gsa-color-success: {color("success")};
        --gsa-color-success-container: {color("success-container")};
        --gsa-color-danger: {color("danger")};
        --gsa-color-danger-container: {color("danger-container")};
        --gsa-border-thin: {BORDER_WIDTH["thin"]};
        --gsa-border-emphasis: {BORDER_WIDTH["emphasis"]};
        --gsa-border-accent: {BORDER_WIDTH["accent"]};
        --gsa-radius-pill: {RADIUS["pill"]};
        --gsa-touch-target: {SIZES["touch-target"]};
        --gsa-step-badge: {SIZES["step-badge"]};
        --gsa-line-tight: {LINE_HEIGHT["tight"]};
      }}
      html {{ scroll-behavior: smooth; overflow-x: hidden; }}
      body, .stApp, [data-testid="stAppViewContainer"] {{
        max-width: 100%;
        overflow-x: hidden;
      }}
      .stApp {{ background: var(--gsa-color-bg); color: var(--gsa-color-text); }}
      [data-testid="stMainBlockContainer"],
      [data-testid="stAppViewContainer"] > .main .block-container {{
        max-width: {LAYOUT["content-max"]};
        padding: var(--gsa-space-section) var(--gsa-space-page-inline)
                 var(--gsa-space-page-bottom);
      }}
      [data-testid="stSidebar"] {{
        background: var(--gsa-color-surface-subtle);
        border-right: var(--gsa-border-thin) solid var(--gsa-color-border);
      }}
      h1, h2, h3, p, label, button, input, textarea, select {{
        font-family: {FONT_FAMILY} !important;
      }}
      h1 {{ letter-spacing: -0.035em; }}
      h2, h3 {{ letter-spacing: -0.02em; }}
      /* Los formularios usan el borde propio de Streamlit: uno con border=False dentro de un
         cuadro no dibuja otra caja (una pregunta, un cuadro). */
      [data-testid="stExpander"], [data-testid="stDataFrame"], [data-testid="stMetric"] {{
        background: var(--gsa-color-surface);
        border: var(--gsa-border-thin) solid var(--gsa-color-border);
        border-radius: var(--gsa-radius-md);
      }}
      [data-testid="stMetric"] {{ padding: var(--gsa-space-component); }}
      .stButton > button, [data-testid="stFormSubmitButton"] > button,
      button[data-testid="stPopoverButton"] {{
        min-height: var(--gsa-touch-target);
        border-radius: var(--gsa-radius-sm);
        font-weight: 700;
        background: var(--gsa-color-surface);
        color: var(--gsa-color-text);
        border: var(--gsa-border-thin) solid var(--gsa-color-border);
      }}
      .stButton > button[kind="secondary"]:hover,
      button[data-testid="stPopoverButton"]:hover {{
        background: var(--gsa-color-selected);
        border-color: var(--gsa-color-border-selected);
        color: var(--gsa-color-on-selected);
      }}
      .stButton > button[kind="primary"],
      [data-testid="stFormSubmitButton"] > button {{
        background: var(--gsa-color-primary);
        color: var(--gsa-color-on-primary);
        border-color: var(--gsa-color-primary);
      }}
      input, textarea, [data-baseweb="select"] > div {{ min-height: var(--gsa-touch-target); }}
      button:focus-visible, a:focus-visible, input:focus-visible, textarea:focus-visible,
      [role="combobox"]:focus-visible {{
        outline: var(--gsa-border-emphasis) solid var(--gsa-color-primary);
        outline-offset: 3px;
      }}
      .gsa-skip-link {{
        position: fixed; left: 1rem; top: -5rem; z-index: 999999;
        padding: var(--gsa-space-control-gap) var(--gsa-space-component);
        background: var(--gsa-color-primary); color: var(--gsa-color-on-primary);
        border-radius: var(--gsa-radius-sm); font-weight: 700;
      }}
      .gsa-skip-link:focus {{ top: 1rem; }}
      .gsa-hero {{
        padding: var(--gsa-space-panel) var(--gsa-space-card);
        background: linear-gradient(
          135deg, var(--gsa-color-surface) 0%, var(--gsa-color-accent-container) 160%
        );
        border: var(--gsa-border-thin) solid var(--gsa-color-border);
        border-radius: var(--gsa-radius-lg);
        margin-bottom: var(--gsa-space-component);
      }}
      .gsa-eyebrow {{
        color: var(--gsa-color-accent); font-size: var(--gsa-font-xs); font-weight: 800;
        letter-spacing: .09em; text-transform: uppercase;
      }}
      .gsa-meta {{ color: var(--gsa-color-text-muted); font-size: var(--gsa-font-sm); }}
      .gsa-stage {{
        margin: var(--gsa-space-stage) 0 var(--gsa-space-component);
        padding: var(--gsa-space-panel) var(--gsa-space-card);
        background: var(--gsa-color-surface);
        border: var(--gsa-border-thin) solid var(--gsa-color-border);
        border-left: var(--gsa-border-accent) solid var(--gsa-color-accent);
        border-radius: var(--gsa-radius-md);
      }}
      .gsa-stage-grid {{
        display: grid; grid-template-columns: repeat(3, 1fr); gap: var(--gsa-space-component);
      }}
      .gsa-stage-grid strong {{ display: block; margin-bottom: var(--gsa-space-label); }}
      .gsa-stage-grid span {{ color: var(--gsa-color-text-muted); font-size: var(--gsa-font-sm); }}
      .gsa-nav {{
        display: grid; gap: var(--gsa-space-inline-tight);
        margin: var(--gsa-space-control-gap) 0 var(--gsa-space-component);
      }}
      .gsa-nav-item {{
        display: grid; grid-template-columns: var(--gsa-step-badge) 1fr;
        gap: var(--gsa-space-control-gap); align-items: start;
        padding: var(--gsa-space-control-gap); border-radius: var(--gsa-radius-sm);
        border: var(--gsa-border-thin) solid var(--gsa-color-border);
        color: var(--gsa-color-text); text-decoration: none;
        background: var(--gsa-color-surface);
      }}
      .gsa-nav-item, .gsa-nav-item * {{ text-decoration: none !important; }}
      .gsa-nav-item {{ line-height: var(--gsa-line-tight); }}
      a.gsa-nav-item:hover {{ border-color: var(--gsa-color-border-selected); }}
      .gsa-nav-item[data-status="current"] {{
        background: var(--gsa-color-selected); color: var(--gsa-color-on-selected);
        border: var(--gsa-border-emphasis) solid var(--gsa-color-border-selected);
        box-shadow: inset var(--gsa-border-accent) 0 var(--gsa-color-border-selected);
      }}
      .gsa-nav-item[data-status="current"] .gsa-nav-number {{
        background: var(--gsa-color-border-selected); color: var(--gsa-color-on-primary);
      }}
      .gsa-nav-item[data-status="current"] .gsa-nav-help {{
        color: var(--gsa-color-on-selected); font-weight: 700;
      }}
      .gsa-nav-item[data-status="complete"] .gsa-nav-number {{
        background: var(--gsa-color-success-container); color: var(--gsa-color-success);
      }}
      .gsa-nav-item[data-status="locked"] {{
        background: var(--gsa-color-disabled); color: var(--gsa-color-on-disabled);
        border: var(--gsa-border-thin) dashed var(--gsa-color-border-disabled);
        cursor: not-allowed;
      }}
      .gsa-nav-item[data-status="locked"] .gsa-nav-number {{
        background: var(--gsa-color-surface); color: var(--gsa-color-on-disabled);
        border: var(--gsa-border-thin) dashed var(--gsa-color-border-disabled);
      }}
      .gsa-nav-number {{
        width: var(--gsa-step-badge); height: var(--gsa-step-badge);
        border-radius: var(--gsa-radius-pill); display: grid; place-items: center;
        background: var(--gsa-color-primary-container);
        color: var(--gsa-color-on-primary-container); font-weight: 800;
      }}
      .gsa-nav-title {{ font-size: var(--gsa-font-sm); font-weight: 750; }}
      .gsa-nav-help {{ color: var(--gsa-color-text-muted); font-size: var(--gsa-font-xs); }}
      .gsa-nav-divider {{
        margin: var(--gsa-space-control-gap) 0 var(--gsa-space-label);
        color: var(--gsa-color-text-muted);
        font-size: var(--gsa-font-xs);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
      }}
      .gsa-nav-item[data-status="locked"] .gsa-nav-help {{ color: var(--gsa-color-on-disabled); }}
      .gsa-project-card {{
        display: grid; gap: var(--gsa-space-inline-tight);
        margin: var(--gsa-space-control-gap) 0;
        padding: var(--gsa-space-component); border-radius: var(--gsa-radius-md);
        background: var(--gsa-color-selected); color: var(--gsa-color-on-selected);
        border: var(--gsa-border-emphasis) solid var(--gsa-color-border-selected);
      }}
      .gsa-project-eyebrow {{
        font-size: var(--gsa-font-xs); font-weight: 800; text-transform: uppercase;
      }}
      .gsa-project-name {{ font-size: var(--gsa-font-base); overflow-wrap: anywhere; }}
      .gsa-project-step {{ font-size: var(--gsa-font-sm); }}
      [data-testid="stSidebar"] [data-testid="stPopover"] button {{
        width: 100%; min-height: var(--gsa-touch-target); font-weight: 700;
        border-radius: var(--gsa-radius-sm);
      }}
      .st-key-create-project-popover button {{
        background: var(--gsa-color-primary); color: var(--gsa-color-on-primary);
        border-color: var(--gsa-color-primary);
      }}
      .st-key-archive-project-popover button {{
        background: var(--gsa-color-surface); color: var(--gsa-color-text);
        border: var(--gsa-border-thin) solid var(--gsa-color-border);
      }}
      .st-key-archive-project-submit button {{
        background: var(--gsa-color-danger-container); color: var(--gsa-color-danger);
        border: var(--gsa-border-emphasis) solid var(--gsa-color-danger);
      }}
      .st-key-archive-project-submit button:disabled {{
        background: var(--gsa-color-disabled); color: var(--gsa-color-on-disabled);
        border: var(--gsa-border-thin) dashed var(--gsa-color-border-disabled);
      }}
      .gsa-empty {{
        padding: var(--gsa-space-panel); background: var(--gsa-color-surface-subtle);
        border: var(--gsa-border-thin) dashed {color("secondary")};
        border-radius: var(--gsa-radius-md);
      }}
      .gsa-field-help {{
        margin: var(--gsa-space-panel) 0 var(--gsa-space-label);
        max-width: {LAYOUT["text-max"]};
        color: var(--gsa-color-text-muted); font-size: var(--gsa-font-sm);
        line-height: {LINE_HEIGHT["body"]};
      }}
      .gsa-field-title {{
        display: block; margin-bottom: var(--gsa-space-label);
        color: var(--gsa-color-text); font-size: var(--gsa-font-base); font-weight: 700;
      }}
      /* UX-06: textos del cargador de archivos de Streamlit en español (Streamlit 1.65). */
      [data-testid="stFileUploaderDropzone"] button [data-testid="stMarkdownContainer"] {{
        display: none;
      }}
      [data-testid="stFileUploaderDropzone"] button::after {{
        content: "Elegir archivo"; font-size: var(--gsa-font-sm);
        margin-left: var(--gsa-space-inline-tight);
      }}
      [data-testid="stFileUploaderDropzoneInstructions"] > div {{ display: none; }}
      [data-testid="stFileUploaderDropzoneInstructions"]::after {{
        content: "Arrástralo aquí o elígelo · CSV de hasta 25 MB";
        color: var(--gsa-color-text-muted); font-size: var(--gsa-font-xs);
      }}
      .gsa-field-state {{
        display: inline-block; margin-right: var(--gsa-space-inline-tight);
        padding: 0 var(--gsa-space-inline-tight);
        border-radius: var(--gsa-radius-pill); background: var(--gsa-color-accent-container);
        color: var(--gsa-color-on-accent-container); font-size: var(--gsa-font-xs);
        font-weight: 700;
      }}
      .stButton > button[kind="primary"]:hover,
      .stButton > button[kind="primary"]:active,
      .stButton > button[kind="primary"]:focus,
      [data-testid="stFormSubmitButton"] > button:hover,
      [data-testid="stFormSubmitButton"] > button:active,
      [data-testid="stFormSubmitButton"] > button:focus {{
        color: var(--gsa-color-on-primary);
        background: var(--gsa-color-primary-hover);
      }}
      .stButton > button:disabled,
      [data-testid="stFormSubmitButton"] > button:disabled {{
        background: var(--gsa-color-surface-subtle);
        color: var(--gsa-color-text-muted);
        border: var(--gsa-border-thin) dashed var(--gsa-color-border);
        cursor: not-allowed; opacity: 1; filter: none;
      }}
      .gsa-profile {{
        display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: var(--gsa-space-control-gap);
        margin: var(--gsa-space-control-gap) 0 var(--gsa-space-component);
      }}
      .gsa-profile-card {{
        padding: var(--gsa-space-component); background: var(--gsa-color-surface);
        border: var(--gsa-border-thin) solid var(--gsa-color-border);
        border-radius: var(--gsa-radius-sm);
      }}
      .gsa-profile-value {{ font-size: {type_size("section-title")}; font-weight: 800; }}
      .gsa-profile-label {{ color: var(--gsa-color-text-muted); font-size: var(--gsa-font-xs); }}
      .gsa-anchor {{ scroll-margin-top: var(--gsa-space-card); }}
      @media (max-width: 900px) {{
        [data-testid="stMainBlockContainer"],
        [data-testid="stAppViewContainer"] > .main .block-container {{
          padding: var(--gsa-space-panel) var(--gsa-space-component)
                   var(--gsa-space-page-bottom-tablet);
        }}
        .gsa-stage-grid, .gsa-profile {{ grid-template-columns: 1fr; }}
      }}
      @media (max-width: 640px) {{
        [data-testid="stMainBlockContainer"],
        [data-testid="stAppViewContainer"] > .main .block-container {{
          width: 100%;
          max-width: 100%;
          padding: var(--gsa-space-component) var(--gsa-space-component)
                   var(--gsa-space-page-bottom-compact);
          overflow-x: hidden;
        }}
        [data-testid="stSidebar"][aria-expanded="true"],
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {{
          width: min(85vw, {LAYOUT["sidebar-max"]});
          min-width: min(85vw, {LAYOUT["sidebar-max"]});
          max-width: min(85vw, {LAYOUT["sidebar-max"]});
        }}
        .gsa-hero, .gsa-stage, .gsa-field-card, .gsa-profile-card,
        [data-testid="stForm"], [data-testid="stExpander"],
        [data-testid="stDataFrame"], [data-testid="stMetric"] {{
          min-width: 0;
          max-width: 100%;
          box-sizing: border-box;
        }}
        .gsa-hero, .gsa-stage {{ padding: var(--gsa-space-component); }}
        h1, h2, h3, p, label, .gsa-field-card, .gsa-nav-item {{
          overflow-wrap: anywhere;
        }}
      }}
      @media (prefers-reduced-motion: reduce) {{ html {{ scroll-behavior: auto; }} }}
    </style>
    """


def apply_theme() -> None:
    st.markdown(build_theme_css(), unsafe_allow_html=True)
