#!/usr/bin/env python3
"""Generate a clean, professional PPTX defense deck from the TFG memory assets."""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt


BASE_DIR = Path(__file__).resolve().parents[1]
MEMORIA_DIR = BASE_DIR / "memoria"
IMAGES_DIR = MEMORIA_DIR / "Imagenes"
DATA_FILE = MEMORIA_DIR / "_datos_proyecto.tex"
OUTPUT_FILE = MEMORIA_DIR / "presentacion_defensa_tfg_uco.pptx"

# Slide geometry (16:9).
SLIDE_W_IN = 13.333
SLIDE_H_IN = 7.5

# Typography (keep Office-safe defaults).
FONT = "Calibri"

# University-inspired palette (from memoria/Portada/cover.tex).
PRIMARY = RGBColor(40, 0, 145)  # #280091
SECONDARY = RGBColor(76, 92, 197)  # #4C5CC5
ACCENT = RGBColor(0, 178, 153)  # #00B299
ACCENT_LIGHT = RGBColor(63, 207, 213)  # #3FCFD5

TEXT_DARK = RGBColor(33, 37, 41)
TEXT_MID = RGBColor(84, 89, 95)

BG_LIGHT = RGBColor(245, 247, 250)
BG_WHITE = RGBColor(255, 255, 255)
BORDER = RGBColor(218, 224, 233)


def parse_project_data(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    pattern = re.compile(r"\\newcommand\{\\(\w+)\}\{(.+?)\}")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line.strip())
        if match:
            data[match.group(1)] = match.group(2)
    return data


def set_solid_background(slide, color: RGBColor) -> None:
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(0),
        Inches(0),
        Inches(SLIDE_W_IN),
        Inches(SLIDE_H_IN),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def add_backdrop(slide) -> None:
    """Subtle decorative shapes for a less flat look (very low-contrast)."""
    s1 = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(-1.6), Inches(4.6), Inches(4.9), Inches(4.9))
    s1.fill.solid()
    s1.fill.fore_color.rgb = ACCENT_LIGHT
    s1.fill.transparency = 0.90
    s1.line.fill.background()

    s2 = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(10.2), Inches(-2.2), Inches(5.6), Inches(5.6))
    s2.fill.solid()
    s2.fill.fore_color.rgb = SECONDARY
    s2.fill.transparency = 0.92
    s2.line.fill.background()


def add_cover_backdrop(slide) -> None:
    band = slide.shapes.add_shape(MSO_SHAPE.PARALLELOGRAM, Inches(-1.0), Inches(5.25), Inches(16.5), Inches(3.2))
    band.fill.solid()
    band.fill.fore_color.rgb = SECONDARY
    band.fill.transparency = 0.18
    band.line.fill.background()

    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0.95), Inches(SLIDE_W_IN), Inches(0.07))
    line.fill.solid()
    line.fill.fore_color.rgb = ACCENT
    line.line.fill.background()


def add_logo(slide, left: float = 11.85, top: float = 0.18, width: float = 1.25, badge: bool = True) -> None:
    logo = IMAGES_DIR / "logo_uco.png"
    if not logo.exists():
        return
    if badge:
        b = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(left - 0.10),
            Inches(top - 0.02),
            Inches(width + 0.20),
            Inches(0.56),
        )
        b.fill.solid()
        b.fill.fore_color.rgb = BG_WHITE
        b.line.color.rgb = BORDER
    slide.shapes.add_picture(str(logo), Inches(left), Inches(top + 0.06), width=Inches(width))


def add_header(slide, title: str, subtitle: str | None = None) -> None:
    set_solid_background(slide, BG_LIGHT)
    add_backdrop(slide)

    header_h = 0.95
    header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(SLIDE_W_IN), Inches(header_h))
    header.fill.solid()
    header.fill.fore_color.rgb = BG_WHITE
    header.line.fill.background()

    stripe = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.24), Inches(header_h))
    stripe.fill.solid()
    stripe.fill.fore_color.rgb = PRIMARY
    stripe.line.fill.background()

    accent = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(header_h), Inches(SLIDE_W_IN), Inches(0.05))
    accent.fill.solid()
    accent.fill.fore_color.rgb = ACCENT
    accent.line.fill.background()

    title_box = slide.shapes.add_textbox(Inches(0.55), Inches(0.16), Inches(10.6), Inches(0.34))
    tf = title_box.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = title
    p.font.name = FONT
    p.font.size = Pt(26)
    p.font.bold = True
    p.font.color.rgb = PRIMARY

    if subtitle:
        sub_box = slide.shapes.add_textbox(Inches(0.55), Inches(0.54), Inches(10.6), Inches(0.28))
        sub_tf = sub_box.text_frame
        sub_tf.clear()
        sp = sub_tf.paragraphs[0]
        sp.text = subtitle
        sp.font.name = FONT
        sp.font.size = Pt(14)
        sp.font.color.rgb = TEXT_MID

    add_logo(slide, left=11.92, top=0.18, width=1.18, badge=False)


def add_footer(slide, idx: int, total: int) -> None:
    base = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(SLIDE_H_IN - 0.06), Inches(SLIDE_W_IN), Inches(0.06))
    base.fill.solid()
    base.fill.fore_color.rgb = BORDER
    base.line.fill.background()

    n_box = slide.shapes.add_textbox(Inches(SLIDE_W_IN - 1.05), Inches(SLIDE_H_IN - 0.38), Inches(0.85), Inches(0.25))
    tf = n_box.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = f"{idx}/{total}"
    p.alignment = PP_ALIGN.RIGHT
    p.font.name = FONT
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MID


def add_text_block(
    slide,
    text: str,
    left: float,
    top: float,
    width: float,
    height: float,
    *,
    size: int = 18,
    bold: bool = False,
    color: RGBColor = TEXT_DARK,
    align=PP_ALIGN.LEFT,
) -> None:
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.alignment = align
    p.font.name = FONT
    p.font.size = Pt(size)
    p.font.bold = bold
    p.font.color.rgb = color


def add_bullets(
    slide,
    items: list[str],
    left: float,
    top: float,
    width: float,
    height: float,
    *,
    size: int = 20,
    color: RGBColor = TEXT_DARK,
    spacing: float = 1.15,
) -> None:
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    for idx, item in enumerate(items):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = f"• {item}"
        p.level = 0
        p.font.name = FONT
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.space_after = Pt(6)
        p.line_spacing = spacing


def add_card(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    title: str,
    body: str,
    *,
    title_color: RGBColor = PRIMARY,
) -> None:
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    card.fill.solid()
    card.fill.fore_color.rgb = BG_WHITE
    card.line.color.rgb = BORDER

    title_box = slide.shapes.add_textbox(Inches(left + 0.18), Inches(top + 0.12), Inches(width - 0.36), Inches(0.33))
    tf_t = title_box.text_frame
    tf_t.clear()
    tp = tf_t.paragraphs[0]
    tp.text = title
    tp.font.name = FONT
    tp.font.size = Pt(16 if width >= 3.6 else 15)
    tp.font.bold = True
    tp.font.color.rgb = title_color

    body_box = slide.shapes.add_textbox(Inches(left + 0.18), Inches(top + 0.46), Inches(width - 0.36), Inches(height - 0.62))
    tf_b = body_box.text_frame
    tf_b.clear()
    tf_b.word_wrap = True

    lines = [ln.rstrip() for ln in body.splitlines() if ln.strip()]
    for idx, line in enumerate(lines):
        p = tf_b.paragraphs[0] if idx == 0 else tf_b.add_paragraph()
        p.text = line
        p.font.name = FONT
        p.font.size = Pt(14)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(2)
        p.line_spacing = 1.15


def add_image_contain(slide, image_path: Path, left: float, top: float, width: float, height: float) -> None:
    if not image_path.exists():
        return
    with Image.open(image_path) as img:
        iw, ih = img.size

    box_ratio = width / height
    img_ratio = iw / ih
    if img_ratio > box_ratio:
        draw_w = width
        draw_h = width / img_ratio
        draw_l = left
        draw_t = top + (height - draw_h) / 2
    else:
        draw_h = height
        draw_w = height * img_ratio
        draw_l = left + (width - draw_w) / 2
        draw_t = top

    slide.shapes.add_picture(str(image_path), Inches(draw_l), Inches(draw_t), width=Inches(draw_w), height=Inches(draw_h))


def add_image_card(slide, image_path: Path, left: float, top: float, width: float, height: float, *, pad: float = 0.14) -> None:
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    card.fill.solid()
    card.fill.fore_color.rgb = BG_WHITE
    card.line.color.rgb = BORDER
    add_image_contain(slide, image_path, left + pad, top + pad, width - 2 * pad, height - 2 * pad)


def build_presentation() -> Path:
    project = parse_project_data(DATA_FILE)
    titulo = project.get("TituloTFG", "Defensa TFG")
    autor = project.get("AutorTFG", "Autor/a")
    director = project.get("DirectorUno", "Director/a")
    grado = project.get("Grado", "Grado en Ingeniería Informática")

    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W_IN)
    prs.slide_height = Inches(SLIDE_H_IN)
    blank = prs.slide_layouts[6]

    # 1) Portada
    slide = prs.slides.add_slide(blank)
    set_solid_background(slide, PRIMARY)
    add_cover_backdrop(slide)
    add_logo(slide, left=11.80, top=0.35, width=1.28, badge=True)

    add_text_block(slide, "Universidad de Córdoba", 0.85, 0.32, 7.8, 0.35, size=18, bold=True, color=ACCENT_LIGHT)
    add_text_block(slide, "Escuela Politécnica Superior de Córdoba", 0.85, 0.72, 9.0, 0.3, size=14, color=BG_WHITE)

    add_text_block(slide, "TRABAJO FIN DE GRADO", 0.85, 1.35, 7.2, 0.35, size=16, bold=True, color=ACCENT_LIGHT)
    # Title is long; keep it large but safely within bounds when projected.
    add_text_block(slide, titulo, 0.85, 1.70, 11.2, 1.95, size=34, bold=True, color=BG_WHITE)
    add_text_block(slide, grado, 0.85, 3.78, 10.0, 0.35, size=16, color=BG_WHITE)
    add_text_block(slide, f"Autor: {autor}\nDirector: {director}", 0.85, 4.28, 7.5, 0.9, size=16, color=BG_WHITE)

    bottom = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(SLIDE_H_IN - 0.9), Inches(SLIDE_W_IN), Inches(0.9))
    bottom.fill.solid()
    bottom.fill.fore_color.rgb = SECONDARY
    bottom.fill.transparency = 0.22
    bottom.line.fill.background()
    add_text_block(slide, "Predicción financiera · Machine Learning · Sentimiento de noticias", 0.85, 6.82, 12.0, 0.3, size=14, color=BG_WHITE)

    # 2) Estructura
    slide = prs.slides.add_slide(blank)
    add_header(slide, "Estructura de la presentación", "Resumen del guion")
    add_bullets(
        slide,
        [
            "Motivación, problema y objetivos",
            "Datos, variables y metodología experimental",
            "Resultados clave (experimentos A-D)",
            "Conclusiones, limitaciones y trabajo futuro",
        ],
        0.9,
        1.50,
        7.1,
        3.2,
        size=20,
    )
    bar = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.9), Inches(5.2), Inches(11.6), Inches(0.95))
    bar.fill.solid()
    bar.fill.fore_color.rgb = BG_WHITE
    bar.line.color.rgb = BORDER
    labels = ["Contexto", "Método", "Resultados", "Cierre"]
    colors = [ACCENT, SECONDARY, SECONDARY, ACCENT]
    x0, y0, w, h = 1.05, 5.37, 11.3, 0.32
    seg_w = w / 4
    for i, (lab, col) in enumerate(zip(labels, colors, strict=True)):
        seg = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(x0 + i * seg_w),
            Inches(y0),
            Inches(seg_w - 0.05),
            Inches(h),
        )
        seg.fill.solid()
        seg.fill.fore_color.rgb = col
        seg.line.fill.background()
        seg.fill.transparency = 0.15
        add_text_block(slide, lab, x0 + i * seg_w, 5.72, seg_w, 0.25, size=13, color=TEXT_MID, align=PP_ALIGN.CENTER)

    # 3) Motivación
    slide = prs.slides.add_slide(blank)
    add_header(slide, "Motivación y pregunta de investigación")
    add_bullets(
        slide,
        [
            "Mercados financieros: alta volatilidad, ruido y cambios de régimen.",
            "Interés creciente por modelos de ML aplicados a predicción de precios.",
            "Hipótesis: el sentimiento de noticias puede aportar señal adicional.",
        ],
        0.9,
        1.55,
        7.2,
        2.6,
        size=19,
    )
    question = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.9), Inches(4.8), Inches(11.55), Inches(1.3))
    question.fill.solid()
    question.fill.fore_color.rgb = RGBColor(233, 247, 245)
    question.line.color.rgb = ACCENT
    add_text_block(
        slide,
        "¿Aporta el sentimiento del mercado una mejora real y consistente\n"
        "en la predicción del precio de cierre frente a usar solo histórico de precios?",
        1.15,
        5.05,
        11.0,
        0.85,
        size=19,
        bold=True,
        color=PRIMARY,
        align=PP_ALIGN.CENTER,
    )

    # 4) Estado del arte
    slide = prs.slides.add_slide(blank)
    add_header(slide, "Estado del arte y preguntas de investigación")
    add_bullets(
        slide,
        [
            "Series financieras: baja relación señal/ruido y cambios de régimen frecuentes.",
            "Baselines simples (Naive/ARIMA) suelen ser referencias difíciles de batir.",
            "La literatura sobre sentimiento muestra resultados mixtos y dependientes del activo.",
            "Preguntas clave: ¿qué familia de modelos generaliza mejor? ¿cuándo aporta el sentimiento?",
        ],
        0.9,
        1.55,
        7.4,
        3.5,
        size=17,
    )
    add_card(
        slide,
        8.55,
        1.55,
        3.95,
        2.1,
        "Hipótesis",
        "La señal de sentimiento\npuede mejorar la predicción,\npero su efecto es condicional\nal activo y al modelo.",
    )
    add_card(
        slide,
        8.55,
        3.95,
        3.95,
        2.1,
        "Aportación del TFG",
        "Comparativa reproducible\nentre 7 modelos + ablación\ncon validación temporal\ny tests estadísticos.",
    )

    # 5) Objetivos
    slide = prs.slides.add_slide(blank)
    add_header(slide, "Objetivos del trabajo", "Visión de lo que se construye y se evalúa")
    add_bullets(
        slide,
        [
            "Construir pipeline reproducible de datos (precios + sentimiento).",
            "Comparar 7 modelos: Naive, ARIMA, RF, XGBoost, LSTM, GRU y CNN.",
            "Evaluar con validación walk-forward y métricas de error/dirección.",
            "Medir impacto del sentimiento con ablación y tests estadísticos.",
        ],
        0.9,
        1.55,
        7.5,
        4.8,
        size=18,
    )
    add_card(
        slide,
        left=8.4,
        top=1.55,
        width=4.1,
        height=2.1,
        title="Activos estudiados",
        body="BTC-USD, TSLA y NVDA\nperfiles distintos para comparar robustez.",
    )
    add_card(
        slide,
        left=8.4,
        top=3.95,
        width=4.1,
        height=2.1,
        title="Horizontes",
        body="Principal: 7 días\nComplementarios: 1 y 30 días.",
    )

    # 6) Datos
    slide = prs.slides.add_slide(blank)
    add_header(slide, "Datos y variables", "Periodo útil: enero 2021 - noviembre 2025")
    add_card(
        slide,
        left=0.9,
        top=1.3,
        width=3.8,
        height=2.2,
        title="Fuente de precios",
        body="Yahoo Finance (OHLCV)\nSeries ajustadas (auto_adjust=True).",
    )
    add_card(
        slide,
        left=4.95,
        top=1.3,
        width=3.8,
        height=2.2,
        title="Fuente de sentimiento",
        body="GDELT Average Tone +\nintensidad de cobertura por activo.",
    )
    add_card(
        slide,
        left=9.0,
        top=1.3,
        width=3.4,
        height=2.2,
        title="Preprocesado",
        body="Alineación temporal,\nforward-fill y escalado sin fuga.",
    )
    add_bullets(
        slide,
        [
            "Indicadores técnicos: SMA20/50, RSI, MACD, Bandas de Bollinger.",
            "Ventanas temporales de 30 días en DL; retardos de 10 días en ML tabular.",
            "Targets y evaluación final siempre sobre precio futuro (USD).",
        ],
        0.95,
        4.15,
        11.7,
        2.2,
        size=17,
    )

    # 7) Metodología + diseño
    slide = prs.slides.add_slide(blank)
    add_header(slide, "Metodología y diseño experimental", "Walk-forward, métricas y experimentos A-D")
    add_bullets(
        slide,
        [
            "Validación walk-forward: 12 folds × 30 días = 360 predicciones por activo.",
            "Sin fuga temporal: entrenamiento solo con pasado; predicción directa a t+h.",
            "Métricas: RMSE, MAE, MAPE; y direccionales: DA, F1, MCC.",
            "Contrastes: Wilcoxon (ablación) y Diebold-Mariano (vs Naive).",
        ],
        0.9,
        1.55,
        6.35,
        3.2,
        size=16,
    )
    add_card(slide, 7.05, 1.55, 5.55, 1.25, "Experimento A", "Comparación de modelos (h=7) en BTC-USD, TSLA y NVDA.")
    add_card(slide, 7.05, 2.95, 5.55, 1.25, "Experimento B", "Impacto del horizonte (1, 7 y 30 días) en BTC-USD.")
    add_card(slide, 7.05, 4.35, 5.55, 1.25, "Experimento C", "Ablación con y sin sentimiento en modelos DL (CNN/LSTM/GRU).")
    add_card(slide, 7.05, 5.75, 5.55, 1.25, "Experimento D", "Rendimiento por periodos de mercado (BTC-USD, h=7).")

    # 8) Modelos
    slide = prs.slides.add_slide(blank)
    add_header(slide, "Modelos comparados")
    add_card(slide, 0.9, 1.55, 3.9, 2.1, "Baselines", "Naive\nARIMA(5,1,0)")
    add_card(slide, 0.9, 3.95, 3.9, 2.1, "ML clásico", "Random Forest\nXGBoost")
    add_card(slide, 5.05, 1.55, 3.9, 2.1, "Deep Learning", "LSTM (2 capas, dropout 20%)\nGRU (2 capas, dropout 20%)")
    add_card(slide, 5.05, 3.95, 3.9, 2.1, "Deep Learning", "CNN 1D\n(64 filtros, kernel=2)")
    add_card(
        slide,
        9.2,
        1.55,
        3.25,
        4.5,
        "Configuración",
        "DL: 10 épocas, Adam, MSE.\n"
        "Ventana: 30 días.\n"
        "Semilla fija (42).\n\n"
        "Objetivo: comparar familias\nde modelos bajo un\nprotocolo reproducible.",
    )

    # 9) Resultados A
    slide = prs.slides.add_slide(blank)
    add_header(slide, "Resultados A: comparación de modelos (h = 7)")
    add_bullets(
        slide,
        [
            "ARIMA logra el mejor MAPE medio: 6.99%.",
            "Naive queda prácticamente empatado: 7.01%.",
            "Mejor por activo: Naive (BTC-USD), ARIMA (TSLA y NVDA).",
            "RF/XGBoost no superan baselines; DL queda claramente detrás.",
        ],
        0.85,
        1.55,
        6.2,
        3.2,
        size=16,
    )
    table_bg = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.9), Inches(4.95), Inches(6.15), Inches(1.55))
    table_bg.fill.solid()
    table_bg.fill.fore_color.rgb = BG_WHITE
    table_bg.line.color.rgb = BORDER
    add_text_block(
        slide,
        "MAPE medio (3 activos):  ARIMA 6.99 | Naive 7.01 | RF 7.93 | XGB 8.20 | GRU 9.68 | LSTM 11.23 | CNN 12.47",
        1.1,
        5.22,
        5.8,
        1.05,
        size=14,
        color=TEXT_MID,
    )
    add_image_card(slide, IMAGES_DIR / "cross_asset_heatmap.png", 7.15, 1.45, 5.95, 5.15, pad=0.12)

    # 10) Vista global
    slide = prs.slides.add_slide(blank)
    add_header(slide, "Vista global del rendimiento")
    add_image_card(slide, IMAGES_DIR / "summary_model_ranking.png", 0.8, 1.45, 7.95, 4.95, pad=0.12)
    add_image_card(slide, IMAGES_DIR / "summary_best_models.png", 8.95, 1.45, 3.95, 2.1, pad=0.12)
    add_image_card(slide, IMAGES_DIR / "summary_dl_vs_baselines.png", 8.95, 3.75, 3.95, 2.65, pad=0.12)
    add_text_block(
        slide,
        "Mensaje clave: en esta configuración, DL no justifica su coste frente a Naive/ARIMA.",
        0.95,
        6.35,
        11.8,
        0.4,
        size=15,
        bold=True,
        color=PRIMARY,
        align=PP_ALIGN.CENTER,
    )

    # 11) Direccionalidad
    slide = prs.slides.add_slide(blank)
    add_header(slide, "Resultados: métricas direccionales", "DA, F1 y MCC (h = 7)")
    add_image_card(slide, IMAGES_DIR / "directional_accuracy_comparison.png", 0.8, 1.45, 7.35, 5.15, pad=0.12)
    add_bullets(
        slide,
        [
            "Resultados modestos: muchas combinaciones cerca del 50% (azar).",
            "ARIMA es el más estable en DA medio (51.11%).",
            "Naive tiene F1 direccional = 0 por predicción plana (cambio = 0).",
            "Conclusión: la dirección es sensible al activo y al periodo.",
        ],
        8.4,
        1.65,
        4.6,
        4.8,
        size=16,
    )

    # 12) Horizonte
    slide = prs.slides.add_slide(blank)
    add_header(slide, "Resultados B: impacto del horizonte (BTC-USD)")
    add_image_card(slide, IMAGES_DIR / "horizon_impact_all_models.png", 0.8, 1.45, 7.25, 5.15, pad=0.12)
    add_bullets(
        slide,
        [
            "Todos los modelos empeoran al alargar horizonte.",
            "GRU es el más robusto dentro de DL: 5.58% (1d), 10.13% (7d), 20.11% (30d).",
            "LSTM y CNN se degradan más; CNN es el peor en 7d y 30d.",
            "Conclusión operativa: priorizar horizontes cortos.",
        ],
        8.35,
        1.65,
        4.5,
        4.9,
        size=16,
    )

    # 13) Predicción vs realidad
    slide = prs.slides.add_slide(blank)
    add_header(slide, "Análisis cualitativo", "Predicción vs realidad (h = 7)")
    add_image_card(slide, IMAGES_DIR / "pred_vs_actual_BTC_USD_h7.png", 0.8, 1.45, 3.95, 3.85, pad=0.10)
    add_image_card(slide, IMAGES_DIR / "pred_vs_actual_TSLA_h7.png", 4.79, 1.45, 3.95, 3.85, pad=0.10)
    add_image_card(slide, IMAGES_DIR / "pred_vs_actual_NVDA_h7.png", 8.78, 1.45, 3.95, 3.85, pad=0.10)
    add_text_block(slide, "BTC-USD", 0.8, 5.35, 3.95, 0.25, size=13, color=TEXT_MID, align=PP_ALIGN.CENTER)
    add_text_block(slide, "TSLA", 4.79, 5.35, 3.95, 0.25, size=13, color=TEXT_MID, align=PP_ALIGN.CENTER)
    add_text_block(slide, "NVDA", 8.78, 5.35, 3.95, 0.25, size=13, color=TEXT_MID, align=PP_ALIGN.CENTER)
    add_bullets(
        slide,
        [
            "Las predicciones suavizan picos y suelen reaccionar con retardo.",
            "Los shocks externos marcan los mayores errores.",
            "A mayor horizonte, más tendencia a la “media suavizada”.",
        ],
        0.9,
        5.78,
        12.2,
        1.20,
        size=14,
    )

    # 14) Ablación
    slide = prs.slides.add_slide(blank)
    add_header(slide, "Resultados C: ablación del sentimiento")
    add_image_card(slide, IMAGES_DIR / "ablation_improvement_heatmap.png", 0.8, 1.45, 6.15, 5.15, pad=0.12)
    add_bullets(
        slide,
        [
            "Efecto heterogéneo, no universal.",
            "BTC-USD: empeora en CNN (-25.1%), GRU (-10.6%) y LSTM (-9.5%).",
            "TSLA: mejora ligera en los tres modelos (+2.7% a +3.6%).",
            "NVDA: mejora solo en LSTM (+2.2%); CNN y GRU empeoran.",
        ],
        7.15,
        1.60,
        5.6,
        4.65,
        size=15,
    )
    add_image_card(slide, IMAGES_DIR / "summary_sentiment_impact.png", 7.15, 5.35, 5.45, 1.25, pad=0.10)

    # 15) Significancia
    slide = prs.slides.add_slide(blank)
    add_header(slide, "Significancia estadística")
    add_bullets(
        slide,
        [
            "Wilcoxon (con vs sin sentimiento):",
            "No hay diferencias significativas en MAPE, DA ni F1 (p<0.05).",
            "Solo un caso aislado en MCC (NVDA-CNN).",
            "Diebold-Mariano frente a Naive: no hay mejoras significativas.",
            "Sí hay varios empeoramientos significativos (sobre todo en BTC-USD).",
        ],
        0.8,
        1.55,
        5.7,
        5.0,
        size=15,
    )
    add_image_card(slide, IMAGES_DIR / "wilcoxon_pvalues_heatmap.png", 6.20, 2.25, 3.45, 2.95, pad=0.10)
    add_image_card(slide, IMAGES_DIR / "diebold_mariano_pvalues_heatmap.png", 9.78, 2.25, 3.45, 2.95, pad=0.10)
    add_text_block(slide, "Wilcoxon", 6.20, 5.28, 3.45, 0.25, size=12, color=TEXT_MID, align=PP_ALIGN.CENTER)
    add_text_block(slide, "Diebold-Mariano", 9.78, 5.28, 3.45, 0.25, size=12, color=TEXT_MID, align=PP_ALIGN.CENTER)

    # 16) Regímenes de mercado
    slide = prs.slides.add_slide(blank)
    add_header(slide, "Resultados D: rendimiento por periodos de mercado (BTC-USD)")
    add_image_card(slide, IMAGES_DIR / "market_periods_all_metrics.png", 6.6, 1.45, 6.35, 5.15, pad=0.12)
    add_bullets(
        slide,
        [
            "Bull-2021: GRU lidera (MAPE 8.71%).",
            "Bear-2022: LSTM lidera (MAPE 6.50%).",
            "Periodo reciente: GRU vuelve a liderar (MAPE 3.71%).",
            "El ranking cambia con el régimen de mercado.",
            "Implicación: monitorizar contexto antes de elegir modelo.",
        ],
        0.85,
        1.70,
        5.55,
        4.9,
        size=16,
    )

    # 17) Lectura práctica
    slide = prs.slides.add_slide(blank)
    add_header(slide, "Lectura práctica", "Magnitud del error y consistencia")
    add_bullets(
        slide,
        [
            "MAPE aparentemente bajo puede implicar errores absolutos elevados.",
            "BTC (~6%): error medio de ~3.200 USD.",
            "TSLA (~8.3%): error medio de ~21 USD.",
            "NVDA (~6.6%): error medio de ~4 USD.",
            "Además de la media, importa la dispersión (estabilidad por fold).",
        ],
        0.85,
        1.65,
        5.75,
        4.8,
        size=16,
    )
    add_image_card(slide, IMAGES_DIR / "error_distribution_boxplot.png", 6.7, 1.45, 6.25, 5.15, pad=0.12)
    add_text_block(
        slide,
        "Error bajo ≠ utilidad económica automática (faltaría backtesting con costes reales).",
        0.9,
        6.35,
        11.9,
        0.35,
        size=13,
        bold=True,
        color=PRIMARY,
        align=PP_ALIGN.CENTER,
    )

    # 18) Conclusiones
    slide = prs.slides.add_slide(blank)
    add_header(slide, "Conclusiones")
    callout = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.9), Inches(1.35), Inches(11.55), Inches(1.35))
    callout.fill.solid()
    callout.fill.fore_color.rgb = RGBColor(231, 242, 255)
    callout.line.color.rgb = SECONDARY
    add_text_block(
        slide,
        "Respuesta a la pregunta central: el sentimiento NO mejora de forma clara y robusta\n"
        "la predicción en esta configuración experimental.",
        1.15,
        1.68,
        11.0,
        0.85,
        size=19,
        bold=True,
        color=PRIMARY,
        align=PP_ALIGN.CENTER,
    )
    add_bullets(
        slide,
        [
            "Baselines (Naive/ARIMA) siguen siendo las referencias más sólidas.",
            "DL aporta flexibilidad, pero no mejora MAPE bajo este protocolo.",
            "El efecto del sentimiento es pequeño, condicional y no significativo.",
            "La validación estadística es imprescindible antes de reclamar ventaja predictiva.",
        ],
        1.1,
        3.25,
        11.0,
        2.7,
        size=19,
    )

    # 19) Limitaciones + futuro + cierre
    slide = prs.slides.add_slide(blank)
    add_header(slide, "Limitaciones, trabajo futuro y cierre")
    add_card(
        slide,
        0.9,
        1.45,
        5.8,
        4.75,
        "Limitaciones",
        "• Sin tuning exhaustivo de hiperparámetros.\n"
        "• Tres activos: alcance limitado para generalización.\n"
        "• Test total de 360 días aún sensible al régimen.\n"
        "• Sin backtesting de estrategias con costes reales.",
    )
    add_card(
        slide,
        6.95,
        1.45,
        5.45,
        4.75,
        "Trabajo futuro",
        "• Probar arquitecturas Transformer para forecasting.\n"
        "• Sentimiento más granular con NLP sobre noticias/tweets.\n"
        "• Ampliar activos y ventanas temporales.\n"
        "• Evaluar utilidad económica con backtesting completo.",
    )
    thanks = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(6.65), Inches(SLIDE_W_IN), Inches(0.85))
    thanks.fill.solid()
    thanks.fill.fore_color.rgb = PRIMARY
    thanks.line.fill.background()
    add_text_block(
        slide,
        "Gracias por su atención · Turno de preguntas",
        0.0,
        6.87,
        SLIDE_W_IN,
        0.4,
        size=21,
        bold=True,
        color=BG_WHITE,
        align=PP_ALIGN.CENTER,
    )

    # Add slide numbers (exclude cover and closing slide).
    body_total = len(prs.slides) - 1
    last_idx = len(prs.slides) - 1
    for i, s in enumerate(prs.slides, start=0):
        if i in (0, last_idx):
            continue
        add_footer(s, i, body_total)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUTPUT_FILE)
    return OUTPUT_FILE


if __name__ == "__main__":
    out = build_presentation()
    print(out)
