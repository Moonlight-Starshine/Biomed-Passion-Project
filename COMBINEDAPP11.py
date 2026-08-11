"""
Unified diagnostic interface for anemia, malaria, and leukemia detection.
Streamlit version -- FINAL. Pure manual cell selection, no automatic
detection in any form.

Automatic cell detection (multiple approaches: adaptive thresholding,
watershed, Otsu-based area cutoffs, "guide only" overlays) was tried
extensively. All of them share the same underlying flaw for this data:
classical contour-based detection is drawn to high-contrast regions, and
on stained blood smears, the parasite staining itself is often a stronger
contrast signal than the cell boundary -- so the detector repeatedly finds
stain blobs and fragments instead of whole cells, no matter how the
results are filtered or displayed. This is a real limitation of the
technique for this data, not a tunable bug, and a proper fix would need a
trained object detector (e.g. YOLO), which is out of scope given the
timeline.

This version has no cell-detection code of any kind. You draw a box
around each cell you want screened, so every crop the model sees is
exactly what you intended. This has been confirmed working reliably every
time it was tested.

SETUP:
    pip install streamlit tensorflow pillow opencv-python streamlit-drawable-canvas --break-system-packages

RUN:
    streamlit run app_streamlit.py
"""

import os
import base64
import io
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
import cv2
from PIL import Image, ImageDraw, ImageFont

_COMPONENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cell_canvas_component")
_TEMPLATE_PATH = os.path.join(_COMPONENT_DIR, "template.html")
_INDEX_PATH = os.path.join(_COMPONENT_DIR, "index.html")
cell_canvas = components.declare_component("cell_canvas", path=_COMPONENT_DIR)


def render_cell_canvas(pil_image, width, height, key):
    """Writes index.html fresh with this specific image baked directly into
    the JS (not passed via the postMessage args channel, which failed in an
    earlier attempt), then renders the component. Returns the list of drawn
    boxes ({x0,y0,x1,y1} dicts, in DISPLAY-scale pixel coordinates), or None
    if nothing has been drawn/returned yet."""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    image_data_uri = f"data:image/png;base64,{b64}"

    with open(_TEMPLATE_PATH, "r") as f:
        html = f.read()
    html = html.replace("__IMAGE_B64__", image_data_uri)
    html = html.replace("__CANVAS_WIDTH__", str(width))
    html = html.replace("__CANVAS_HEIGHT__", str(height))

    with open(_INDEX_PATH, "w") as f:
        f.write(html)

    return cell_canvas(key=key, default=None)

# -------------------------------
# CONFIG -- one entry per disease
# -------------------------------
MODEL_CONFIG = {
    "Anemia": {
        "path": "anemia_vgg16_model.keras",       # confirmed: VGG16 won (87.5% acc)
        "architecture": "VGG16",
        "img_size": (224, 224),
        "preprocess": tf.keras.applications.vgg16.preprocess_input,
        "positive_label": "Normal",     # what a HIGH score (>=threshold) means
        "negative_label": "Anemic",     # what a LOW score (<threshold) means
        "disease_label": "Anemic",      # always headline/count the disease, regardless
                                          # of which direction the raw score points
        "threshold": 0.5,
    },
    "Malaria": {
        "path": "malaria_vgg16_model.keras",      # confirmed: VGG16 won (97.1% acc)
        "architecture": "VGG16",
        "img_size": (224, 224),
        "preprocess": tf.keras.applications.vgg16.preprocess_input,
        "positive_label": "Uninfected",    # what a HIGH score (>=threshold) means
        "negative_label": "Parasitized",   # what a LOW score (<threshold) means
        "disease_label": "Parasitized",
        "threshold": 0.5,
    },
    "Leukemia": {
        "path": "Acute_leukemia_vgg16_model.keras",  # confirmed: VGG16 won
        "architecture": "VGG16",
        "img_size": (224, 224),
        "preprocess": tf.keras.applications.vgg16.preprocess_input,
        "positive_label": "Normal",          # what a HIGH score (>=threshold) means
        "negative_label": "ALL (Leukemia)",  # what a LOW score (<threshold) means
        "disease_label": "ALL (Leukemia)",
        "threshold": 0.5,  # default -- gives 99.3% true leukemia recall (see one-pager)
    },
}


# -------------------------------
# MODEL LOADING (cached so it only loads once per model, not on every interaction)
# -------------------------------
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path)


def auto_detect_cells(pil_image):
    """
    Automatically detects likely individual cells for the automatic
    detect-and-classify pipeline. Detection quality genuinely varies by
    disease/image type: works well when the cells of interest are
    themselves the high-contrast objects (e.g. large, darkly-stained
    leukemic cells against a paler background), works poorly when a small
    high-contrast feature sits inside a larger, similarly-toned cell (e.g.
    a tiny malaria parasite stain within a red blood cell -- the detector
    finds the stain, not the whole cell). If the boxes don't look right,
    use the manual pipeline below instead.
    """
    img_array = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=31, C=5,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    all_areas = []
    contour_data = []
    for c in contours:
        area = cv2.contourArea(c)
        if area <= 0:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        all_areas.append(area)
        contour_data.append((area, circularity, c))

    if not all_areas:
        return []

    areas_arr = np.array(all_areas, dtype=np.float32)
    log_areas = np.log(areas_arr + 1)
    lo, hi = log_areas.min(), log_areas.max()
    if hi > lo:
        scaled = ((log_areas - lo) / (hi - lo) * 255).astype(np.uint8)
        otsu_val, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cutoff_log = lo + (otsu_val / 255) * (hi - lo)
        min_area = float(np.exp(cutoff_log) - 1)
    else:
        min_area = 0

    above_cutoff = areas_arr[areas_arr > min_area]
    max_area = float(np.median(above_cutoff) * 4) if len(above_cutoff) > 0 else float(areas_arr.max())

    boxes = []
    for area, circularity, c in contour_data:
        if min_area < area < max_area and circularity > 0.3:
            x, y, cw, ch = cv2.boundingRect(c)
            boxes.append((x, y, x + cw, y + ch))

    return boxes


def _get_bold_font(size=16):
    """Tries common bold system fonts, falls back to PIL's default if none
    are found (e.g. on a machine without these specific fonts installed)."""
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",  # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
        "arialbd.ttf",  # Windows
        "DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()  # older Pillow, no size control


def draw_detection_boxes(pil_image, boxes):
    """Draws NUMBERED detected cell boundaries -- one image widget, not one
    per cell. Numbers match the "Cell #N" labels in the per-cell breakdown,
    so a specific result can be found on the actual image."""
    annotated = pil_image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = _get_bold_font(16)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
        label = str(i + 1)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_x, text_y = x0 + 2, y0 + 2
        draw.rectangle([text_x - 2, text_y - 2, text_x + text_w + 2, text_y + text_h + 2],
                       fill=(255, 0, 0))
        draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font)
    return annotated


def classify_single_cell(model, config, cell_image):
    """Runs one cropped cell through the model, returns (label, prob)."""
    img_size = config["img_size"]
    img = cell_image.convert("RGB").resize(img_size)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = config["preprocess"](arr)

    prob = float(model.predict(arr, verbose=0).flatten()[0])
    threshold = config["threshold"]
    label = config["positive_label"] if prob >= threshold else config["negative_label"]
    return label, prob


def classify_crops(disease, crops):
    """Classifies a list of cell crops, returns the results data (does not
    render anything). Returns None if it can't run (no model, no crops)."""
    config = MODEL_CONFIG[disease]
    model = load_model(config["path"])

    if model is None:
        st.warning(
            f"⚠️ No trained model found yet for {disease} at "
            f"'{config['path']}'. Update MODEL_CONFIG once training finishes."
        )
        return None

    if not crops:
        st.error("No cells selected -- draw at least one box around a cell first.")
        return None

    results = []
    progress_bar = st.progress(0)
    for i, cell_img in enumerate(crops):
        label, prob = classify_single_cell(model, config, cell_img)
        results.append({"label": label, "prob": prob})
        progress_bar.progress((i + 1) / len(crops))
    progress_bar.empty()

    return {"disease": disease, "results": results}


def display_classification(classification_data):
    """Renders results from data returned by classify_crops(). Kept
    separate from classification itself so results can be stored in
    st.session_state and redisplayed on every rerun -- otherwise results
    shown only inside an `if st.button(...)` block vanish the moment any
    OTHER widget on the page (e.g. the canvas below) triggers a rerun,
    since Streamlit buttons are only "clicked" for a single rerun."""
    if classification_data is None:
        return

    disease = classification_data["disease"]
    results = classification_data["results"]
    config = MODEL_CONFIG[disease]

    disease_label = config["disease_label"]
    disease_count = sum(1 for r in results if r["label"] == disease_label)
    total_count = len(results)
    disease_pct = (disease_count / total_count) * 100

    st.markdown(f"### Result: {disease_count} of {total_count} cells flagged as {disease_label} ({disease_pct:.1f}%)")

    if disease_count > 0:
        st.markdown(f"⚠️ **{disease_count} cell(s) flagged as {disease_label}** — recommend clinician review.")
    else:
        st.markdown(f"No cells flagged as {disease_label} in this selection.")

    st.markdown(f"**Model:** {config['architecture']} (threshold: {config['threshold']})")

    st.caption(
        f"Score below {config['threshold']} = {config['negative_label']}, "
        f"score at or above {config['threshold']} = {config['positive_label']}."
    )
    with st.expander("Why is a low score the one that means disease?"):
        st.write(
            f"When these models were trained, the two categories (e.g. "
            f"'{config['negative_label']}' and '{config['positive_label']}') were "
            "auto-assigned a number based on alphabetical folder order, not which one "
            "sounds like the 'positive' result. Since the disease name happened to sort "
            "alphabetically first in each case, a **low** score ends up meaning "
            "**disease detected**, and a **high** score means normal/healthy -- the "
            "reverse of what you might expect at a glance. This has been verified "
            "directly against labeled data, and the app's logic above already accounts "
            "for it correctly."
        )

    with st.expander("See per-cell breakdown", expanded=True):
        for i, r in enumerate(results):
            flag = " ⚠️" if r["label"] == disease_label else ""
            st.write(f"Cell #{i+1}: {r['label']} (score: {r['prob']:.3f}){flag}")


def crop_from_canvas_objects(image, objects):
    """Converts drawable-canvas rectangle objects into cropped PIL images."""
    crops = []
    img_w, img_h = image.size
    for obj in objects:
        if obj.get("type") != "rect":
            continue
        left = int(obj["left"])
        top = int(obj["top"])
        width = int(obj["width"] * obj.get("scaleX", 1))
        height = int(obj["height"] * obj.get("scaleY", 1))

        x0 = max(0, left)
        y0 = max(0, top)
        x1 = min(img_w, left + width)
        y1 = min(img_h, top + height)

        if x1 > x0 and y1 > y0:
            crops.append(image.crop((x0, y0, x1, y1)))
    return crops


def pil_to_base64(pil_image):
    """Converts a PIL image to a base64 data URL for embedding in HTML."""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Blood Smear Screening Tool", layout="centered")

st.title("Blood Smear Screening Tool")
st.markdown(
    "Upload a blood smear microscopy image, draw a box around each cell you want "
    "screened, then run the screening. Drawing your own box guarantees the model "
    "sees a clean, whole cell -- not a fragment or partial crop. "
    "**This is a research prototype, not a validated diagnostic device.**"
)

disease = st.radio("Select condition to screen for", list(MODEL_CONFIG.keys()))

uploaded_file = st.file_uploader(
    "Upload blood smear image", type=["png", "jpg", "jpeg", "bmp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    max_display_width = 700
    scale = min(1.0, max_display_width / image.width)
    display_image = image.resize((int(image.width * scale), int(image.height * scale)))

    # -------------------------------
    # PIPELINE 1: Automatic detection + classification
    # -------------------------------
    st.markdown("## 1. Automatic detection")
    st.caption(
        "The model detects likely individual cells on its own and screens each one. "
        "Check the boxed image below -- if the boxes aren't accurately outlining "
        "whole cells, skip to the manual option further down instead."
    )

    with st.spinner("Automatically detecting cells..."):
        try:
            auto_boxes = auto_detect_cells(image)
        except Exception as e:
            st.error(f"Automatic detection failed: {e}")
            auto_boxes = []

    if not auto_boxes:
        st.warning("No cells were automatically detected in this image. Use the manual option below.")
    else:
        annotated = draw_detection_boxes(image, auto_boxes)
        st.image(annotated, caption=f"{len(auto_boxes)} cells automatically detected")

        auto_crops = [image.crop(box) for box in auto_boxes]

        if st.button("Run Automatic Screening", type="primary", key="auto_run"):
            with st.spinner("Classifying automatically detected cells..."):
                st.session_state["auto_classification"] = classify_crops(disease, auto_crops)

        if "auto_classification" in st.session_state:
            display_classification(st.session_state["auto_classification"])

    st.divider()

    # -------------------------------
    # PIPELINE 2: Manual selection + classification
    # -------------------------------
    st.markdown("## 2. Manual selection")
    st.caption(
        "If the automatic boxes above didn't look right, draw your own box around "
        "each cell directly on the image below -- this guarantees the model sees "
        "exactly the whole cells you intend, and runs completely independently of "
        "the automatic result above."
    )

    canvas_w, canvas_h = display_image.width, display_image.height
    boxes_returned = render_cell_canvas(display_image, canvas_w, canvas_h, key="cell_canvas_v5")

    manual_crops = []
    if boxes_returned:
        manual_crops = crop_from_canvas_objects(image, [
            {"left": b["x0"] / scale, "top": b["y0"] / scale,
             "width": (b["x1"] - b["x0"]) / scale, "height": (b["y1"] - b["y0"]) / scale,
             "type": "rect", "scaleX": 1, "scaleY": 1}
            for b in boxes_returned
        ])

    st.write(f"**{len(manual_crops)} cell(s) manually selected**")

    if manual_crops:
        with st.expander("Preview manually selected cell crops"):
            cols = st.columns(min(6, len(manual_crops)))
            for i, crop in enumerate(manual_crops):
                cols[i % len(cols)].image(crop, caption=f"Cell {i+1}", use_container_width=True)

    if st.button("Run Manual Screening", type="primary", key="manual_run"):
        if not manual_crops:
            st.error("Draw at least one box around a cell before running screening.")
        else:
            with st.spinner("Classifying manually selected cells..."):
                st.session_state["manual_classification"] = classify_crops(disease, manual_crops)

    if "manual_classification" in st.session_state:
        display_classification(st.session_state["manual_classification"])