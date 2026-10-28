import os
import uuid

import cv2
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from ai_detection import analyze_ai_indicators
from image_analysis import analyze_image
from metadata import extract_metadata
from hashing import calculate_sha256, calculate_md5, calculate_phash
from source_tracing import trace_source
from report_generator import create_report


# ─────────────────────────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FORENSIGHT-AI",
    page_icon="🔎",
    layout="wide",
)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR  = os.path.join(BASE_DIR, "uploads")
REPORT_DIR  = os.path.join(BASE_DIR, "reports")
REFERENCE_DIR = os.path.join(BASE_DIR, "reference_images")

os.makedirs(UPLOAD_DIR,    exist_ok=True)
os.makedirs(REPORT_DIR,    exist_ok=True)
os.makedirs(REFERENCE_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("🔎 FORENSIGHT-AI")
st.caption(
    "AI-generated image screening  ·  Digital image forensics  ·  "
    "Facial measurement  ·  Evidence reporting"
)

with st.sidebar:
    st.header("⚙️ Settings")
    st.info(
        "AI detection uses a pretrained ViT classifier "
        "(umm-maybe/AI-image-detector).  "
        "Results are screening indicators — not legal proof of origin."
    )
    st.divider()
    st.caption("FORENSIGHT-AI  v1.0")


# ─────────────────────────────────────────────────────────────────────────────
#  Upload
# ─────────────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload image evidence",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
)

if not uploaded:
    st.markdown("""
### How it works

1. **Upload** any image (photo or AI-generated).
2. **Evidence hashes** (SHA-256, MD5, perceptual) are calculated.
3. **AI screening** classifies the image as AI-generated, human-created, or inconclusive.
4. **Image forensics** — Error Level Analysis, noise map, edge density.
5. **Facial analysis** — MediaPipe landmark detection with pixel measurements.
6. **Metadata** — EXIF fields are extracted and displayed.
7. **Source tracing** — perceptual hash compared against local reference images.
8. **PDF report** is generated and available for download.

> Upload a known AI-generated image (e.g. Gemini / DALL-E output) and a real photograph to see the difference.
""")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
#  Save & hash
# ─────────────────────────────────────────────────────────────────────────────
safe_name   = os.path.basename(uploaded.name)
evidence_id = "FS-" + uuid.uuid4().hex[:8].upper()
saved_path  = os.path.join(UPLOAD_DIR, f"{evidence_id}_{safe_name}")

with open(saved_path, "wb") as f:
    f.write(uploaded.getbuffer())

pil_image  = Image.open(saved_path).convert("RGB")
width, height = pil_image.size
file_size  = os.path.getsize(saved_path)

sha256 = calculate_sha256(saved_path)
md5    = calculate_md5(saved_path)
phash  = calculate_phash(saved_path)


# ─────────────────────────────────────────────────────────────────────────────
#  Analysis
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Running FORENSIGHT-AI analysis…"):
    ai_result       = analyze_ai_indicators(saved_path)
    image_result    = analyze_image(saved_path)
    metadata_result = extract_metadata(saved_path)
    source_result   = trace_source(saved_path, reference_dir=REFERENCE_DIR)

dimensions     = f"{width} × {height} px"
file_size_text = f"{file_size / 1024:.2f} KB"

report_path = os.path.join(
    REPORT_DIR,
    f"{evidence_id}_FORENSIGHT_Report.pdf",
)

create_report(
    filename=report_path,
    evidence_id=evidence_id,
    file_name=safe_name,
    file_size=file_size_text,
    dimensions=dimensions,
    sha256=sha256,
    md5=md5,
    phash=phash,
    ai_info=ai_result,
    image_info=image_result,
    metadata_info=metadata_result,
    source_info=source_result,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: annotate image with dimension rulers
# ─────────────────────────────────────────────────────────────────────────────
def _annotate_dimensions(img: Image.Image) -> Image.Image:
    w, h = img.size
    out  = img.copy().convert("RGBA")
    draw = ImageDraw.Draw(out)

    font_size = max(14, min(w, h) // 30)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    margin   = max(6,  font_size // 2)
    tick     = max(4,  font_size // 3)
    lw       = max(1,  font_size // 10)
    color    = (255, 80, 0)
    bg       = (20, 20, 20)

    # Width ruler — top edge
    y_w = margin + font_size + tick + 2
    draw.line([(0, y_w), (w - 1, y_w)], fill=color, width=lw)
    draw.line([(0, y_w - tick), (0, y_w + tick)], fill=color, width=lw)
    draw.line([(w - 1, y_w - tick), (w - 1, y_w + tick)], fill=color, width=lw)
    w_label = f"W: {w} px"
    bb = draw.textbbox((0, 0), w_label, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    tx, ty = (w - tw) // 2, y_w - th - tick - 2
    draw.rectangle([tx - 3, ty - 2, tx + tw + 3, ty + th + 2], fill=bg)
    draw.text((tx, ty), w_label, fill=color, font=font)

    # Height ruler — left edge
    x_h = margin + font_size + tick + 2
    draw.line([(x_h, 0), (x_h, h - 1)], fill=color, width=lw)
    draw.line([(x_h - tick, 0), (x_h + tick, 0)], fill=color, width=lw)
    draw.line([(x_h - tick, h - 1), (x_h + tick, h - 1)], fill=color, width=lw)
    h_label = f"H: {h} px"
    bb2 = draw.textbbox((0, 0), h_label, font=font)
    tw2, th2 = bb2[2] - bb2[0], bb2[3] - bb2[1]
    lbl = Image.new("RGBA", (tw2 + 6, th2 + 4), (0, 0, 0, 0))
    ld  = ImageDraw.Draw(lbl)
    ld.rectangle([0, 0, tw2 + 5, th2 + 3], fill=bg)
    ld.text((3, 2), h_label, fill=color, font=font)
    lbl = lbl.rotate(90, expand=True)
    tx2 = max(0, x_h - tick - lbl.width - 2)
    ty2 = max(0, min((h - lbl.height) // 2, h - lbl.height))
    out.paste(lbl, (tx2, ty2), lbl)

    return out.convert("RGB")


# ─────────────────────────────────────────────────────────────────────────────
#  Tabs
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📌 Overview",
    "🤖 AI Detection",
    "🖼️ Image Forensics",
    "👤 Facial Analysis",
    "🔐 Evidence & Metadata",
    "🌐 Source Tracing",
    "📄 Report",
])


# ── Tab 0 · Overview ──────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Evidence")
    c1, c2 = st.columns([1.3, 1])

    with c1:
        st.image(
            _annotate_dimensions(pil_image),
            caption=f"{safe_name}  —  {width} × {height} px",
            use_container_width=True,
        )

    with c2:
        st.markdown(f"**Evidence ID:** `{evidence_id}`")
        st.markdown(f"**File:** {safe_name}")
        st.markdown(f"**Dimensions:** {dimensions}")
        st.markdown(f"**File size:** {file_size_text}")
        st.write("**SHA-256**")
        st.code(sha256, language=None)

    st.divider()
    st.subheader("Screening Assessment")

    assessment = ai_result.get("assessment", "UNAVAILABLE")
    ai_pct     = ai_result.get("ai_percentage", 0)
    hum_pct    = ai_result.get("human_percentage", 0)

    if ai_result.get("status") == "success":
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("AI score",    f"{ai_pct:.1f}%")
        col_b.metric("Human score", f"{hum_pct:.1f}%")
        col_c.metric("Verdict",     assessment)

        st.progress(min(max(ai_pct / 100, 0.0), 1.0))

        if assessment == "AI-GENERATED SIGNAL":
            st.warning(f"⚠️  **AI-GENERATED SIGNAL** — detector confidence {ai_pct:.1f}% AI")
        elif assessment == "HUMAN-CREATED SIGNAL":
            st.success(f"✅  **HUMAN-CREATED SIGNAL** — detector confidence {hum_pct:.1f}% human")
        else:
            st.info("⚪  **INCONCLUSIVE** — score falls in the 30–70 % uncertainty band")
    else:
        st.error("AI screening unavailable.")
        for f in ai_result.get("findings", []):
            st.write("•", f)


# ── Tab 1 · AI Detection ──────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("🤖 AI Media Screening")

    if ai_result.get("status") == "success":
        col_a, col_b = st.columns(2)
        col_a.metric("AI Score",    f"{ai_pct:.2f}%")
        col_b.metric("Human Score", f"{hum_pct:.2f}%")

        st.progress(min(max(ai_pct / 100, 0.0), 1.0),
                    text=f"AI probability: {ai_pct:.1f}%")

        st.markdown(f"**Verdict:** {assessment}")
        st.markdown(f"**Confidence:** {ai_result.get('confidence', '')}")
        st.markdown(f"**Risk level:** {ai_result.get('risk_level', '')}")
        st.markdown(f"**Model:** `{ai_result.get('model', '')}`")

        st.divider()
        st.subheader("Findings")
        for finding in ai_result.get("findings", []):
            st.write("•", finding)

        st.caption(
            "Percentages are model outputs, not calibrated forensic probabilities.  "
            "The 30–70 % band is treated as inconclusive."
        )
    else:
        st.error("AI detector error — check that the model is accessible.")
        for finding in ai_result.get("findings", []):
            st.write("•", finding)


# ── Tab 2 · Image Forensics ───────────────────────────────────────────────────
with tabs[2]:
    st.subheader("🖼️ Image Forensic Analysis")

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("ELA Score",      image_result["ela_score"])
    col_b.metric("Noise Score",    image_result["noise_score"])
    col_c.metric("Edge Density",   f"{image_result['edge_density']}%")
    col_d.metric("Indicator",      f"{image_result['image_score']} / 50")

    if image_result.get("ela_image") is not None:
        st.subheader("Error Level Analysis")
        st.image(
            image_result["ela_image"],
            caption="ELA map — bright areas indicate higher re-compression error",
            use_container_width=True,
        )

    st.subheader("Score Breakdown")
    st.write(f"ELA contribution: {image_result['ela_indicator']} / 25")
    st.progress(min(image_result["ela_indicator"] / 25, 1.0))

    st.write(f"Noise contribution: {image_result['noise_indicator']} / 15")
    st.progress(min(image_result["noise_indicator"] / 15, 1.0))

    st.write(f"Edge contribution: {image_result['edge_indicator']} / 10")
    st.progress(min(image_result["edge_indicator"] / 10, 1.0))


# ── Tab 3 · Facial Analysis ───────────────────────────────────────────────────
with tabs[3]:
    st.subheader("👤 Facial Feature Analysis")

    face_count = image_result["face_count"]
    st.metric("Faces Detected", face_count)

    if face_count > 0:
        annotated = image_result["facial_image"]
        if annotated is not None:
            st.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                caption="Facial landmarks and pixel measurements",
                use_container_width=True,
            )

        st.subheader("Pixel Measurements")
        for face in image_result["facial_measurements"]:
            st.markdown(f"#### Face {face['Face']}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Interpupillary Distance", f"{face['Interpupillary Distance']} px")
                st.metric("Nose Width",              f"{face['Nose Width']} px")
                st.metric("Nose Length",             f"{face['Nose Length']} px")
            with col2:
                st.metric("Mouth Width",      f"{face['Mouth Width']} px")
                st.metric("Left Eye Width",   f"{face['Left Eye Width']} px")
                st.metric("Right Eye Width",  f"{face['Right Eye Width']} px")
            with col3:
                st.metric("Face Height", f"{face['Face Height']} px")
            st.divider()

        st.caption(
            "Measurements are image-space pixel distances.  "
            "They are not real-world anatomical dimensions."
        )
    else:
        st.info(image_result["facial_message"])


# ── Tab 4 · Evidence & Metadata ───────────────────────────────────────────────
with tabs[4]:
    st.subheader("🔐 Evidence Hashes")

    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**SHA-256**");  st.code(sha256, language=None)
        st.write("**MD5**");      st.code(md5,    language=None)
    with col_b:
        st.write("**Perceptual Hash (pHash)**")
        st.code(phash, language=None)

    st.divider()
    st.subheader("EXIF Metadata")
    metadata = metadata_result.get("metadata", {})
    if metadata:
        st.json(metadata)
    else:
        st.info("No readable EXIF metadata found in this file.")


# ── Tab 5 · Source Tracing ────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("🌐 Local Source Tracing")
    st.write("**Perceptual hash of uploaded image:**")
    st.code(source_result.get("phash", phash), language=None)

    matches = source_result.get("matches", [])
    if matches:
        st.write(f"**{len(matches)} visually similar reference image(s) found:**")
        for i, match in enumerate(matches, start=1):
            sim = match["similarity"]
            msg = f"#{i}  {match['filename']}  —  {sim}% similarity"
            if sim >= 90:
                st.error(msg)
            elif sim >= 70:
                st.warning(msg)
            else:
                st.info(msg)
    else:
        st.info("No visually similar image found in the local reference database.")

    st.caption(
        "Source tracing compares perceptual hashes against images in the "
        "`reference_images/` folder only — it is not an internet reverse-image search."
    )


# ── Tab 6 · Report ────────────────────────────────────────────────────────────
with tabs[6]:
    st.subheader("📄 FORENSIGHT-AI Forensic Report")
    st.success("Report generated successfully.")

    with open(report_path, "rb") as pdf:
        st.download_button(
            label="⬇️  Download PDF Report",
            data=pdf,
            file_name=os.path.basename(report_path),
            mime="application/pdf",
        )

    st.caption(f"Evidence ID: {evidence_id}  ·  Saved to: {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "FORENSIGHT-AI is a forensic screening prototype.  "
    "All automated results must be corroborated by expert examination and additional evidence."
)
