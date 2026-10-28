from datetime import datetime
import os
import tempfile

import cv2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader


def _safe_text(value):
    return str(value).replace("\n", " ")


def create_report(
    filename,
    evidence_id,
    file_name,
    file_size,
    dimensions,
    sha256,
    md5,
    phash,
    ai_info,
    image_info,
    metadata_info,
    source_info,
):
    pdf = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    y = height - 18 * mm

    def heading(text):
        nonlocal y
        if y < 35 * mm:
            pdf.showPage()
            y = height - 18 * mm

        pdf.setFont("Helvetica-Bold", 13)
        pdf.drawString(20 * mm, y, text)
        y -= 7 * mm
        pdf.setFont("Helvetica", 9)

    def line(text, indent=25):
        nonlocal y
        if y < 18 * mm:
            pdf.showPage()
            y = height - 18 * mm
            pdf.setFont("Helvetica", 9)

        pdf.drawString(indent * mm, y, _safe_text(text)[:125])
        y -= 5 * mm

    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(20 * mm, y, "FORENSIGHT-AI")
    y -= 8 * mm
    pdf.setFont("Helvetica", 10)
    pdf.drawString(
        20 * mm,
        y,
        "AI Media Detection & Digital Image Forensic Screening Report",
    )
    y -= 12 * mm

    line(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    line(f"Evidence ID: {evidence_id}")

    heading("1. Evidence Identification")
    line(f"File Name: {file_name}")
    line(f"File Size: {file_size}")
    line(f"Image Dimensions: {dimensions}")
    line(f"SHA-256: {sha256}")
    line(f"MD5: {md5}")
    line(f"Perceptual Hash: {phash}")

    heading("2. AI Media Screening")

    if ai_info.get("status") == "success":
        line(f"Assessment: {ai_info.get('assessment', 'N/A')}")
        line(f"Custom model AI score: {ai_info.get('ai_percentage', 0)}%")
        line(f"Custom model human score: {ai_info.get('human_percentage', 0)}%")
        line(f"Confidence: {ai_info.get('confidence', 'N/A')}")
        if ai_info.get("secondary_ai_percentage") is not None:
            line(
                "Independent cross-check AI score: "
                f"{ai_info['secondary_ai_percentage']}%"
            )
            line(
                "Independent cross-check human score: "
                f"{ai_info['secondary_human_percentage']}%"
            )

        for finding in ai_info.get("findings", []):
            line("• " + finding)
    else:
        line("AI screening unavailable.")
        for finding in ai_info.get("findings", []):
            line("• " + finding)

    heading("3. Image Forensic Screening")
    line(f"ELA Score: {image_info.get('ela_score', 'N/A')}")
    line(f"Noise Score: {image_info.get('noise_score', 'N/A')}")
    line(f"Edge Density: {image_info.get('edge_density', 'N/A')}%")
    line(
        f"Image screening indicator: "
        f"{image_info.get('image_score', 'N/A')}/50"
    )

    heading("4. Facial Feature Analysis")
    measurements = image_info.get("facial_measurements", [])

    if measurements:
        for face in measurements:
            line(f"Face {face['Face']}")
            line(
                f"Interpupillary Distance: "
                f"{face['Interpupillary Distance']} px",
                30,
            )
            line(f"Nose Width: {face['Nose Width']} px", 30)
            line(f"Nose Length: {face['Nose Length']} px", 30)
            line(f"Mouth Width: {face['Mouth Width']} px", 30)
            line(f"Left Eye Width: {face['Left Eye Width']} px", 30)
            line(f"Right Eye Width: {face['Right Eye Width']} px", 30)
            line(f"Face Height: {face['Face Height']} px", 30)

        line("Measurements are image-space distances in pixels.")
    else:
        line("No human face detected.")

    annotated = image_info.get("facial_image")

    if annotated is not None:
        try:
            if y < 105 * mm:
                pdf.showPage()
                y = height - 18 * mm

            pdf.setFont("Helvetica-Bold", 11)
            pdf.drawString(
                20 * mm,
                y,
                "Annotated Facial Measurement Image",
            )
            y -= 6 * mm

            fd, temp_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            cv2.imwrite(temp_path, annotated)

            h, w = annotated.shape[:2]
            display_width = 165 * mm
            display_height = display_width * h / w
            max_height = 95 * mm

            if display_height > max_height:
                display_height = max_height
                display_width = display_height * w / h

            pdf.drawImage(
                ImageReader(temp_path),
                20 * mm,
                y - display_height,
                width=display_width,
                height=display_height,
                preserveAspectRatio=True,
            )
            y -= display_height + 10 * mm

            try:
                os.remove(temp_path)
            except Exception:
                pass
        except Exception:
            pass

    heading("5. Metadata Examination")

    metadata = metadata_info.get("metadata", {})
    if metadata:
        preferred = [
            ("Camera", "Image Make"),
            ("Camera Model", "Image Model"),
            ("Software", "Image Software"),
            ("Date Taken", "EXIF DateTimeOriginal"),
            ("GPS", "GPS GPSLatitude"),
        ]

        shown = set()

        for title, key in preferred:
            if key in metadata:
                line(f"{title}: {metadata[key]}")
                shown.add(key)

        for key, value in list(metadata.items())[:25]:
            if key not in shown:
                line(f"{key}: {value}")

    else:
        line("No readable EXIF metadata detected.")

    heading("6. Source Tracing")
    line(f"Perceptual hash: {source_info.get('phash', 'N/A')}")

    matches = source_info.get("matches", [])
    if matches:
        for i, match in enumerate(matches[:5], start=1):
            line(
                f"{i}. {match['filename']} — "
                f"{match['similarity']}% pHash similarity"
            )
    else:
        line("No visually similar image found in the local reference database.")

    heading("7. Interpretation & Limitations")
    line(
        "All outputs are automated screening indicators and require "
        "expert interpretation."
    )
    line(
        "The AI detector does not establish authenticity, authorship, "
        "fraud, or manipulation conclusively."
    )
    line(
        "Facial measurements are image-space distances and are not "
        "actual anatomical measurements."
    )
    line(
        "Source tracing checks only the configured local reference "
        "database; it is not an internet reverse-image search."
    )

    pdf.setFont("Helvetica", 8)
    pdf.drawString(
        20 * mm,
        10 * mm,
        "FORENSIGHT-AI — automated forensic screening prototype",
    )

    pdf.save()
    return filename


# Compatibility alias.
generate_report = create_report
