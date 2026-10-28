import cv2
import numpy as np
import tempfile
import os

from face_analysis import analyze_faces


def error_level_analysis(image_path, quality=90):
    image = cv2.imread(image_path)
    if image is None:
        return 0.0, None

    fd, temp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)

    try:
        cv2.imwrite(
            temp_path,
            image,
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )
        recompressed = cv2.imread(temp_path)

        if recompressed is None:
            return 0.0, None

        difference = cv2.absdiff(image, recompressed)
        ela = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
        display = cv2.normalize(
            ela, None, 0, 255, cv2.NORM_MINMAX
        )
        return float(np.mean(ela)), display
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass


def noise_analysis(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0

    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    residual = cv2.absdiff(image, blurred)
    return float(np.mean(residual))


def edge_analysis(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0

    edges = cv2.Canny(image, 100, 200)
    return float(np.count_nonzero(edges) / edges.size * 100.0)


def analyze_image(image_path):
    ela_score, ela_image = error_level_analysis(image_path)
    noise_score = noise_analysis(image_path)
    edge_density = edge_analysis(image_path)
    face_result = analyze_faces(image_path)

    ela_indicator = min((ela_score / 20.0) * 25.0, 25.0)
    noise_indicator = min((noise_score / 15.0) * 15.0, 15.0)
    edge_indicator = min((edge_density / 30.0) * 10.0, 10.0)

    image_score = min(
        int(ela_indicator + noise_indicator + edge_indicator),
        50,
    )

    return {
        "ela_score": round(ela_score, 2),
        "noise_score": round(noise_score, 2),
        "edge_density": round(edge_density, 2),
        "ela_indicator": round(ela_indicator, 2),
        "noise_indicator": round(noise_indicator, 2),
        "edge_indicator": round(edge_indicator, 2),
        "image_score": image_score,
        "ela_image": ela_image,
        "face_detected": face_result["face_detected"],
        "face_count": face_result["face_count"],
        "facial_image": face_result["annotated_image"],
        "facial_measurements": face_result["measurements"],
        "facial_message": face_result["message"],
    }


# Backwards-compatible name used by older versions of the app.
analyse_image = analyze_image
