import cv2
import numpy as np

# Plain import. If it fails, the real reason is kept and shown in the app.
_MP_ERROR = None
try:
    from mediapipe.python.solutions import face_mesh as mp_face_mesh
    from mediapipe.python.solutions import face_detection as mp_face_detection
except Exception as _exc:
    mp_face_mesh = None
    mp_face_detection = None
    _MP_ERROR = f"{type(_exc).__name__}: {_exc}"


def _distance(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _midpoint(a, b):
    return (int((a[0] + b[0]) / 2), int((a[1] + b[1]) / 2))


def _draw_measurement(image, p1, p2, label):
    color = (0, 255, 255)

    cv2.line(image, p1, p2, color, 2)
    cv2.circle(image, p1, 4, color, -1)
    cv2.circle(image, p2, 4, color, -1)

    center = _midpoint(p1, p2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

    x = max(5, min(center[0] - tw // 2, image.shape[1] - tw - 8))
    y = max(th + 8, center[1] - 7)

    cv2.rectangle(image, (x - 3, y - th - 3), (x + tw + 3, y + 3), (0, 0, 0), -1)
    cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _mesh_points(rgb, mesh):
    """Run face mesh on an RGB image; return a list of faces, each a list of (x, y) pixels."""
    res = mesh.process(rgb)
    h, w = rgb.shape[:2]
    faces = []
    for fl in (res.multi_face_landmarks or []):
        faces.append([(lm.x * w, lm.y * h) for lm in fl.landmark])
    return faces


def _get_all_face_points(image, max_faces=10):
    """
    Detect faces with the full-range detector (better for small / distant faces),
    then run Face Mesh on a padded, upscaled crop of each face.
    Falls back to a whole-image Face Mesh pass if nothing was found.
    Returns a list of faces; each face is a list of 478 (x, y) pixel points.
    """
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    all_pts = []

    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.3
    ) as detector, mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
    ) as mesh:

        detections = detector.process(rgb).detections or []
        detections = sorted(
            detections, key=lambda d: d.score[0] if d.score else 0, reverse=True
        )[:max_faces]

        for d in detections:
            bb = d.location_data.relative_bounding_box
            bx, by = bb.xmin * w, bb.ymin * h
            bw, bh = bb.width * w, bb.height * h

            pad = 0.4
            x1 = int(max(0, bx - bw * pad))
            y1 = int(max(0, by - bh * pad))
            x2 = int(min(w, bx + bw * (1 + pad)))
            y2 = int(min(h, by + bh * (1 + pad)))

            crop = rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            ch, cw = crop.shape[:2]
            scale = max(1.0, 320 / max(ch, cw))
            if scale > 1.0:
                crop = cv2.resize(
                    crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
                )
            sx = crop.shape[1] / cw
            sy = crop.shape[0] / ch

            found = _mesh_points(crop, mesh)
            if found:
                all_pts.append(
                    [(x1 + px / sx, y1 + py / sy) for px, py in found[0]]
                )

    # Fallback: whole-image pass
    if not all_pts:
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.3,
        ) as mesh_full:
            all_pts = _mesh_points(rgb, mesh_full)

    # Number faces left to right
    all_pts.sort(key=lambda face: min(p[0] for p in face))
    return all_pts


def analyze_faces(image_path, max_faces=10):
    image = cv2.imread(image_path)

    if image is None:
        return {
            "face_detected": False,
            "face_count": 0,
            "annotated_image": None,
            "measurements": [],
            "message": "Unable to read image.",
        }

    if mp_face_mesh is None or mp_face_detection is None:
        return {
            "face_detected": False,
            "face_count": 0,
            "annotated_image": image,
            "measurements": [],
            "message": f"MediaPipe failed to load: {_MP_ERROR}",
        }

    h, w = image.shape[:2]

    try:
        import os as _os
        _os.environ.setdefault("GLOG_minloglevel", "3")
        _os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

        all_pts = _get_all_face_points(image, max_faces)
    except Exception as exc:
        return {
            "face_detected": False,
            "face_count": 0,
            "annotated_image": image,
            "measurements": [],
            "message": f"Facial analysis error: {exc}",
        }

    if not all_pts:
        return {
            "face_detected": False,
            "face_count": 0,
            "annotated_image": image,
            "measurements": [],
            "message": "No human faces detected.",
        }

    annotated = image.copy()
    measurements = []

    LEFT_EYE_OUTER = 33
    LEFT_EYE_INNER = 133
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263
    LEFT_IRIS_CENTER = 468
    RIGHT_IRIS_CENTER = 473
    NOSE_ROOT = 168
    NOSE_TIP = 1
    LEFT_NOSTRIL = 98
    RIGHT_NOSTRIL = 327
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    LEFT_EYEBROW = 70
    RIGHT_EYEBROW = 300
    CHIN = 152

    for face_number, raw in enumerate(all_pts, start=1):
        if len(raw) <= RIGHT_IRIS_CENTER:
            continue  # iris landmarks missing, skip this face

        pts = [
            (max(0, min(w - 1, int(x))), max(0, min(h - 1, int(y))))
            for x, y in raw
        ]

        left_pupil = pts[LEFT_IRIS_CENTER]
        right_pupil = pts[RIGHT_IRIS_CENTER]
        left_eye_outer = pts[LEFT_EYE_OUTER]
        left_eye_inner = pts[LEFT_EYE_INNER]
        right_eye_inner = pts[RIGHT_EYE_INNER]
        right_eye_outer = pts[RIGHT_EYE_OUTER]
        nose_root = pts[NOSE_ROOT]
        nose_tip = pts[NOSE_TIP]
        left_nostril = pts[LEFT_NOSTRIL]
        right_nostril = pts[RIGHT_NOSTRIL]
        mouth_left = pts[MOUTH_LEFT]
        mouth_right = pts[MOUTH_RIGHT]
        left_eyebrow = pts[LEFT_EYEBROW]
        right_eyebrow = pts[RIGHT_EYEBROW]
        chin = pts[CHIN]

        ipd = _distance(left_pupil, right_pupil)
        nose_width = _distance(left_nostril, right_nostril)
        nose_length = _distance(nose_root, nose_tip)
        mouth_width = _distance(mouth_left, mouth_right)
        left_eye_width = _distance(left_eye_outer, left_eye_inner)
        right_eye_width = _distance(right_eye_inner, right_eye_outer)
        eyebrow_center = _midpoint(left_eyebrow, right_eyebrow)
        face_height = _distance(eyebrow_center, chin)

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1 = max(0, min(xs) - 10)
        y1 = max(0, min(ys) - 10)
        x2 = min(w - 1, max(xs) + 10)
        y2 = min(h - 1, max(ys) + 10)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated, f"FACE {face_number}",
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA,
        )

        for p in [left_pupil, right_pupil, left_nostril, right_nostril,
                  nose_root, nose_tip, mouth_left, mouth_right,
                  left_eye_outer, left_eye_inner, right_eye_inner, right_eye_outer,
                  eyebrow_center, chin]:
            cv2.circle(annotated, p, 3, (0, 255, 0), -1)

        _draw_measurement(annotated, left_pupil, right_pupil, f"IPD: {ipd:.1f}px")
        _draw_measurement(annotated, left_nostril, right_nostril, f"Nose W: {nose_width:.1f}px")
        _draw_measurement(annotated, nose_root, nose_tip, f"Nose L: {nose_length:.1f}px")
        _draw_measurement(annotated, mouth_left, mouth_right, f"Mouth W: {mouth_width:.1f}px")
        _draw_measurement(annotated, left_eye_outer, left_eye_inner, f"Eye L: {left_eye_width:.1f}px")
        _draw_measurement(annotated, right_eye_inner, right_eye_outer, f"Eye R: {right_eye_width:.1f}px")
        _draw_measurement(annotated, eyebrow_center, chin, f"Face H: {face_height:.1f}px")

        measurements.append({
            "Face": face_number,
            "Interpupillary Distance": round(ipd, 2),
            "Nose Width": round(nose_width, 2),
            "Nose Length": round(nose_length, 2),
            "Mouth Width": round(mouth_width, 2),
            "Left Eye Width": round(left_eye_width, 2),
            "Right Eye Width": round(right_eye_width, 2),
            "Face Height": round(face_height, 2),
        })

    if not measurements:
        return {
            "face_detected": False,
            "face_count": 0,
            "annotated_image": image,
            "measurements": [],
            "message": "Faces were found but landmarks were incomplete.",
        }

    cv2.rectangle(annotated, (10, 10), (540, 55), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        f"FORENSIC FACIAL ANALYSIS - {len(measurements)} FACE(S)",
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA,
    )

    return {
        "face_detected": True,
        "face_count": len(measurements),
        "annotated_image": annotated,
        "measurements": measurements,
        "message": f"{len(measurements)} human face(s) detected.",
    }