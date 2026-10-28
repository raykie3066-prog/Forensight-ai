import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))

print("=" * 70)
print("FORENSIGHT-AI SUBMISSION CHECK")
print("=" * 70)

required = [
    "app.py",
    "ai_detection.py",
    "image_analysis.py",
    "face_analysis.py",
    "metadata.py",
    "hashing.py",
    "source_tracing.py",
    "report_generator.py",
]

for name in required:
    path = os.path.join(BASE, name)
    print(f"{name:25} {'OK' if os.path.exists(path) else 'MISSING'}")

print("\nModel directories:")
for name in ("forensics_ai_detector_v2", "forensics_ai_detector"):
    path = os.path.join(BASE, name)
    print(f"{name:25} {'FOUND' if os.path.isdir(path) else 'not found'}")

print("\nPython:", sys.version)

try:
    import torch
    print("Torch:", torch.__version__)
except Exception as e:
    print("Torch ERROR:", e)

try:
    import transformers
    print("Transformers:", transformers.__version__)
except Exception as e:
    print("Transformers ERROR:", e)

try:
    import mediapipe as mp
    print("MediaPipe:", getattr(mp, "__version__", "unknown"))
    print("mp.solutions:", hasattr(mp, "solutions"))
except Exception as e:
    print("MediaPipe ERROR:", e)

try:
    import cv2
    print("OpenCV:", cv2.__version__)
except Exception as e:
    print("OpenCV ERROR:", e)

try:
    import reportlab
    print("ReportLab: OK")
except Exception as e:
    print("ReportLab ERROR:", e)

print("=" * 70)
