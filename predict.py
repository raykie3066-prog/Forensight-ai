import sys
import os

from ai_detection import analyze_ai_indicators


# ============================================================
# CHECK ARGUMENT
# ============================================================

if len(sys.argv) < 2:

    print()
    print("FORENSIGHT-AI IMAGE DETECTOR")
    print("=" * 50)
    print()
    print("Usage:")
    print()
    print('python predict.py "path_to_image.jpg"')
    print()

    sys.exit()


image_path = sys.argv[1]


# ============================================================
# CHECK FILE
# ============================================================

if not os.path.exists(image_path):

    print()
    print("ERROR: Image file not found.")
    print()
    print("Path:")
    print(image_path)
    print()

    sys.exit()


# ============================================================
# RUN DETECTOR
# ============================================================

print()
print("=" * 60)
print("FORENSIGHT-AI IMAGE ANALYSIS")
print("=" * 60)
print()

result = analyze_ai_indicators(image_path)


# ============================================================
# ERROR
# ============================================================

if result.get("status") != "success":

    print("Detection failed.")

    for finding in result.get("findings", []):

        print("•", finding)

    sys.exit()


# ============================================================
# RESULTS
# ============================================================

print()
print("RESULT")
print("-" * 60)

print(
    "Prediction:",
    result["prediction"]
)

print(
    f"AI model score: "
    f"{result['ai_probability']:.2f}%"
)

print(
    f"Human-created score: "
    f"{result['human_probability']:.2f}%"
)

print(
    "Model confidence:",
    result["confidence"]
)

print(
    "Detection level:",
    result["level"]
)

print(
    f"Forensic score: "
    f"{result['score']:.2f}"
)


# ============================================================
# FINDINGS
# ============================================================

print()
print("FINDINGS")
print("-" * 60)

for finding in result["findings"]:

    print("•", finding)


# ============================================================
# MODEL
# ============================================================

print()
print("MODEL")
print("-" * 60)

print(result["model"])

print()
print("=" * 60)