import sys
import os
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


# ============================================================
# MODEL PATHS
# ============================================================

CAPCHECK_DIR = (
    r"C:\Users\manud\.cache\huggingface\hub"
    r"\models--capcheck--ai-human-generated-image-detection"
    r"\snapshots\67eacdf4e1953353f8d2c0ca22fde0f3226f1049"
)

DELPOT_DIR = (
    r"C:\Users\manud\.cache\huggingface\hub"
    r"\models--delpot--steganograph-ia-detector"
    r"\snapshots\b395557b96de82e5aaf97206054af416d657655a"
)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_detector(model_dir):
    processor = AutoImageProcessor.from_pretrained(
        model_dir,
        local_files_only=True
    )

    model = AutoModelForImageClassification.from_pretrained(
        model_dir,
        local_files_only=True
    )

    model.to(DEVICE)
    model.eval()

    return processor, model


print()
print("=" * 70)
print("FORENSIGHT-AI — MODEL COMPARISON")
print("=" * 70)

if len(sys.argv) < 2:
    print()
    print("ERROR: No image supplied.")
    print()
    print("Usage:")
    print('python compare_detectors.py "C:\\path\\to\\image.jpg"')
    print()
    sys.exit(1)


image_path = sys.argv[1]

if not os.path.isfile(image_path):
    print()
    print("ERROR: Image file does not exist.")
    print()
    print(image_path)
    print()
    sys.exit(1)


print()
print("Image:")
print(os.path.abspath(image_path))

print()
print("Device:", DEVICE)

print()
print("Loading CAPCHECK...")
cap_processor, cap_model = load_detector(CAPCHECK_DIR)

print("CAPCHECK labels:")
print(cap_model.config.id2label)

print()
print("Loading DELPOT...")
del_processor, del_model = load_detector(DELPOT_DIR)

print("DELPOT labels:")
print(del_model.config.id2label)


# ============================================================
# OPEN IMAGE
# ============================================================

image = Image.open(image_path).convert("RGB")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(processor, model, ai_labels):
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

    labels = model.config.id2label

    ai_index = None
    real_index = None

    for index, label in labels.items():

        label_lower = str(label).lower()

        if any(word in label_lower for word in ai_labels):
            ai_index = int(index)

        if any(word in label_lower for word in [
            "real",
            "human"
        ]):
            real_index = int(index)

    if ai_index is None or real_index is None:
        raise RuntimeError(
            f"Could not determine labels: {labels}"
        )

    ai_probability = (
        probabilities[ai_index].item() * 100
    )

    real_probability = (
        probabilities[real_index].item() * 100
    )

    predicted_index = int(
        torch.argmax(probabilities).item()
    )

    predicted_label = labels[predicted_index]

    return {
        "ai_probability": ai_probability,
        "real_probability": real_probability,
        "predicted_label": predicted_label
    }


# ============================================================
# RUN CAPCHECK
# ============================================================

print()
print("=" * 70)
print("CAPCHECK RESULT")
print("=" * 70)

cap_result = predict(
    cap_processor,
    cap_model,
    [
        "ai",
        "fake",
        "generated"
    ]
)

print()
print(
    f"AI probability:    "
    f"{cap_result['ai_probability']:.2f}%"
)

print(
    f"Human probability: "
    f"{cap_result['real_probability']:.2f}%"
)

print(
    f"Predicted class:   "
    f"{cap_result['predicted_label']}"
)


# ============================================================
# RUN DELPOT
# ============================================================

print()
print("=" * 70)
print("DELPOT RESULT")
print("=" * 70)

del_result = predict(
    del_processor,
    del_model,
    [
        "ai",
        "generated"
    ]
)

print()
print(
    f"AI probability:    "
    f"{del_result['ai_probability']:.2f}%"
)

print(
    f"Real probability:  "
    f"{del_result['real_probability']:.2f}%"
)

print(
    f"Predicted class:   "
    f"{del_result['predicted_label']}"
)


# ============================================================
# COMPARISON
# ============================================================

print()
print("=" * 70)
print("COMPARISON")
print("=" * 70)

cap_ai = cap_result["ai_probability"]
del_ai = del_result["ai_probability"]

average_ai = (
    cap_ai + del_ai
) / 2

print()
print(f"CAPCHECK AI score : {cap_ai:.2f}%")
print(f"DELPOT AI score   : {del_ai:.2f}%")
print(f"Average AI score  : {average_ai:.2f}%")

print()

difference = abs(cap_ai - del_ai)

print(
    f"Model difference  : {difference:.2f} percentage points"
)

print()

if cap_ai >= 50 and del_ai >= 50:

    print("Both models indicate an AI-generation signal.")

elif cap_ai < 50 and del_ai < 50:

    print("Both models indicate a stronger real/human signal.")

else:

    print(
        "The models disagree. "
        "Treat the result as INCONCLUSIVE."
    )

print()
print("=" * 70)
print("IMPORTANT")
print("=" * 70)
print()
print(
    "These are screening models. Their probabilities are "
    "not proof of image origin."
)
print()