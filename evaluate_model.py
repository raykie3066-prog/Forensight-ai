import os
import numpy as np
import torch

from PIL import Image

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = "./forensics_ai_detector_v2"

TEST_DIR = r"C:\Users\manud\Downloads\archive\test"

MAX_TEST_PER_CLASS = None


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("FORENSIGHT-AI MODEL EVALUATION")
print("=" * 60)


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    device = torch.device("cuda")

    print("\nUsing device: CUDA")

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

else:

    device = torch.device("cpu")

    print("\nUsing device: CPU")


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_DIR):

    raise FileNotFoundError(
        "\nTrained model was not found.\n\n"
        f"Expected location:\n"
        f"{os.path.abspath(MODEL_DIR)}\n"
    )


# ============================================================
# CHECK TEST DIRECTORIES
# ============================================================

AI_DIR = os.path.join(
    TEST_DIR,
    "FAKE"
)

REAL_DIR = os.path.join(
    TEST_DIR,
    "REAL"
)


if not os.path.exists(AI_DIR):

    raise FileNotFoundError(
        "\nAI test folder not found:\n"
        f"{os.path.abspath(AI_DIR)}"
    )


if not os.path.exists(REAL_DIR):

    raise FileNotFoundError(
        "\nReal test folder not found:\n"
        f"{os.path.abspath(REAL_DIR)}"
    )


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {

    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"

}


# ============================================================
# FIND IMAGES
# ============================================================

def find_images(folder):

    files = []

    for root, dirs, filenames in os.walk(
        folder
    ):

        for filename in filenames:

            extension = (
                os.path.splitext(
                    filename
                )[1]
                .lower()
            )

            if extension in IMAGE_EXTENSIONS:

                files.append(
                    os.path.join(
                        root,
                        filename
                    )
                )

    return files


# ============================================================
# LOAD IMAGE FILES
# ============================================================

print("\nScanning test dataset...")


ai_files = find_images(
    AI_DIR
)

real_files = find_images(
    REAL_DIR
)


print(
    "\nAI images found:",
    len(ai_files)
)

print(
    "Real images found:",
    len(real_files)
)


# ============================================================
# CHECK DATASET
# ============================================================

if len(ai_files) == 0:

    raise ValueError(
        "No AI images found in test/ai."
    )


if len(real_files) == 0:

    raise ValueError(
        "No real images found in test/real."
    )


# ============================================================
# SHUFFLE
# ============================================================

rng = np.random.default_rng(42)


rng.shuffle(
    ai_files
)

rng.shuffle(
    real_files
)


# ============================================================
# LOAD PROCESSOR
# ============================================================

print(
    "\nLoading image processor..."
)


processor = (
    AutoImageProcessor
    .from_pretrained(
        MODEL_DIR
    )
)


# ============================================================
# LOAD MODEL
# ============================================================

print(
    "Loading trained model..."
)


model = (
    AutoModelForImageClassification
    .from_pretrained(
        MODEL_DIR
    )
)


model.to(device)

model.eval()


print(
    "\nModel loaded successfully."
)


print(
    "Model labels:",
    model.config.id2label
)


# ============================================================
# RESOLVE WHICH CLASS INDEX MEANS "AI" AND WHICH MEANS "REAL"
# ============================================================
# Do NOT assume index 0 = AI and index 1 = Real. Hugging Face
# assigns ids from the training folder names, and that order
# is not guaranteed. Resolve it from the label text instead,
# the same way ai_detection.py does, so evaluation always
# matches what the model actually outputs.

AI_INDEX = None
REAL_INDEX = None

for class_id, label in model.config.id2label.items():

    label_lower = str(label).lower()

    if "ai" in label_lower or "generated" in label_lower or "fake" in label_lower:
        AI_INDEX = int(class_id)

    if "real" in label_lower or "human" in label_lower:
        REAL_INDEX = int(class_id)

if AI_INDEX is None or REAL_INDEX is None:

    raise ValueError(
        "\nCould not resolve AI/Real class indices from "
        f"model labels: {model.config.id2label}\n"
        "Update the label-matching keywords above."
    )

print(
    f"\nResolved class indices -> AI: {AI_INDEX}, "
    f"Real: {REAL_INDEX}"
)


# ============================================================
# SELECT BALANCED TEST SET
# ============================================================

if MAX_TEST_PER_CLASS is not None:
    ai_files = ai_files[:MAX_TEST_PER_CLASS]
    real_files = real_files[:MAX_TEST_PER_CLASS]


test_images = []


for image_path in ai_files:

    test_images.append(
        (
            image_path,
            AI_INDEX
        )
    )


for image_path in real_files:

    test_images.append(
        (
            image_path,
            REAL_INDEX
        )
    )


rng.shuffle(
    test_images
)


print("\nTest set selected:")

print(
    "AI images:",
    len(ai_files)
)

print(
    "Real images:",
    len(real_files)
)

print(
    "Total:",
    len(test_images)
)


# ============================================================
# STORAGE
# ============================================================

true_labels = []

predicted_labels = []

ai_probabilities = []


correct = 0

failed = 0


# ============================================================
# START EVALUATION
# ============================================================

print("\n")

print("=" * 60)

print(
    "STARTING EVALUATION"
)

print("=" * 60)


for index, (
    image_path,
    true_label
) in enumerate(test_images):


    try:

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")


        # ----------------------------------------------------
        # Process image
        # ----------------------------------------------------

        inputs = processor(

            images=image,

            return_tensors="pt"

        )


        inputs = {

            key: value.to(device)

            for key, value
            in inputs.items()

        }


        # ----------------------------------------------------
        # Model inference
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = model(
                **inputs
            )


        # ----------------------------------------------------
        # Probabilities
        # ----------------------------------------------------

        probabilities = torch.softmax(

            outputs.logits,

            dim=-1

        )[0]


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = torch.argmax(
            probabilities
        ).item()


        # ----------------------------------------------------
        # AI probability
        # ----------------------------------------------------

        ai_probability = float(
            probabilities[AI_INDEX]
        )


        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        true_labels.append(
            true_label
        )

        predicted_labels.append(
            prediction
        )

        ai_probabilities.append(
            ai_probability
        )


        if prediction == true_label:

            correct += 1


    except Exception as error:

        failed += 1

        print(
            "\nError processing:"
        )

        print(
            image_path
        )

        print(
            "Error:",
            error
        )


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    processed = index + 1


    if (
        processed % 50 == 0
        or
        processed == len(test_images)
    ):

        print(

            f"Processed "
            f"{processed}/"
            f"{len(test_images)}"

        )


# ============================================================
# CHECK RESULTS
# ============================================================

if len(true_labels) == 0:

    raise RuntimeError(
        "No images were successfully evaluated."
    )


# ============================================================
# CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(

    true_labels,

    predicted_labels

)


precision = precision_score(

    true_labels,

    predicted_labels,

    pos_label=AI_INDEX,

    zero_division=0

)


recall = recall_score(

    true_labels,

    predicted_labels,

    pos_label=AI_INDEX,

    zero_division=0

)


f1 = f1_score(

    true_labels,

    predicted_labels,

    pos_label=AI_INDEX,

    zero_division=0

)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

print("\n")

print("=" * 60)

print(
    "MODEL PERFORMANCE"
)

print("=" * 60)


print(

    f"\nAccuracy : "
    f"{accuracy * 100:.2f}%"

)


print(

    f"Precision: "
    f"{precision * 100:.2f}%"

)


print(

    f"Recall   : "
    f"{recall * 100:.2f}%"

)


print(

    f"F1 Score : "
    f"{f1 * 100:.2f}%"

)


print(

    f"\nSuccessfully evaluated: "
    f"{len(true_labels)}"

)


print(

    f"Failed images: "
    f"{failed}"

)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")

print("=" * 60)

print(
    "CLASSIFICATION REPORT"
)

print("=" * 60)


print(

    classification_report(

        true_labels,

        predicted_labels,

        labels=[AI_INDEX, REAL_INDEX],

        target_names=[
            "AI",
            "Real"
        ],

        zero_division=0

    )

)


# ============================================================
# CONFUSION MATRIX
# ============================================================

matrix = confusion_matrix(

    true_labels,

    predicted_labels,

    labels=[AI_INDEX, REAL_INDEX]

)


print("\n")

print("=" * 60)

print(
    "CONFUSION MATRIX"
)

print("=" * 60)


print(
    "\n                 Predicted"
)

print(
    "                 AI       Real"
)


print(

    f"Actual AI       "
    f"{matrix[0][0]:5d}    "
    f"{matrix[0][1]:5d}"

)


print(

    f"Actual Real     "
    f"{matrix[1][0]:5d}    "
    f"{matrix[1][1]:5d}"

)


# ============================================================
# ERROR ANALYSIS
# ============================================================

false_positives = matrix[1][0]

false_negatives = matrix[0][1]


print("\n")

print("=" * 60)

print(
    "ERROR ANALYSIS"
)

print("=" * 60)


print(

    "\nFalse Positives "
    "(Real → AI):",

    false_positives

)


print(

    "False Negatives "
    "(AI → Real):",

    false_negatives

)


# ============================================================
# INTERPRETATION
# ============================================================

print("\n")

print("=" * 60)

print(
    "INTERPRETATION"
)

print("=" * 60)


if accuracy >= 0.90:

    print(
        "\nExcellent initial accuracy."
    )

elif accuracy >= 0.80:

    print(
        "\nGood initial accuracy, "
        "but further testing is recommended."
    )

elif accuracy >= 0.70:

    print(
        "\nModerate accuracy. "
        "More training/data may be required."
    )

else:

    print(
        "\nLow accuracy. "
        "The model needs further investigation."
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n")

print("=" * 60)

print(
    "EVALUATION COMPLETE"
)

print("=" * 60)