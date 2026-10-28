from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification
)


MODEL_DIR = (
    "./forensics_ai_detector"
)


print("=" * 60)
print("FORENSIGHT-AI MODEL CHECK")
print("=" * 60)


print("\nLoading processor...")


processor = (
    AutoImageProcessor
    .from_pretrained(
        MODEL_DIR
    )
)


print(
    "Processor loaded."
)


print("\nLoading model...")


model = (
    AutoModelForImageClassification
    .from_pretrained(
        MODEL_DIR
    )
)


print(
    "Model loaded."
)


print("\nMODEL LABELS:")


print(
    model.config.id2label
)


print("\nLABEL -> ID:")


print(
    model.config.label2id
)


print("\nMODEL CONFIG:")


print(
    model.config
)


print("\nModel check completed.")