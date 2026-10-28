import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)
from torchvision import transforms


# ============================================================
# FORENSIGHT-AI — MODEL V2 TRAINING
# ============================================================

DATASET_DIR = r"C:\Users\manud\Downloads\archive"

TRAIN_FAKE_DIR = os.path.join(DATASET_DIR, "train", "FAKE")
TRAIN_REAL_DIR = os.path.join(DATASET_DIR, "train", "REAL")

OUTPUT_DIR = r"C:\Users\manud\OneDrive\Desktop\Forensic Science\codes\forensics_ai_detector_v2"

MODEL_NAME = "google/mobilenet_v2_1.0_224"

# 20,000 images from each class
MAX_PER_CLASS = 20000

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)


print("=" * 70)
print("FORENSIGHT-AI — CPU TRAINING V2")
print("=" * 70)

print(f"\nDataset: {DATASET_DIR}")
print(f"Output : {OUTPUT_DIR}")
print(f"Model  : {MODEL_NAME}")
print("Device : CPU")
print(f"\nImages per class: {MAX_PER_CLASS}")


# ============================================================
# COLLECT IMAGES
# ============================================================

def get_images(folder):
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    files = []

    for root, _, filenames in os.walk(folder):
        for filename in filenames:
            if filename.lower().endswith(valid_extensions):
                files.append(os.path.join(root, filename))

    return files


print("\nScanning dataset...")

fake_files = get_images(TRAIN_FAKE_DIR)
real_files = get_images(TRAIN_REAL_DIR)

print(f"\nFAKE images available: {len(fake_files)}")
print(f"REAL images available: {len(real_files)}")


if len(fake_files) < MAX_PER_CLASS:
    raise ValueError("Not enough FAKE images.")

if len(real_files) < MAX_PER_CLASS:
    raise ValueError("Not enough REAL images.")


# Reproducible random selection
random.shuffle(fake_files)
random.shuffle(real_files)

fake_files = fake_files[:MAX_PER_CLASS]
real_files = real_files[:MAX_PER_CLASS]

print("\nTraining images selected:")
print(f"FAKE: {len(fake_files)}")
print(f"REAL: {len(real_files)}")
print(f"TOTAL: {len(fake_files) + len(real_files)}")


# ============================================================
# CREATE DATASET
# ============================================================

all_files = []
all_labels = []

# 0 = human
# 1 = AI-generated

for f in real_files:
    all_files.append(f)
    all_labels.append(0)

for f in fake_files:
    all_files.append(f)
    all_labels.append(1)


combined = list(zip(all_files, all_labels))
random.shuffle(combined)

all_files, all_labels = zip(*combined)

all_files = list(all_files)
all_labels = list(all_labels)


# ============================================================
# PROCESSOR
# ============================================================

print("\nLoading image processor...")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

print("Image processor loaded.")


# ============================================================
# AUGMENTATION
# ============================================================

mean = processor.image_mean
std = processor.image_std

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(
        degrees=8
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10,
        hue=0.03
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=mean,
        std=std
    ),
])


# ============================================================
# DATASET CLASS
# ============================================================

class ForensicDataset(Dataset):

    def __init__(self, files, labels, transform):
        self.files = files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        path = self.files[idx]
        label = self.labels[idx]

        try:
            image = Image.open(path).convert("RGB")
            image = self.transform(image)

        except Exception as e:
            print(f"\nWarning: Could not read {path}")
            print(e)

            # fallback image
            image = Image.new(
                "RGB",
                (224, 224),
                "black"
            )

            image = self.transform(image)

        return {
            "pixel_values": image,
            "labels": torch.tensor(label, dtype=torch.long)
        }


dataset = ForensicDataset(
    all_files,
    all_labels,
    train_transform
)


# ============================================================
# MODEL
# ============================================================

print("\nLoading pretrained MobileNetV2...")

id2label = {
    0: "human",
    1: "AI-generated"
}

label2id = {
    "human": 0,
    "AI-generated": 1
}

model = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

print("\nModel loaded.")
print(f"Labels: {model.config.id2label}")


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred):

    import numpy as np

    predictions, labels = eval_pred

    predictions = np.argmax(predictions, axis=1)

    accuracy = np.mean(predictions == labels)

    return {
        "accuracy": float(accuracy)
    }


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    num_train_epochs=2,

    per_device_train_batch_size=8,

    learning_rate=2e-5,

    weight_decay=0.01,

    logging_steps=100,

    save_strategy="epoch",

    save_total_limit=2,

    remove_unused_columns=False,

    report_to="none",

    dataloader_num_workers=0,

    fp16=False,

    bf16=False,
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=dataset,

    compute_metrics=compute_metrics,
)


# ============================================================
# START TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STARTING V2 TRAINING")
print("=" * 70)

print(f"\nTraining on {len(dataset):,} images.")

print("\nThis will take considerably longer than V1.")
print("Do not close PowerShell while training.\n")


trainer.train()


# ============================================================
# SAVE MODEL
# ============================================================

print("\n" + "=" * 70)
print("SAVING V2 MODEL")
print("=" * 70)

trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)


print("\n" + "=" * 70)
print("V2 TRAINING COMPLETE")
print("=" * 70)

print(f"\nModel saved to:")
print(OUTPUT_DIR)

print("\nLabels:")
print(id2label)

print("\nIMPORTANT:")
print("Evaluate V2 on the untouched test dataset before using it.")