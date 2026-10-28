import os
import torch
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)
from torchvision.transforms import (
    Compose,
    Resize,
    RandomHorizontalFlip,
    ToTensor,
    Normalize,
)

# ============================================================
# PATHS
# ============================================================

DATASET_DIR = r"C:\Users\manud\Downloads\archive"

OUTPUT_DIR = (
    r"C:\Users\manud\OneDrive\Desktop\Forensic Science"
    r"\codes\forensics_ai_detector_test"
)

# ============================================================
# SMALL CPU TEST
# ============================================================

MAX_PER_CLASS = 100

# ============================================================
# DEVICE
# ============================================================

print("=" * 60)
print("FORENSIGHT-AI TRAINING SANITY TEST")
print("=" * 60)

print()
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device: CPU")

# ============================================================
# CHECK DATASET
# ============================================================

fake_dir = os.path.join(
    DATASET_DIR,
    "train",
    "FAKE"
)

real_dir = os.path.join(
    DATASET_DIR,
    "train",
    "REAL"
)

if not os.path.exists(fake_dir):
    raise FileNotFoundError(
        f"FAKE folder not found:\n{fake_dir}"
    )

if not os.path.exists(real_dir):
    raise FileNotFoundError(
        f"REAL folder not found:\n{real_dir}"
    )

print()
print("Dataset found.")
print("FAKE:", fake_dir)
print("REAL:", real_dir)

# ============================================================
# CREATE SMALL DATASET
# ============================================================

print()
print("Collecting sample images...")

extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}


def get_images(folder, limit):

    files = []

    for filename in os.listdir(folder):

        path = os.path.join(
            folder,
            filename
        )

        if not os.path.isfile(path):
            continue

        extension = os.path.splitext(
            filename
        )[1].lower()

        if extension in extensions:
            files.append(path)

        if len(files) >= limit:
            break

    return files


fake_files = get_images(
    fake_dir,
    MAX_PER_CLASS
)

real_files = get_images(
    real_dir,
    MAX_PER_CLASS
)

print()
print("FAKE images selected:", len(fake_files))
print("REAL images selected:", len(real_files))

if len(fake_files) == 0:
    raise ValueError("No FAKE images found.")

if len(real_files) == 0:
    raise ValueError("No REAL images found.")

# ============================================================
# LOAD IMAGE DATASET
# ============================================================

data = []

for path in fake_files:

    data.append({
        "image": path,
        "label": 1
    })


for path in real_files:

    data.append({
        "image": path,
        "label": 0
    })


print()
print("Total test-training images:", len(data))

# ============================================================
# LOAD MODEL
# ============================================================

MODEL_NAME = "google/vit-base-patch16-224"

print()
print("Loading processor...")
processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME
)

print("Loading pretrained model...")

model = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={
        0: "human",
        1: "AI-generated"
    },
    label2id={
        "human": 0,
        "AI-generated": 1
    },
    ignore_mismatched_sizes=True
)

print()
print("Model loaded.")

# ============================================================
# IMAGE TRANSFORM
# ============================================================

image_mean = processor.image_mean
image_std = processor.image_std

image_size = processor.size["height"]

transform = Compose([
    Resize(
        (image_size, image_size)
    ),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(
        mean=image_mean,
        std=image_std
    )
])

# ============================================================
# DATASET CLASS
# ============================================================

from torch.utils.data import Dataset


class ImageDataset(Dataset):

    def __init__(self, items):

        self.items = items

    def __len__(self):

        return len(self.items)

    def __getitem__(self, index):

        from PIL import Image

        item = self.items[index]

        image = Image.open(
            item["image"]
        ).convert("RGB")

        image = transform(image)

        return {
            "pixel_values": image,
            "labels": torch.tensor(
                item["label"],
                dtype=torch.long
            )
        }


train_dataset = ImageDataset(data)

# ============================================================
# TRAINING SETTINGS
# ============================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    num_train_epochs=1,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=1,

    learning_rate=2e-5,

    logging_steps=20,

    save_strategy="no",

    report_to="none",

    dataloader_num_workers=0,

    remove_unused_columns=False,

    use_cpu=True
)

# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset
)

# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 60)
print("STARTING SMALL CPU TRAINING TEST")
print("=" * 60)

trainer.train()

# ============================================================
# SAVE
# ============================================================

print()
print("Saving test model...")

trainer.save_model(
    OUTPUT_DIR
)

processor.save_pretrained(
    OUTPUT_DIR
)

print()
print("=" * 60)
print("SANITY TEST COMPLETE")
print("=" * 60)

print()
print("Model saved to:")

print(OUTPUT_DIR)

print()
print("Labels:")

print({
    0: "human",
    1: "AI-generated"
})

print()
print("If you see this message, the training pipeline works.")
print()
