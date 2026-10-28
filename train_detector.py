import os
import random
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import (
    Compose,
    Resize,
    RandomHorizontalFlip,
    ToTensor,
    Normalize,
)

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = r"C:\Users\manud\Downloads\archive"

OUTPUT_DIR = (
    r"C:\Users\manud\OneDrive\Desktop\Forensic Science"
    r"\codes\forensics_ai_detector"
)

MODEL_NAME = "google/mobilenet_v2_1.0_224"

# Start with 5,000 images from each class.
MAX_PER_CLASS = 5000

SEED = 42


# ============================================================
# SET SEED
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("FORENSIGHT-AI — CPU TRAINING")
print("=" * 70)

print()
print("Dataset:", DATASET_DIR)
print("Output :", OUTPUT_DIR)
print("Model  :", MODEL_NAME)
print("Device : CPU")

print()
print("Images per class:", MAX_PER_CLASS)


# ============================================================
# CHECK DATASET
# ============================================================

FAKE_DIR = os.path.join(
    DATASET_DIR,
    "train",
    "FAKE"
)

REAL_DIR = os.path.join(
    DATASET_DIR,
    "train",
    "REAL"
)


if not os.path.isdir(FAKE_DIR):
    raise FileNotFoundError(
        f"FAKE folder not found:\n{FAKE_DIR}"
    )


if not os.path.isdir(REAL_DIR):
    raise FileNotFoundError(
        f"REAL folder not found:\n{REAL_DIR}"
    )


# ============================================================
# IMAGE FILES
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}


def collect_images(folder):

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

        if extension in IMAGE_EXTENSIONS:
            files.append(path)

    random.shuffle(files)

    return files


print()
print("Scanning dataset...")


fake_files = collect_images(
    FAKE_DIR
)

real_files = collect_images(
    REAL_DIR
)


print()
print("FAKE images available:", len(fake_files))
print("REAL images available:", len(real_files))


# ============================================================
# SELECT BALANCED DATA
# ============================================================

fake_files = fake_files[
    :MAX_PER_CLASS
]

real_files = real_files[
    :MAX_PER_CLASS
]


print()
print("Training images selected:")
print("FAKE:", len(fake_files))
print("REAL:", len(real_files))
print("TOTAL:", len(fake_files) + len(real_files))


if len(fake_files) < MAX_PER_CLASS:
    raise ValueError(
        "Not enough FAKE images."
    )


if len(real_files) < MAX_PER_CLASS:
    raise ValueError(
        "Not enough REAL images."
    )


# ============================================================
# CREATE DATASET ITEMS
# ============================================================

items = []

# FAKE = AI-generated = label 1

for path in fake_files:

    items.append({
        "path": path,
        "label": 1
    })


# REAL = human-created = label 0

for path in real_files:

    items.append({
        "path": path,
        "label": 0
    })


random.shuffle(items)


# ============================================================
# LOAD PROCESSOR
# ============================================================

print()
print("Loading image processor...")

processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME
)


# ============================================================
# LOAD PRETRAINED MODEL
# ============================================================

print()
print("Loading pretrained MobileNetV2...")

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
print("Labels:", model.config.id2label)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

image_size = 224

image_mean = processor.image_mean
image_std = processor.image_std


transform = Compose([

    Resize(
        (image_size, image_size)
    ),

    RandomHorizontalFlip(
        p=0.5
    ),

    ToTensor(),

    Normalize(
        mean=image_mean,
        std=image_std
    )
])


# ============================================================
# DATASET CLASS
# ============================================================

class ImageDataset(Dataset):

    def __init__(
        self,
        items
    ):

        self.items = items


    def __len__(self):

        return len(self.items)


    def __getitem__(
        self,
        index
    ):

        item = self.items[index]

        image = Image.open(
            item["path"]
        ).convert("RGB")

        image = transform(
            image
        )

        return {
            "pixel_values": image,
            "labels": torch.tensor(
                item["label"],
                dtype=torch.long
            )
        }


train_dataset = ImageDataset(
    items
)


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    num_train_epochs=2,

    per_device_train_batch_size=8,

    gradient_accumulation_steps=1,

    learning_rate=2e-5,

    weight_decay=0.01,

    logging_steps=100,

    save_strategy="epoch",

    save_total_limit=1,

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
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

print()
print(
    "This will train on "
    f"{len(train_dataset):,} images."
)

print()
print("Please do not close PowerShell while training.")


trainer.train()


# ============================================================
# SAVE FINAL MODEL
# ============================================================

print()
print("=" * 70)
print("SAVING FINAL MODEL")
print("=" * 70)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


trainer.save_model(
    OUTPUT_DIR
)


processor.save_pretrained(
    OUTPUT_DIR
)


# ============================================================
# VERIFY
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print()
print("Model saved to:")

print(OUTPUT_DIR)

print()
print("Labels:")

print(
    model.config.id2label
)

print()
print("IMPORTANT:")
print(
    "The model must be evaluated on the separate "
    "test dataset before claiming accuracy."
)

print()