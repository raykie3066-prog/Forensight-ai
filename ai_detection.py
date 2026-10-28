import os
from functools import lru_cache

# Disable HuggingFace Hub symlinks — avoids [Errno 22] on Windows without
# Developer Mode enabled.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ── Detector model ────────────────────────────────────────────────────────────
# umm-maybe/AI-image-detector: ViT trained on DALL-E, Stable Diffusion,
# Midjourney, and real photographs.  Labels: { 0: "artificial", 1: "human" }.
MODEL_NAME = "umm-maybe/AI-image-detector"


def _resolve_indices(model):
    """
    Return (ai_indices: list[int], human_index: int) for any classification
    model whose id2label uses recognisable AI / human label strings.
    """
    ai_indices = []
    human_index = None

    for idx, label in model.config.id2label.items():
        norm = str(label).lower()
        if any(x in norm for x in (
            "ai", "generated", "fake", "synthetic", "artificial",
            "dalle", "sd", "midjourney", "gan", "diffusion",
        )):
            ai_indices.append(int(idx))
        if any(x in norm for x in ("real", "human", "natural", "authentic")):
            human_index = int(idx)

    if not ai_indices or human_index is None:
        raise ValueError(
            f"Cannot map AI/human labels from {model.config.id2label}"
        )
    if len(ai_indices) == 1 and ai_indices[0] == human_index:
        raise ValueError("AI and human indices are identical.")
    return ai_indices, human_index


@lru_cache(maxsize=1)
def load_detector():
    """Load the detector once and cache it for the lifetime of the process."""
    try:
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=False)
    except Exception:
        # Older transformers builds (< 5) need the explicit class name.
        from transformers import ViTImageProcessor
        processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    model.to("cpu")
    ai_indices, human_index = _resolve_indices(model)
    return processor, model, ai_indices, human_index


def _run(image, processor, model, ai_indices, human_index):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits[0], dim=-1)
    ai_prob   = float(sum(probs[i] for i in ai_indices))
    hum_prob  = float(probs[human_index])
    return ai_prob, hum_prob


def _verdict(ai_prob):
    if ai_prob >= 0.70:
        return "AI-GENERATED"
    if ai_prob <= 0.30:
        return "HUMAN-CREATED"
    return "INCONCLUSIVE"


def analyze_ai_indicators(image_path, use_secondary=False):
    """
    Screen an image for AI-generation indicators using a single reliable
    pretrained ViT detector.

    Returns a dict with keys:
        status, assessment, ai_percentage, human_percentage,
        confidence, risk_level, findings, model
    """
    try:
        image = Image.open(image_path).convert("RGB")
        processor, model, ai_indices, human_index = load_detector()
        ai_prob, hum_prob = _run(image, processor, model, ai_indices, human_index)
        verdict = _verdict(ai_prob)

        if verdict == "AI-GENERATED":
            assessment  = "AI-GENERATED SIGNAL"
            risk_level  = "HIGH"
            confidence  = f"Detector score: {ai_prob*100:.1f}% AI"
        elif verdict == "HUMAN-CREATED":
            assessment  = "HUMAN-CREATED SIGNAL"
            risk_level  = "LOW"
            confidence  = f"Detector score: {hum_prob*100:.1f}% human"
        else:
            assessment  = "INCONCLUSIVE"
            risk_level  = "UNCERTAIN"
            confidence  = (
                f"AI score {ai_prob*100:.1f}% falls in the 30–70% uncertainty band"
            )

        findings = [
            f"AI score:    {ai_prob  * 100:.2f}%",
            f"Human score: {hum_prob * 100:.2f}%",
            f"Verdict:     {assessment}",
            "This result is an automated screening indicator — not definitive proof of origin.",
        ]

        return {
            "status":               "success",
            "assessment":           assessment,
            "prediction":           assessment,
            "ai_probability":       round(ai_prob,  6),
            "human_probability":    round(hum_prob, 6),
            "ai_percentage":        round(ai_prob  * 100, 2),
            "human_percentage":     round(hum_prob * 100, 2),
            "confidence":           confidence,
            "risk_level":           risk_level,
            "model":                MODEL_NAME,
            "secondary_model":      None,
            "secondary_ai_percentage":    None,
            "secondary_human_percentage": None,
            "findings":             findings,
        }

    except Exception as exc:
        return {
            "status":            "error",
            "assessment":        "UNAVAILABLE",
            "prediction":        "UNAVAILABLE",
            "ai_percentage":     0,
            "human_percentage":  0,
            "confidence":        "N/A",
            "risk_level":        "UNAVAILABLE",
            "findings":          [str(exc)],
            "model":             None,
            "secondary_model":   None,
            "secondary_ai_percentage":    None,
            "secondary_human_percentage": None,
        }
