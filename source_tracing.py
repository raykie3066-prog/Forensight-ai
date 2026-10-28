import os
import cv2
import numpy as np

from hashing import calculate_phash


def _phash_bits(hex_hash):
    try:
        value = int(hex_hash, 16)
        return np.array(
            [(value >> i) & 1 for i in range(64)],
            dtype=np.uint8,
        )
    except Exception:
        return None


def hamming_similarity(hash_a, hash_b):
    a = _phash_bits(hash_a)
    b = _phash_bits(hash_b)

    if a is None or b is None:
        return 0.0

    distance = int(np.count_nonzero(a != b))
    return round((1.0 - distance / 64.0) * 100.0, 2)


def trace_source(image_path, reference_dir=None):
    phash = calculate_phash(image_path)

    if reference_dir is None:
        reference_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "reference_images",
        )

    matches = []

    if os.path.isdir(reference_dir):
        for root, _, files in os.walk(reference_dir):
            for filename in files:
                if not filename.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".webp", ".bmp")
                ):
                    continue

                path = os.path.join(root, filename)
                candidate_hash = calculate_phash(path)
                similarity = hamming_similarity(phash, candidate_hash)

                if similarity >= 70:
                    matches.append({
                        "filename": filename,
                        "similarity": similarity,
                        "path": path,
                    })

    matches.sort(key=lambda x: x["similarity"], reverse=True)

    if matches:
        findings = [
            "Potentially similar reference images were found.",
            "Similarity is based on perceptual hashing only.",
        ]
    else:
        findings = [
            "No visually similar image was found in the current local reference database.",
            "This is not an internet reverse-image search.",
        ]

    return {
        "status": "success",
        "phash": phash,
        "matches": matches[:10],
        "findings": findings,
    }
