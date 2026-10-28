import hashlib
import cv2
import numpy as np


def calculate_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def calculate_md5(file_path):
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def calculate_phash(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        return "UNAVAILABLE"

    resized = cv2.resize(image, (32, 32))
    dct = cv2.dct(np.float32(resized))
    low = dct[:8, :8]
    median = np.median(low[1:, :])
    bits = (low > median).flatten()

    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)

    return f"{value:016x}"


# Compatibility with the older project.
def generate_hash(file_path):
    return calculate_sha256(file_path)
