import exifread


def extract_metadata(file_path):
    metadata = {}

    try:
        with open(file_path, "rb") as f:
            tags = exifread.process_file(f, details=False)

        for tag, value in tags.items():
            metadata[str(tag)] = str(value)

    except Exception as exc:
        return {
            "status": "error",
            "metadata": {},
            "findings": [str(exc)],
        }

    findings = []

    if metadata:
        findings.append(f"{len(metadata)} metadata field(s) detected.")
    else:
        findings.append("No readable EXIF metadata was detected.")

    return {
        "status": "success",
        "metadata": metadata,
        "findings": findings,
    }
