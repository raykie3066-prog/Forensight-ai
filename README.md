FORENSIGHT-AI SUBMISSION BUILD
==============================

This build is designed to stabilize the project for submission.

IMPORTANT MODEL BEHAVIOR
------------------------
The custom detector is retained, but it is not allowed to force a conclusion
when the independent detector disagrees.

Final AI screening:
- both AI direction -> AI-GENERATED SIGNAL
- both human direction -> HUMAN-CREATED SIGNAL
- disagreement/uncertainty -> INCONCLUSIVE

The percentages are model scores, not calibrated forensic probabilities.

FACIAL ANALYSIS
---------------
Uses MediaPipe Face Mesh and supports up to 10 faces.
The annotated image marks:
- face box
- eye/pupil landmarks
- interpupillary distance
- nose width
- nose length
- mouth width
- left/right eye width
- face height

The same annotated image is inserted into the PDF report.

SOURCE TRACING
--------------
This version performs local perceptual-hash matching against:
    reference_images/

It is not an internet reverse-image search.

RUN
---
From the project folder:

python -m pip install -r requirements.txt
python check_submission.py
python -m streamlit run app.py

If MediaPipe currently reports:
    module 'mediapipe' has no attribute 'solutions'

run:
    python -m pip uninstall mediapipe -y
    python -m pip install mediapipe==0.10.21

Then run:
    python check_submission.py

MODEL
-----
Copy the folder:
    forensics_ai_detector_v2

into the same directory as app.py.

If v2 does not exist, the app also checks:
    forensics_ai_detector

OPTIONAL REFERENCE DATABASE
---------------------------
Create:
    reference_images

Put known reference images there. Source tracing will compare the uploaded
image to those local references using perceptual hashing.

SUBMISSION NOTE
---------------
Do not claim the detector is 100% accurate.
Describe it as an automated AI-media screening prototype with corroborating
image, metadata, facial and source-tracing modules.
