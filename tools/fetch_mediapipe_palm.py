"""
Fetches MediaPipe palm_detection_full (192x192) and converts the TFLite to ONNX
for Unity Inference Engine.

Run once on the laptop:

    pip install tensorflow tf2onnx onnx
    python tools/fetch_mediapipe_palm.py

Output:
    App/Assets/Models/palm_detection_full.onnx

Then in Unity:
- Wait for Unity to import the .onnx (a .meta file appears next to it).
- On the GameObject that has HandBoundingBoxDetector, add the new
  MediaPipeHandDetector component, drag palm_detection_full.onnx into its
  'Palm Model' slot, and disable HandBoundingBoxDetector.
- HandSignManager will pick up MediaPipeHandDetector automatically because
  its 'handDetector' field is HandDetectorBase.

If tf2onnx names the outputs differently from "Identity" / "Identity_1",
the conversion log will show the real names. Set them in the
'Regressors Output Name' / 'Scores Output Name' fields of the component.
"""

from __future__ import annotations

import os
import subprocess
import sys
import urllib.request

TFLITE_URLS = [
    # Current MediaPipe asset bucket (used by mediapipe-tasks).
    "https://storage.googleapis.com/mediapipe-assets/palm_detection_full.tflite",
    # Older bucket layout, kept as a fallback.
    "https://storage.googleapis.com/mediapipe-models/palm_detection/"
    "palm_detection_full/float16/1/palm_detection_full.tflite",
    # Repo-hosted copy as a last resort.
    "https://github.com/google-ai-edge/mediapipe/raw/master/"
    "mediapipe/modules/palm_detection/palm_detection_full.tflite",
]

# Pre-converted fp32 ONNX (PINTO model zoo). Use if Google's float16 tflite
# saturates after tf2onnx conversion. Set USE_PINTO=1 in the environment to
# skip the download+convert path entirely.
PINTO_ONNX_URL = (
    "https://github.com/PINTO0309/PINTO_model_zoo/raw/main/"
    "033_Hand_Detection_and_Tracking/03_palm_detection_full/"
    "saved_model_192x192/palm_detection_full_192x192.onnx"
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "App", "Assets", "Models")
TFLITE_PATH = os.path.join(OUT_DIR, "palm_detection_full.tflite")
ONNX_PATH = os.path.join(OUT_DIR, "palm_detection_full.onnx")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    if os.environ.get("USE_PINTO"):
        print(f"USE_PINTO set; downloading pre-converted ONNX from {PINTO_ONNX_URL}")
        urllib.request.urlretrieve(PINTO_ONNX_URL, ONNX_PATH)
        print(f"Wrote {ONNX_PATH}")
        return

    if not os.path.exists(TFLITE_PATH):
        last_err = None
        for url in TFLITE_URLS:
            try:
                print(f"Downloading {url}")
                urllib.request.urlretrieve(url, TFLITE_PATH)
                print(f"  -> {TFLITE_PATH}")
                last_err = None
                break
            except Exception as e:  # noqa: BLE001
                print(f"  failed: {e}")
                last_err = e
        if last_err is not None:
            raise last_err
    else:
        print(f"Already have {TFLITE_PATH}")

    print("Converting TFLite -> ONNX (tf2onnx)...")
    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--tflite", TFLITE_PATH,
        "--output", ONNX_PATH,
        "--opset", "13",
    ]
    subprocess.check_call(cmd)
    print(f"Wrote {ONNX_PATH}")
    print()
    print("Done. Open Unity to let it import the ONNX, then assign it to "
          "MediaPipeHandDetector.palmModel.")


if __name__ == "__main__":
    main()
