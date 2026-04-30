# NinjutsuAR — Model branch (`abel-model-code`)

This branch contains the **PyTorch training pipeline and exported ONNX models** for the 15-class hand-seal classifier used by NinjutsuAR. Everything you need to (a) run inference with the already-trained model and (b) retrain it from scratch lives in [NewModel/](NewModel/). The Unity AR app is on a different branch — this branch is intentionally just the model.

The classifier is **MobileViT-XXS** fine-tuned on a Roboflow-managed custom dataset of 15 classes:

```
Bird, Boar, Dog, Dragon, Hare, HollowPurple, Horse, InfiniteVoid,
MalevolentShrine, Monkey, Ox, Ram, Rat, Snake, Tiger
```

The class index ↔ name mapping the trained model expects is shipped in
[NewModel/class_names.json](NewModel/class_names.json).

---

## Folder layout

```
NewModel/
  Download_dataset.py       # Pull dataset Version 6 from Roboflow
  Crop_hands.py             # MediaPipe Hand Landmarker → tight 256x256 hand crops
  hand_landmarker.task      # MediaPipe model file (auto-downloaded by Crop_hands.py)
  train.py                  # Two-phase MobileViT-XXS fine-tuning
  export.py                 # Best PyTorch checkpoint → ONNX
  infer_onnx.py             # CLI inference on a single image (or random pick)
  debug.py / debug2.py      # Sanity checks for the data + model pipelines
  pipeline.ipynb            # End-to-end notebook version of the same flow
  best_model.pth            # Trained PyTorch weights
  mobilevit_seals.onnx      # Initial ONNX export
  mobilevit_seals_v2.onnx   # Iteration with cleaner crops
  mobilevit_seals_v3.onnx   # Latest export (id2label / label2id baked in)
  class_names.json          # Index → class name map
mobilevit_seals.onnx        # (Top-level mirror used by the Unity branch)
App/                        # Stub Unity folder (full app lives on another branch)
```

---

## 1. Set up Python

Tested on Python 3.10 / 3.11.

```bash
# Clone and switch to this branch
git clone https://github.com/YousefAyesh/NinjutsuAR.git
cd NinjutsuAR
git checkout abel-model-code
cd NewModel

# (Recommended) create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision transformers
pip install onnx onnxruntime
pip install opencv-python pillow numpy
pip install mediapipe
pip install roboflow
```

If you have an NVIDIA GPU, install a CUDA-enabled PyTorch build from
https://pytorch.org/get-started/locally/ first (the `train.py` script auto-selects
CUDA → MPS → CPU in that order).

---

## 2. Run the trained model (no retraining)

The fastest way to confirm the environment works.

### Single image

```bash
cd NewModel
python infer_onnx.py path/to/your/hand_sign.jpg --model mobilevit_seals_v3.onnx
```

### Random image from the dataset

After you have run the dataset + crop steps in section 3, you can let the
script pick a random cropped sample for you:

```bash
python infer_onnx.py --model mobilevit_seals_v3.onnx --dataset-root dataset_cropped
```

Sample output:

```
Image: dataset_cropped/test/Tiger/img_0042.jpg
Model: mobilevit_seals_v3.onnx
Top-1: class=Tiger | index=14 | logit=8.421 | prob=0.972
Final prediction with threshold 0.55: Tiger
Top predictions:
  1. Tiger        prob=0.9721
  2. Dragon       prob=0.0178
  ...
```

Useful flags:

- `--topk 5` — number of predictions to print.
- `--unknown-threshold 0.55` — top-1 probability below this prints `UNKNOWN`.
- `--classes class_names.json` — explicit class map (defaults to the file in CWD).

The same preprocessing as training is applied: resize to 256×256, scale to
`[0, 1]`, then normalize with mean/std `0.5` (mapping to `[-1, 1]`).

---

## 3. Retrain the model from scratch

The full pipeline is **download → crop → train → export**. All commands run from
inside `NewModel/`.

### 3.1 Download the dataset (Roboflow)

`Download_dataset.py` pulls Version 6 of the project's Roboflow workspace into a
folder structured as `NinjutsuAR---4152-6/{train,valid,test}/<class>/<image>`.

The committed file uses a placeholder API key. Replace it with your own
Roboflow API key (Roboflow → Settings → API key) before running:

```python
# NewModel/Download_dataset.py
rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
```

Then:

```bash
python Download_dataset.py
```

You should end up with `NewModel/NinjutsuAR---4152-6/` containing
roughly 16,000 images split 70/20/10 across `train/`, `valid/`, `test/`.

### 3.2 Crop hands with MediaPipe

`Crop_hands.py` runs Google's MediaPipe Hand Landmarker on every image,
expands a tight bounding box around the 21 landmarks by 20 px, resizes to
256×256, and writes the result to a parallel `dataset_cropped/` tree.
The `hand_landmarker.task` model is downloaded automatically on first run.

```bash
python Crop_hands.py
```

The script prints progress per split. Frames where MediaPipe fails to find a
hand are skipped (and counted) — this is expected and matches the failure
analysis in the report.

The resulting layout is:

```
dataset_cropped/
  train/<class>/*.jpg
  valid/<class>/*.jpg
  test/<class>/*.jpg
```

### 3.3 (Optional) Sanity-check the pipeline

```bash
python debug.py     # Verifies the DataLoader produces [B, 3, 256, 256] tensors
python debug2.py    # Forward-passes a batch through MobileViT-XXS, prints [B, 15]
```

If either crashes, fix it before launching a multi-hour training run.

### 3.4 Train MobileViT-XXS

```bash
python train.py
```

What happens:

1. Loads `dataset_cropped/{train,valid,test}` with `torchvision.ImageFolder`.
2. Asserts that the class order is identical across all three splits.
3. Builds `apple/mobilevit-xx-small` from the HuggingFace hub with a fresh
   15-logit classification head.
4. **Phase 1 (10 epochs):** freezes the backbone, trains only the classifier
   head with AdamW at `lr=1e-2`.
5. **Phase 2 (20 epochs):** unfreezes everything, fine-tunes with AdamW at
   `lr=1e-4`, cosine annealing, label smoothing `0.1`, gradient clipping at
   `1.0`, and stronger weight decay (`5e-2`).
6. Saves the best validation checkpoint to `best_model.pth` whenever it improves.

Training transforms include random crop/pad, random resized crop, affine
jitter, color jitter, occasional grayscale, and mild Gaussian blur — chosen to
mimic the looser palm-detector crops produced at inference time. Validation
and test splits use only resize → grayscale(3) → normalize.

Hardware notes:

- The script auto-selects MPS (Apple Silicon) → CUDA → CPU.
- On CPU, expect training to take several hours; on a modern GPU it finishes
  in well under an hour.
- DataLoader workers default to `4`, but drop to `0` automatically inside
  IPython/Jupyter on Windows (see `choose_num_workers`).

### 3.5 Export to ONNX

Once `best_model.pth` exists:

```bash
python export.py
```

This:

- Reloads `apple/mobilevit-xx-small` with 15 classes.
- Bakes the class names into `model.config.id2label` / `label2id`.
- Loads the best PyTorch weights.
- Runs `torch.onnx.export` with a `1×3×256×256` dummy input, opset 13, named
  I/O (`pixel_values` → `logits`), and a dynamic batch axis.
- Writes `mobilevit_seals_v3.onnx` next to the script.

Verify the export end-to-end:

```bash
python infer_onnx.py --model mobilevit_seals_v3.onnx --dataset-root dataset_cropped
```

If the predictions look right, the ONNX is ready to drop into the Unity
project on the app branch (under `App/Assets/`).

---

## 4. Reproducing the notebook flow

If you prefer a single-file walkthrough, [`NewModel/pipeline.ipynb`](NewModel/pipeline.ipynb)
runs the same `download → crop → train → export → infer` sequence with inline
commentary. It is functionally equivalent to the scripts above.

---

## 5. Troubleshooting

- **`OSError: cannot import mediapipe`** — install the prebuilt wheel: `pip install mediapipe`. On Apple Silicon use Python 3.10/3.11; 3.12 wheels are not always available.
- **Roboflow auth fails** — make sure you put your own API key in `Download_dataset.py`. The committed key is a placeholder.
- **`Class order mismatch between splits`** during training — one of the split folders is missing a class or has an extra one. Inspect `dataset_cropped/{train,valid,test}/` and re-run `Crop_hands.py` if needed.
- **Out of memory on GPU** — drop `batch_size=64` in `train.py` to `32` or `16`.
- **`UNKNOWN` predictions on every image** — your top-1 probability is below the default `0.55` threshold. Pass `--unknown-threshold 0.0` to disable, or check that the input image actually contains a hand seal.
- **ONNX runtime errors loading `mobilevit_seals_v3.onnx`** — install/upgrade `onnxruntime` (`pip install -U onnxruntime`).

---

## 6. What this branch does _not_ contain

- The Unity 6 ARFoundation/ARCore project, scenes, ScriptableObjects, jutsu sequence recognizer, person-segmentation overlay, and on-device hand detector. Those live on the app branch (e.g. `main` / `abel`). The `App/` folder here is a stub.
- The HaGRID baseline experiments mentioned in the report.

For the full system architecture and AR build instructions, switch to the app
branch and read its `README.md`.
