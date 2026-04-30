# NinjutsuAR

Real-time hand-sign recognition for Android AR. Perform Naruto-style seals (and a few Jujutsu Kaisen poses) in front of your phone camera; the app recognizes the seal, chains seals into "jutsu", and spawns AR effects anchored to the floor through ARCore.

The on-device pipeline runs three ONNX models through Unity's Inference Engine:

- `palm_detection_full.onnx` (MediaPipe, 192x192) — hand localization
- `mobilevit_seals_v3.onnx` (MobileViT-XXS, 256x256) — 15-class seal classifier
- `yolov8n-seg.onnx` (YOLOv8n-seg, 640x640) — person segmentation for the Domain Expansion overlay

The Unity project lives in [App/](App/). Training and dataset scripts live at the repo root.

---

## 1. Requirements

### Hardware

- An **ARCore-supported Android phone** (Android 10 / API 29 or newer). Check the official list: https://developers.google.com/ar/devices
- A USB-C cable for deploying via USB.

### Software (Windows / macOS / Linux)

- **Unity Hub** + **Unity 6000.4.3f1** (the exact version this project was authored with).
  When installing the editor, tick:
  - **Android Build Support**
    - OpenJDK
    - Android SDK & NDK Tools
- **Git** (with Git LFS if you plan to pull the larger ONNX/Unity asset packs).
- (Optional, for retraining only) **Python 3.10+** with the packages in [requirements.txt](requirements.txt).

---

## 2. Get the code

```bash
git clone https://github.com/YousefAyesh/NinjutsuAR.git
cd NinjutsuAR
```

The Unity project root is `App/`. The first time Unity Hub opens it, it will reimport assets and resolve packages — this can take several minutes.

---

## 3. Open the project in Unity

1. Launch **Unity Hub**.
2. Click **Add → Add project from disk** and select the `App/` folder.
3. Confirm the editor version is **6000.4.3f1**. If Hub prompts to install it, do so before opening.
4. Open the project. Unity will import packages and ONNX models on first launch.
5. Open the scene: `App/Assets/Scenes/SampleScene.unity`.

If you see model-import errors, make sure `com.unity.ai.inference` (Inference Engine 2.x) finished installing under **Window → Package Manager**.

---

## 4. Configure for Android build

In the Unity editor:

1. **File → Build Profiles** (or **Build Settings** in older menus) → select **Android** → **Switch Platform**.
2. **Edit → Project Settings → Player → Android tab**:
   - **Other Settings → Identification**
     - Package Name: `com.<yourorg>.ninjutsuar` (anything unique).
     - Minimum API Level: **Android 10.0 (API 29)** — required by the ARCore package shipped here.
     - Target API Level: **Automatic (highest installed)**.
   - **Other Settings → Configuration**
     - Scripting Backend: **IL2CPP**.
     - Target Architectures: tick **ARM64** (untick ARMv7).
   - **Resolution and Presentation**
     - Default Orientation: **Portrait** (the camera-to-screen mapping in `HandSignManager.CropHandRegion` is written for portrait).
3. **Project Settings → XR Plug-in Management → Android tab**: enable **ARCore**.
4. **Project Settings → Player → Android → Publishing Settings → Build**: tick **Custom Main Manifest** only if you plan to add extra permissions; the AR camera permission is already requested by ARFoundation.

---

## 5. Enable the phone for development

On the Android device:

1. **Settings → About phone** → tap **Build number** seven times to unlock Developer options.
2. **Settings → System → Developer options** → enable **USB debugging**.
3. Plug the phone into the PC with USB-C. When prompted on the phone, **Allow USB debugging** for this computer.
4. (First time only) Make sure **Google Play Services for AR** is installed/updated from the Play Store.

Verify the phone is visible to Unity:

- **File → Build Profiles → Android → Run Device** dropdown — your device should appear. Click **Refresh** if not.

---

## 6. Build and run

The fastest path is **Build And Run** with the phone plugged in:

1. **File → Build Profiles → Android**.
2. Confirm the scene `Scenes/SampleScene` is in **Scenes In Build**.
3. Pick your device under **Run Device**.
4. Click **Build And Run**. Choose any output path for the `.apk` (e.g. `App/build/`).
5. Unity will compile, install, and launch the app on the phone. Grant the **Camera** permission when prompted.

To produce just an APK without installing, click **Build** instead.

---

## 7. Using the app

- Point the camera at a flat surface for a second or two so ARCore detects the floor plane.
- Make a seal with one hand in view of the camera. The label and confidence appear on screen.
- Hold a seal: triggers a **Hold** jutsu after `holdSeconds` (e.g. Malevolent Shrine).
- Chain seals: the recognizer matches the buffer suffix against any registered **Sequence** jutsu (e.g. Dragon → Tiger → Fireball).
- Default thresholds (configurable in the Inspector on the `HandSignManager` GameObject):
  - `inferenceEveryNFrames = 6`
  - `minConfidence = 0.65`

### Capturing real-device frames (optional)

`HandSignManager` can save or upload each crop for dataset growth.

- Local: tick **Capture Enabled** and untick **Upload To Server**. PNGs land under `/Android/data/<package>/files/captures/` on the phone.
- Network: tick **Upload To Server**, set **Upload Url** to your laptop's LAN IP (e.g. `http://192.168.1.42:8000/upload`), and on the laptop run:

  ```bash
  python tools/capture_server.py --host 0.0.0.0 --port 8000 --out captures
  ```

  Make sure the firewall allows inbound on that port.

---

## 8. Troubleshooting

- **Black screen / "AR not supported"**: device isn't on Google's ARCore list, or **Google Play Services for AR** is missing from the Play Store.
- **Camera permission denied**: uninstall and reinstall the app, or grant it manually in **Settings → Apps → NinjutsuAR → Permissions**.
- **Inference Engine errors on import**: open **Window → Package Manager**, find **AI Inference Engine**, and click **Update** / **Reimport**.
- **Hand never detected**: confirm `MediaPipeHandDetector` on the scene's hand-detector GameObject has `palm_detection_full.onnx` assigned in the **Palm Model** slot. If the ONNX is missing, run `python tools/fetch_mediapipe_palm.py` from the repo root to regenerate it.
- **Stuck classifier output**: lower `minConfidence` on `HandSignManager`, or increase `inferenceEveryNFrames` if the device is overheating.
- **Wrong orientation / mirrored crop**: the crop math assumes portrait. Force the app to portrait in **Player → Resolution and Presentation**.

---

## 9. (Optional) Retrain the classifier

Only needed if you change the seal vocabulary or collect more data.

```bash
pip install -r requirements.txt
python -c "from roboflow import Roboflow; \
  Roboflow(api_key='<your-key>') \
    .workspace('ammans-workspace').project('ninjutsuar-4152') \
    .version(6).download('folder')"   # downloads dataset/
python train.py     # writes best_model.pth
python export.py    # writes mobilevit_seals.onnx
```

Drop the new ONNX into `App/Assets/` and assign it to `SealClassifier.modelAsset` in the scene.

---

## Repo layout

```
App/                       Unity 6000.4.3f1 ARFoundation/ARCore project
  Assets/Scripts/          C# MonoBehaviours (HandSignManager, SealClassifier, ...)
  Assets/Models/           palm_detection_full.onnx, yolov8*-seg.onnx
  Assets/mobilevit_seals_*.onnx   Trained classifier exports
  Assets/Scenes/SampleScene.unity Main AR scene
tools/
  fetch_mediapipe_palm.py  Downloads + converts MediaPipe palm detector to ONNX
  capture_server.py        Tiny HTTP receiver for on-device debug captures
train.py / export.py       PyTorch training + ONNX export for MobileViT-XXS
inspect_model.py           Quick ONNX I/O sanity check
report.tex                 Project report
```
