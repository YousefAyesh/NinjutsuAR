import urllib.request
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import cv2
import os
from PIL import Image

# Download the hand landmarker model if not already present
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
if not os.path.exists(MODEL_PATH):
    print(f"Downloading hand landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")

_base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
_options = mp_vision.HandLandmarkerOptions(
    base_options=_base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    running_mode=mp_vision.RunningMode.IMAGE,
)
landmarker = mp_vision.HandLandmarker.create_from_options(_options)


def crop_hand(image_path, output_path, padding=20):
    image = cv2.imread(image_path)
    if image is None:
        return False

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return False

    h, w, _ = image.shape
    landmarks = result.hand_landmarks[0]

    x_coords = [lm.x * w for lm in landmarks]
    y_coords = [lm.y * h for lm in landmarks]

    x_min = max(0, int(min(x_coords)) - padding)
    x_max = min(w, int(max(x_coords)) + padding)
    y_min = max(0, int(min(y_coords)) - padding)
    y_max = min(h, int(max(y_coords)) + padding)

    cropped = image[y_min:y_max, x_min:x_max]
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cropped_rgb)
    pil_image = pil_image.resize((256, 256))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pil_image.save(output_path)
    return True


if __name__ == "__main__":
    import time

    input_root  = "NinjutsuAR---4152-6"
    output_root = "dataset_cropped"
    splits      = ["train", "valid", "test"]

    # Count total images up front so we can show a percentage
    all_images = [
        os.path.join(root, f)
        for split in splits
        for root, _, files in os.walk(os.path.join(input_root, split))
        for f in files
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    grand_total = len(all_images)
    print(f"Found {grand_total} images to process.\n")

    total, success, failed = 0, 0, 0
    failed_classes = {}
    start_time = time.time()

    for split in splits:
        split_path = os.path.join(input_root, split)
        if not os.path.exists(split_path):
            continue
        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if not os.path.isdir(class_path):
                continue
            class_fail = 0
            for img_file in os.listdir(class_path):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                input_path  = os.path.join(class_path, img_file)
                output_path = os.path.join(output_root, split, class_name, img_file)
                total += 1
                if crop_hand(input_path, output_path):
                    success += 1
                else:
                    failed += 1
                    class_fail += 1

                # Progress update every 50 images
                if total % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = total / elapsed
                    remaining = (grand_total - total) / rate if rate > 0 else 0
                    print(f"  [{total}/{grand_total}] "
                          f"{total/grand_total*100:.1f}%  |  "
                          f"{rate:.1f} img/s  |  "
                          f"ETA: {remaining/60:.1f} min")
            if class_fail > 0:
                failed_classes[f"{split}/{class_name}"] = class_fail

    print(f"\nTotal: {total}")
    print(f"Successfully cropped: {success}")
    print(f"Failed (no hand detected): {failed}")
    if total > 0:
        print(f"Overall drop rate: {failed/total*100:.1f}%")
    print(f"\nFailed by class:")
    for cls, count in failed_classes.items():
        print(f"  {cls}: {count} failed")
