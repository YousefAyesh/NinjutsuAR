import argparse
import json
import os
import random

import cv2
import numpy as np
import onnxruntime as ort


def preprocess_image(image_path):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (256, 256), interpolation=cv2.INTER_AREA)

    image = image_rgb.astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image


def softmax(logits):
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def pick_random_image(dataset_root):
    exts = (".jpg", ".jpeg", ".png")
    candidates = []
    for root, _, files in os.walk(dataset_root):
        for filename in files:
            if filename.lower().endswith(exts):
                candidates.append(os.path.join(root, filename))

    if not candidates:
        raise FileNotFoundError(
            f"No images found under '{dataset_root}'. "
            "Provide an image path or check your dataset_cropped folder."
        )
    return random.choice(candidates)


def build_class_map_from_dataset(dataset_root):
    # Prefer ImageFolder-style train split if present.
    class_root = os.path.join(dataset_root, "train")
    if not os.path.isdir(class_root):
        class_root = dataset_root

    classes = sorted(
        [d for d in os.listdir(class_root) if os.path.isdir(os.path.join(class_root, d))]
    )
    if not classes:
        raise FileNotFoundError(
            "Could not infer class names from dataset folders. "
            "Run export.py first or ensure dataset_cropped/train/<class>/ exists."
        )
    return {str(i): name for i, name in enumerate(classes)}


def main():
    parser = argparse.ArgumentParser(description="Run ONNX inference with class names + logits output")
    parser.add_argument("image", nargs="?", default=None, help="Path to input image")
    parser.add_argument("--model", default="mobilevit_seals.onnx", help="Path to ONNX model")
    parser.add_argument("--classes", default="class_names.json", help="Path to class mapping JSON")
    parser.add_argument(
        "--dataset-root",
        default="dataset_cropped",
        help="Dataset root used for random image selection when no image path is provided",
    )
    parser.add_argument("--topk", type=int, default=5, help="Number of top predictions to show")
    parser.add_argument(
        "--unknown-threshold",
        type=float,
        default=0.55,
        help="Report UNKNOWN when top-1 probability is below this threshold (default: 0.55)",
    )
    args = parser.parse_args()

    image_path = args.image if args.image else pick_random_image(args.dataset_root)

    if os.path.exists(args.classes):
        with open(args.classes, "r") as f:
            class_map = json.load(f)
    else:
        print(
            f"Warning: '{args.classes}' not found. Falling back to class names inferred "
            f"from '{args.dataset_root}'."
        )
        class_map = build_class_map_from_dataset(args.dataset_root)

    session = ort.InferenceSession(args.model)
    input_tensor = preprocess_image(image_path)
    logits = session.run(["logits"], {"pixel_values": input_tensor})[0][0]

    probs = softmax(logits)
    topk = max(1, min(args.topk, len(logits)))
    topk_idx = np.argsort(logits)[-topk:][::-1]
    best_idx = int(topk_idx[0])
    best_name = class_map.get(str(best_idx), f"class_{best_idx}")
    best_prob = float(probs[best_idx])

    print(f"Image: {image_path}")
    print(f"Model: {args.model}")
    print(
        f"Top-1: class={best_name} | index={best_idx} | "
        f"logit={float(logits[best_idx]):.6f} | prob={best_prob:.4f}"
    )
    is_unknown = best_prob < args.unknown_threshold
    verdict = "UNKNOWN" if is_unknown else best_name
    print(
        f"Final prediction with threshold {args.unknown_threshold:.2f}: {verdict}"
    )

    print("\nTop predictions:")
    for rank, idx in enumerate(topk_idx, start=1):
        idx_int = int(idx)
        name = class_map.get(str(idx_int), f"class_{idx_int}")
        print(
            f"  {rank}. class={name} | index={idx_int} | "
            f"logit={float(logits[idx_int]):.6f} | prob={float(probs[idx_int]):.4f}"
        )


if __name__ == "__main__":
    main()
