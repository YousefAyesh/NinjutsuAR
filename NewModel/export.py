import json
import os
import torch
from transformers import MobileViTForImageClassification
from torchvision import datasets, transforms

dataset = datasets.ImageFolder("dataset_cropped/train",
                              transform=transforms.ToTensor())
class_names = dataset.classes
print(f"Classes: {class_names}")

model = MobileViTForImageClassification.from_pretrained(
    "apple/mobilevit-xx-small",
    num_labels=len(class_names),
    ignore_mismatched_sizes=True
)
model.config.id2label = {i: name for i, name in enumerate(class_names)}
model.config.label2id = {name: i for i, name in enumerate(class_names)}
model.load_state_dict(torch.load("best_model.pth",
                                  map_location="cpu",
                                  weights_only=False))
model.eval()

dummy = torch.randn(1, 3, 256, 256)

torch.onnx.export(
    model,
    dummy,
    "mobilevit_seals_v3.onnx",
    input_names=["pixel_values"],
    output_names=["logits"],
    opset_version=13,
    dynamic_axes={"pixel_values": {0: "batch_size"}}
)
print("Exported — mobilevit_seals_v3.onnx ready")

# Save class index → name mapping alongside the ONNX model
class_map = {str(i): name for i, name in enumerate(class_names)}
with open("class_names.json", "w") as f:
    json.dump(class_map, f, indent=2)
print(f"Saved class mapping → class_names.json  ({len(class_names)} classes)")

# Quick inference demo showing class name output
print("\nDemo: running a dummy input through the exported ONNX model...")
try:
    import onnxruntime as ort
    import numpy as np

    session = ort.InferenceSession("mobilevit_seals_v3.onnx")
    dummy_np = np.random.randn(1, 3, 256, 256).astype(np.float32)
    logits = session.run(["logits"], {"pixel_values": dummy_np})[0]
    logit_row = logits[0]
    pred_idx = int(np.argmax(logit_row))
    pred_name = class_map[str(pred_idx)]
    print(f"  Predicted index : {pred_idx}")
    print(f"  Predicted class : {pred_name}")
    print(f"  Predicted logit : {float(logit_row[pred_idx]):.6f}")

    topk_idx = np.argsort(logit_row)[-5:][::-1]
    print("\n  Top-5 classes by logit:")
    for rank, idx in enumerate(topk_idx, start=1):
        print(f"    {rank}. {class_map[str(int(idx))]} -> {float(logit_row[idx]):.6f}")
except ImportError:
    print("  (Install onnxruntime to run the demo: pip install onnxruntime)")
