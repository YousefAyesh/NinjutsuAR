import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("mobilevit_seals.onnx")

# Print input/output names and shapes
for inp in sess.get_inputs():
    print("Input:", inp.name, inp.shape)
for out in sess.get_outputs():
    print("Output:", out.name, out.shape)

# Run a dummy inference to see raw output values
shape = [s if isinstance(s, int) else 1 for s in sess.get_inputs()[0].shape]
dummy = np.zeros(shape, dtype=np.float32)
result = sess.run(None, {sess.get_inputs()[0].name: dummy})
print("Raw output (logits):", result[0])

# Check for embedded class labels in model metadata
meta = sess.get_modelmeta().custom_metadata_map
if meta:
    print("Model metadata:", meta)
else:
    print("No embedded class labels in metadata.")