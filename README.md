# NinjutsuAR
A computer vision model that allows you to do a sequence of Naruto hand signs to do Jutsu's
Midpoint CheckIn Presentation April 8th
Final Project Submission Report/Submission April 29th 


Phase 1 — Dataset collection and model training (start here)
This is the foundation everything else depends on. You need a solid dataset of the 12 Naruto hand seals + the JJK poses. Practically, you'll want 200–500 photos per class at minimum, shot under varied lighting, hand sizes, and angles. Roboflow is great for annotation and augmentation. Use MediaPipe to extract the 21 3D joint coordinates as input features alongside the raw images — this gives MobileViT-XXS both visual and skeletal cues, which really helps disambiguate seals that are similar (like Bird vs Rat, which differ only in subtle finger angles).
For fine-tuning MobileViT-XXS, start from a HuggingFace pretrained checkpoint and train in PyTorch, then export to ONNX. Test classification accuracy thoroughly before moving to Unity.

Phase 2 — Unity + Niantic ARDK setup
Get ARDK running early on a test device (preferably iOS for better ARKit depth). Stub out placeholder effects (a simple colored sphere as your "fireball") before building the real VFX. This lets you validate the pipeline — camera → inference → effect spawn — before you invest in polished particle systems.

Phase 3 — Sequence buffer and gesture state machine
The sliding buffer with debounce is surprisingly tricky to get right. I'd implement it as a simple ring buffer in C# with three parameters you tune manually: a hold threshold (how many frames you need to confirm a sign), a debounce window (frames to suppress re-adding the same sign), and an inactivity timeout (frames of no detection that reset the buffer). Matching is just a linear scan against your sequence dictionary once the buffer length hits a threshold.

Phase 4 — VFX and person segmentation
Domain Expansion is your showstopper demo moment — skybox swap + YOLOv8-seg overlays on all detected people. Build it last once the recognition pipeline is solid. For YOLOv8-seg, run it as a separate Python backend on your laptop and stream masks to Unity over a local socket during development, then optimize for on-device later if time allows.
Biggest risks to watch for:
The hardest part will likely be latency — running MobileViT-XXS via Unity Sentis + YOLOv8-seg + Niantic ARDK all on-device is a lot. Profile early and aggressively quantize your models (INT8 ONNX). The second risk is dataset quality — hand seals are subtle, so bad training data will tank your classifier regardless of model choice. Budget significant time for data collection and cleaning.

