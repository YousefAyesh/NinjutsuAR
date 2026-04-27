from roboflow import Roboflow

rf = Roboflow(api_key="ASkGpINzJ68QGp6wPCO9")
project = rf.workspace("ammans-workspace").project("ninjutsuar-4152")
version = project.version(6)
dataset = version.download("folder")

print("Dataset downloaded successfully")
