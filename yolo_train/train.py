from ultralytics import YOLO

# Load a model
model = YOLO('yolo11n.yaml')  # build a new model from scratch

# Use the model
results = model.train(data='D:/Python/datasets/DeepWeed/data.yaml', epochs=8, imgsz=640)  # train the model