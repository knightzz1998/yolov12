from ultralytics import YOLO

# Initialize a YOLO11n model from a YAML configuration file
model = YOLO("yolov12_p1.yaml")

# If a pre-trained model is available, use it instead
# model = YOLO("model.pt")

# Display model information
model.info()

# Train the model using the COCO8 dataset for 100 epochs
model.train(data="coco8.yaml", epochs=100)