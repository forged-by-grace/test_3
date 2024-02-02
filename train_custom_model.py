from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO('yolov8m.pt')

# Train the model using the 'data_custom.yaml' dataset for 3 epochs
results = model.train(data='./data_custom.yaml', epochs=3, batch=8, imgsz=640, workers=1)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
results = model('./training/memory/out1.png')
