# app.py
from flask import Flask, render_template, request, jsonify
import torch
import cv2
from yolov5.models.yolo import Model  # Update with your actual module structure
import os

app = Flask(__name__)

# Load the YOLOv5 model
model_path = '/home/captain/Desktop/Cocoa Detection/yolov5/runs/train/exp2/weights/best.pt'
num_classes = 3  # Update with the actual number of classes
model_config_path = '/home/captain/Desktop/Cocoa Detection/yolov5/models/yolov5s.yaml'  # Update with your model configuration path

model = Model(cfg=model_config_path, ch=3, nc=num_classes)
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
# Check if the loaded checkpoint is a state_dict or the whole model
if isinstance(checkpoint, dict):
    # If 'model' key exists, load the state_dict
    if 'model' in checkpoint:
        model = checkpoint['model']
    else:
        model = checkpoint
else:
    model = checkpoint  # Assume the entire model is loaded

model.eval()

model_path = model_path

# Load the model with custom weights
model = torch.hub.load('yolov5', 'custom', path=model_path, source='local')

# Set the model to evaluation mode
model.eval()

# Ensure 'uploads' directory exists
uploads_dir = 'uploads'
os.makedirs(uploads_dir, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the uploaded image
    image_path = os.path.join(uploads_dir, 'input_image.jpg')
    file.save(image_path)

    # Perform inference
    img = cv2.imread(image_path)
    detections = model(img)

    # Ensure detections is a list
    if not isinstance(detections, list):
        detections = [detections]

    # Convert results to a JSON-serializable format
    results_json = jsonify_results(detections)

    return jsonify({'results': results_json})


def jsonify_results(results):
    # Convert the Detections objects to a serializable format (list of dictionaries)
    json_results = []
    for detection in results:
        json_detection = {
            'class': detection['class'],
            'confidence': detection['confidence'],
            'box': detection['box']
        }
        json_results.append(json_detection)
    return json_results


def perform_inference(image):
    # Resize and normalize the image
    image_resized = cv2.resize(image, (640, 640))
    image_resized = image_resized / 255.0
    image_resized = torch.from_numpy(image_resized).permute(2, 0, 1).unsqueeze(0).float()

    # Check the model's data type and convert input accordingly
    if next(model.parameters()).dtype == torch.float16:
        image_resized = image_resized.half()  # Convert input to half precision

    # Run the model
    with torch.no_grad():
        prediction = model(image_resized)

    # The detections are in the first element of the tuple
    detections = prediction[0]

    # Filter out predictions with low confidence
    confidence_threshold = 0.5
    boxes = detections[detections[:, 4] > confidence_threshold]

    # Extract results for returning
    results = []
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        result = {
            'class': int(cls),
            'confidence': float(conf),
            'box': [int(x1), int(y1), int(x2), int(y2)]
        }
        results.append(result)

    return results  # Now returns a list of detection results


if __name__ == '__main__':
    app.run(debug=True)
