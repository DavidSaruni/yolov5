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

# Ensure 'uploads' directory exists
uploads_dir = 'uploads'
os.makedirs(uploads_dir, exist_ok=True)

@app.route('/')
def index():
    return render_template('index1.html')

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
    results = perform_inference(img)

    return jsonify({'results': results})

def perform_inference(image):
    # Resize and normalize the image
    image_resized = cv2.resize(image, (320, 320))
    image_resized = image_resized / 255.0
    image_resized = torch.from_numpy(image_resized).permute(2, 0, 1).unsqueeze(0).float()

    # Convert input image to float32
    image_resized = image_resized.float()

    print("Input Image Shape:", image_resized.shape)


    # Declare model with a default value
    model = None

    # Declare prediction with a default value
    prediction = None

    # Check if the model is initialized
    if model is not None:
        # Check the model's datatype and convert input accordingly
        if next(model.parameters()).dtype == torch.float16:
            model = model.float()  # Convert model to float32

        # Run the model
        with torch.no_grad():
            try:
                prediction = model(image_resized)
            except Exception as e:
                print("Error during model inference:", e)
                raise  # Reraise the exception for detailed traceback

    # Print the entire prediction tensor
    print("Prediction:", prediction)

    # Check if prediction is not None before accessing its elements
    if prediction is not None:
        # The detections are in the first element of the tuple
        detections = prediction[0]

        # Rest of the code remains the same
        confidence_threshold = 0.0
        boxes = detections[detections[:, 4] > confidence_threshold]

        print("Filtered Boxes:", boxes)

        results = []
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            result = {
                'class': int(cls),
                'confidence': float(conf),
                'box': [int(x1), int(y1), int(x2), int(y2)]
            }
            results.append(result)
    else:
        # Handle the case when prediction is None
        results = []

    return results

if __name__ == '__main__':
    app.run(debug=True)
