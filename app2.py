# app.py
from flask import Flask, render_template, request, jsonify
import torch
import cv2
# Update this line based on your actual module structure and model class
from yolov5.models.yolo import Model as YOLOv5Model
import os
import numpy as np

app = Flask(__name__)

# Load the YOLOv5 model
model_path = '/home/captain/Desktop/Cocoa Detection/yolov5/runs/train/exp2/weights/best.pt'
num_classes = 3  # Update with the actual number of classes
model_config_path = '/home/captain/Desktop/Cocoa Detection/yolov5/models/yolov5s.yaml'  # Update with your model configuration path

model = YOLOv5Model(cfg=model_config_path, ch=3, nc=num_classes)
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

    if img is None:
        raise ValueError("Failed to read the image. Check the image file.")
    
    # Print the type and shape of the input image
    print("Input image type:", type(img))
    print("Input image shape:", img.shape)


    detections = perform_inference(img, model)

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
        if detection is not None:
            json_detection = {
                'confidence': detection.confidence if hasattr(detection, 'confidence') else None,
                'class': detection.class_name if hasattr(detection, 'class_name') else None,
                'box': detection.box if hasattr(detection, 'box') else None
            }
            json_results.append(json_detection)
    return json_results



def perform_inference(image, model):
    # Print the type and shape of the input image
    print("Input image type in perform_inference:", type(image))
    print("Input image shape in perform_inference:", image.shape)

    # Ensure img is a valid NumPy array
    if not isinstance(image, np.ndarray):
        raise ValueError("Input image is not a valid NumPy array.")

    # Convert the NumPy array to a PyTorch tensor
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure correct color format
    image = torch.from_numpy(image).permute(2, 0, 1).float()

    # Print image shape for debugging
    print("Image shape before resize:", image.shape)

    # Resize and normalize the image
    try:
        image_resized = cv2.resize(image.numpy(), (640, 640))  # Convert back to NumPy for resize
    except Exception as e:
        print("Error during resize:", e)
        raise

    image_resized = image_resized / 255.0
    image_resized = torch.from_numpy(image_resized).permute(2, 0, 1).unsqueeze(0).float()

    # Check the model's data type and convert input accordingly
    if next(model.parameters()).dtype == torch.float16:
        # Convert input to half precision if the model uses float16
        image_resized = image_resized.half()
    else:
        # Convert input to float32 if the model uses float32
        image_resized = image_resized.float()

    # Run the model
    with torch.no_grad():
        try:
            prediction = model(image_resized)
        except RuntimeError as e:
            if 'slow_conv2d_cpu' in str(e):
                # If "slow_conv2d_cpu" not implemented for 'Half' error occurs, try converting the model to float32
                model = model.to(torch.float32)
                image_resized = image_resized.float()
                prediction = model(image_resized)
            else:
                # Re-raise the exception if it's a different error
                raise e

    # The detections are in the first element of the tuple
    detections = prediction[0]

    # Filter out predictions with low confidence
    confidence_threshold = 0.5
    filtered_boxes = detections[detections[:, 4] > confidence_threshold]

    # Extract results for returning
    results = []
    for box in filtered_boxes:
        box_np = box.cpu().numpy()  # Convert to numpy array

        # Directly access the values in the box tensor
        x_center, y_center, width, height, conf, cls = box_np[:6]

        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        result = {
            'class': int(cls),
            'confidence': float(conf),
            'box': [x1, y1, x2, y2]
        }
        results.append(result)

    return results



if __name__ == '__main__':
    app.run(debug=True)
