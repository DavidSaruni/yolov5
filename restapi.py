"""
Run a rest API exposing the yolov5s object detection model
"""

import argparse
import io
from PIL import Image

import torch
from flask import Flask, request

app = Flask(__name__)

# Load the model
model = torch.hub.load('DavidSaruni/yolov5', 'yolov5s', pretrained=True)
model.eval()

# Define the API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the file from the request
        file = request.files['file']
        if not file:
            return "No file found", 400

        # Read the image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))

        # Run the model
        results = model(img, size=640)

        # Return the results
        return results.pandas().xyxy[0].to_json(orient="records")

# Run the server
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask api exposing yolov5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    parser.add_argument('--model', default='yolov5s', help='model to run, i.e. --model yolov5s')
    args = parser.parse_args()

    model = torch.hub.load('DavidSaruni/yolov5', args.model)
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat

