# Import dependencies
from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO

# Init flask
app = Flask(__name__)


@app.route('/detect_memory', methods=['POST'])
def detect_memory():
    # Check if the POST request has an image file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    # Get file from request
    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Process the file 
    result = process_file(file=file)
    
    # Save image path
    result_img_path = f"{result[0].save_dir}\{result[0].path}"
    
    # Return the result
    return jsonify({'result': 'success', 'image_path': result_img_path})


def process_file(file):
    # Init yolo with our trained best.pt
    model = YOLO("./runs/detect/train/weights/best.pt")

    # from ndarray
    im2 = cv2.imread(file)
    results = model.predict(source=im2, save=True, save_txt=True, conf=0.9)  # save predictions as labels

    return results


if __name__ == '__main__':
    app.run(debug=True, port=5030)
