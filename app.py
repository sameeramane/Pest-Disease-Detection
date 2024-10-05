from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
from PIL import Image
import os

app = Flask(__name__)

# Load YOLOv8 models
pest_model = YOLO('pest.pt')
disease_model = YOLO('best(2).pt')

# Function to get info from txt file
def get_info_from_file(detected_objects, txt_file):
    info_dict = {}
    with open(txt_file, 'r') as f:
        for line in f:
            item, info = line.strip().split(": ", 1)
            info_dict[item] = info
    detected_info = [f"{obj}: {info_dict.get(obj, 'No information available.')}" for obj in detected_objects]
    return detected_info

# Route to serve the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle pest detection
@app.route('/detect_pest', methods=['POST'])
def detect_pest():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        file_path = os.path.join('static', file.filename)
        file.save(file_path)

        # Perform pest detection
        results = pest_model(file_path)
        detected_objects = list(set([pest_model.names[int(box.cls)] for box in results[0].boxes]))

        # Get information about detected pests
        info = get_info_from_file(detected_objects, 'pest_info.txt')

        return render_template('index.html', image_path=file_path, detected=detected_objects, info=info, detection_type='Pest')

# Route to handle disease detection
@app.route('/detect_disease', methods=['POST'])
def detect_disease():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        file_path = os.path.join('static', file.filename)
        file.save(file_path)

        # Perform disease detection
        results = disease_model(file_path)
        detected_objects = list(set([disease_model.names[int(box.cls)] for box in results[0].boxes]))

        # Get information about detected diseases
        info = get_info_from_file(detected_objects, 'disease_info.txt')

        return render_template('index.html', image_path=file_path, detected=detected_objects, info=info, detection_type='Disease')

if __name__ == "__main__":
    app.run(debug=True)
