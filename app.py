from flask import Flask, request, jsonify, render_template
import base64, json
from io import BytesIO
from model import MyModel
import numpy as np
import os
from math import floor
from werkzeug.utils import secure_filename

# declare constants
HOST = '0.0.0.0'
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# initialize flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Read model to keep it ready all the time
model = MyModel('cnn2.pth', 'cpu')

CLASS_MAPPING = ['metal', 'plastic', 'cardboard', 'paper', 'trash', 'glass']

# Make the prediction human-readable
img_class_map = None
with open('index_to_name.json') as f:
    img_class_map = json.load(f)

def render_prediction(index):
    stridx = str(index)
    class_name = 'Unknown'
    if img_class_map is not None:
        if stridx in img_class_map:
            class_name = img_class_map[stridx]
    return class_name

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/infer', methods=['GET', 'POST'])
def success():
    if request.method == 'POST':
        # Check if file is in request
        if 'file' not in request.files:
            return render_template('index.html'), 400
        
        f = request.files['file']
        
        # Check if file was selected
        if f.filename == '':
            return render_template('index.html'), 400
        
        # Check if file type is allowed
        if not allowed_file(f.filename):
            return "Invalid file type. Please upload PNG, JPG, or JPEG", 400
        
        try:
            # Secure the filename
            filename = secure_filename(f.filename)
            
            # Create full save path
            saveLocation = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the file
            f.save(saveLocation)
            
            # Make inference
            inference, confidence = model.infer(saveLocation)
            
            # Get class name
            class_name = render_prediction(inference)
            
            # Make a percentage with 2 decimal points
            confidence = floor(confidence * 10000) / 100
            
            # Delete file after making an inference
            if os.path.exists(saveLocation):
                os.remove(saveLocation)
            
            # Respond with the inference
            return render_template('inference.html', name=class_name, confidence=confidence)
        
        except Exception as e:
            # Clean up file if it exists
            if 'saveLocation' in locals() and os.path.exists(saveLocation):
                os.remove(saveLocation)
            
            print(f"Error during inference: {str(e)}")
            return f"An error occurred: {str(e)}", 500
    
    # If GET request, redirect to home
    return render_template('index.html')

if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 80))
    app.run(host='0.0.0.0', port=port, debug=True)