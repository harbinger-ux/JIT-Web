import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from inference import MedCLIPPredictor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

predictor = MedCLIPPredictor()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
        
    file = request.files['image']
    plane = request.form.get('plane')
    threshold_str = request.form.get('threshold', '0.5')
    
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
        
    if plane not in ['axial', 'sagittal']:
        return jsonify({'error': 'Invalid plane selected.'}), 400

    try:
        threshold = float(threshold_str)
    except ValueError:
        return jsonify({'error': 'Invalid threshold value.'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Pass the dynamic threshold to the predictor
            results = predictor.predict(filepath, plane, threshold=threshold)
            os.remove(filepath) 
            return jsonify({'success': True, 'plane': plane, 'results': results})
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)