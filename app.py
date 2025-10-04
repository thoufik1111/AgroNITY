from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import agronity_test as ag
import numpy as _np # Import numpy at the top for the helper function

# Small helper to convert numpy types to Python native types
def to_py(obj):
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_py(i) for i in obj]
    if _np is not None:
        if isinstance(obj, _np.generic):
            return obj.item()
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    return obj

# App setup
app = Flask(__name__, static_folder='.')
CORS(app)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models and data at startup
preprocessor, clf, reg = ag.load_models()
data_df = ag.load_data()

@app.route('/')
def root():
    # serve the HTML page
    return send_from_directory('.', 'sih.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if preprocessor is None or clf is None or reg is None or data_df is None:
        return jsonify({"error": "Models or data not loaded on server. Check server logs."}), 500

    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Invalid JSON payload"}), 400

    crop = payload.get('crop')
    district = payload.get('district')
    area = payload.get('area')
    soil = payload.get('soil')

    if not all([crop, district, area, soil]):
        return jsonify({"error": "Missing required fields: crop, district, area, soil"}), 400

    try:
        # Call your existing analysis function
        result = ag.analyze_feasibility(preprocessor, clf, reg, data_df, crop, district, area, soil)
        result = to_py(result)
        return jsonify(result)
    except Exception as e:
        # Return a helpful message — check server logs for traceback
        return jsonify({"error": f"Server error during analysis: {str(e)}"}), 500

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    if data_df is None:
        return jsonify({"error": "Dataset not loaded on server."}), 500
    
    payload = request.get_json(silent=True)
    if not payload or 'filename' not in payload:
        return jsonify({"error": "Invalid JSON payload"}), 400
        
    filename = payload.get('filename')
    
    try:
        result = ag.analyze_image(data_df, filename)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Server error during image analysis: {str(e)}"}), 500


if __name__ == '__main__':
    print("Starting AgroNity backend on http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
