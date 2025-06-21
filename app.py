from flask import Flask, request, jsonify
import pickle
import os
from utils import process_input_data, predict_with_model

app = Flask(__name__)

#  load one model (adjust file name as needed)
MODEL_PATH = os.path.join("models", "xgb_beth_best.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def home():
    return "API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    model_type = 'xgb'
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.endswith('.csv'):
        try:
            # Save the uploaded file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # Process the input data

            # Expecting process_input_data to return scaled_data and original_info_df
            processed_data, original_info_df = process_input_data(filepath)
            
            # Get predictions

            predictions = predict_with_model(processed_data, model_type)
            
            # Convert predictions to a format suitable for display
            # and include timestamp and userId
            results = []
            anomaly_count = 0
            total_predictions = len(predictions)

            for i, pred in enumerate(predictions):
                results.append({
                    'timestamp': original_info_df.iloc[i]['timestamp'],
                    'userId': original_info_df.iloc[i]['userId'],
                    'prediction': int(pred)
                })
                if int(pred) == 1:
                    anomaly_count += 1
            
            anomaly_rate = 0
            if total_predictions > 0:
                anomaly_rate = (anomaly_count / total_predictions) * 100
            
            # print(results) # For server-side debugging
            return jsonify({
                'success': True,
                'results': results,
                'anomaly_rate': float(f"{anomaly_rate:.2f}") # Format to 2 decimal places
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == "__main__":
    app.run(debug=True,port=5000)