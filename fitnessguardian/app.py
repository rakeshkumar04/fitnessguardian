from flask import Flask, render_template, request, jsonify
import joblib
from datetime import datetime

app = Flask(__name__)

model = joblib.load('models/multiple_regression_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Parse input timestamp from form data
        timestamp_str = request.form['timestamp']
        
        # Convert timestamp string to datetime object
        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        
        # Extract features from the timestamp
        hour = timestamp.hour
        day_of_week = timestamp.weekday() 
        month = timestamp.month
        
        # Predict with the loaded model
        prediction = model.predict([[hour, day_of_week, month]])

        

        
        # Prepare the prediction result
        result = {
            'Heartbeat': prediction[0][0],
            'SpO2': prediction[0][1],
            'Temperature': prediction[0][2]
        }
        
        # Return the prediction result as JSON
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
