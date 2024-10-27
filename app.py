from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
from datetime import datetime
from custom_transformers import BinaryClassifierTransformer

app = Flask(__name__)

# Load the model (make sure this path is correct)
with open('final_pipeline.pkl', 'rb') as f:
    model = joblib.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve data from the form
    founded_at = float(request.form['founded_at'])
    funding_rounds = float(request.form['funding_rounds'])
    funding_total_usd = float(request.form['funding_total_usd'])
    milestones = float(request.form['milestones'])
    relationships = float(request.form['relationships'])
    
  
    # Prepare the input data for the model
    input_data = np.array([[founded_at, funding_rounds, funding_total_usd, milestones, relationships]])

    # Make prediction using the model
    prediction = model.predict(input_data)[0]

    # Render the result in the same page
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
