from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
from custom_transformers import BinaryClassifierTransformer


app = Flask(__name__)

# Load the model (make sure this path is correct)
with open('../flask-app/final_pipeline_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    investment_time_days = float(request.form['investment_time_days'])
    funding_per_milestone = float(request.form['funding_per_milestone'])
    investment_rounds_per_year = float(request.form['investment_rounds_per_year'])
    funding_total_usd = float(request.form['funding_total_usd'])

    # Prepare the input data as a numpy array (1 row, 4 features)
    input_data = np.array([[investment_time_days, funding_per_milestone, investment_rounds_per_year, funding_total_usd]])

    # Make prediction using the model
    prediction = model.predict(input_data)[0]

    # Render result in the same page
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
