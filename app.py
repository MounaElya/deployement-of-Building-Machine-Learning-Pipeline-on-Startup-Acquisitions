from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
from custom_transformers import BinaryClassifierTransformer  # Ensure this import is correct and available

app = Flask(__name__)

# Load the models at startup
with open('binary_pipeline.pkl', 'rb') as f:
    binary_pipeline = joblib.load(f)

with open('multiclass_clf.pkl', 'rb') as f:
    multiclass_pipeline = joblib.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    founded_at = float(request.form['founded_at'])  # Adjust as needed for date processing
    funding_rounds = float(request.form['funding_rounds'])
    funding_total_usd = float(request.form['funding_total_usd'])
    milestones = float(request.form['milestones'])
    relationships = float(request.form['relationships'])
    
    # Arrange the features into an array for prediction
    X = np.array([[founded_at, funding_rounds, funding_total_usd, milestones, relationships]])

    # Step 1: Transform the input features to get the binary classification probability added as a feature
    X_with_prob = binary_pipeline.transform(X)

    # Step 2: Use the multiclass pipeline to make the final prediction
    multiclass_pred = multiclass_pipeline.predict(X_with_prob)
    
    # Render the prediction result page
    return render_template('result.html', prediction=int(multiclass_pred[0]))

if __name__ == '__main__':
    app.run(debug=True)
