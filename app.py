import pandas as pd
import numpy as np
import tensorflow_addons
from tensorflow_addons.metrics import F1Score
import joblib
from flask import Flask, request, jsonify

# App Initialization
app = Flask(__name__)

# Load The Models
with open('Processer.pkl', 'rb') as file_1:
  model_pipeline = joblib.load(file_1)

def f1_score_macro():
    return tfa.metrics.F1Score(num_classes = 1, average = 'macro', name = 'f1_score_macro', threshold = 0.5)

from tensorflow.keras.models import load_model
model_ann = load_model('PredictModel.h5', custom_objects = {'F1score' : f1_score_macro})

# Route : Homepage
@app.route('/')
def home():
    return '<h1> It Works! </h1>'

@app.route('/predict', methods = ['POST'])
def predict():
    args = request.json

    data_inf = {
        'customerID' : args.get('customerID'),
        'gender' : args.get('gender'),
        'SeniorCitizen' : args.get('SeniorCitizen'),
        'Partner' : args.get('Partner'),
        'Dependents' : args.get('Dependents'),
        'tenure' : args.get('tenure'),
        'PhoneService' : args.get('PhoneService'),
        'MultipleLines' : args.get('MultipleLines'),
        'InternetService' : args.get('InternetService'),
        'OnlineSecurity' : args.get('OnlineSecurity'),
        'OnlineBackup' : args.get('OnlineBackup'),
        'DeviceProtection' : args.get('DeviceProtection'),
        'TechSupport' : args.get('TechSupport'),
        'StreamingTV' : args.get('StreamingTV'),
        'StreamingMovies' : args.get('StreamingMovies'),
        'Contract' : args.get('Contract'),
        'PaperlessBilling' : args.get('PaperlessBilling'),
        'PaymentMethod' : args.get('PaymentMethod'),
        'MonthlyCharges' : args.get('MonthlyCharges'),
        'TotalCharges' : args.get('TotalCharges'),
    }

    print('[DEBUG] Data Inference :', data_inf)

    # Transform Inference-set
    data_inf = pd.DataFrame([data_inf])
    data_inf_transform = model_pipeline.transform(data_inf)

    y_pred_inf = model_ann.predict(data_inf_transform)
    y_pred_inf = np.where(y_pred_inf >= 0.5, 1, 0)

    if y_pred_inf == 0:
        label = 'No'
    else:
        label = 'Yes'
    
    print('[DEBUG] Result : ', y_pred_inf, label, '\n')

    response = jsonify(
        result = str(y_pred_inf),
        label_names = label
    )

    return response

if __name__ == '__main__':
    app.run(host = '0.0.0.0')