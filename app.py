# NEW CODE - USE THIS
from flask import Flask, request,render_template
import pandas as pd
from sklearn.compose import ColumnTransformer
import numpy as np
import pickle
import os

# Get the directory where the current Python file is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the files
model_path = os.path.join(base_dir, 'save_model', 'decision_tree_regressor_model.pkl')
transformer_path = os.path.join(base_dir, 'save_model', 'transformer.pkl')

try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully.")
    
    with open(transformer_path, 'rb') as file:
        preprocessor = pickle.load(file)
    print("Preprocessor loaded successfully.")

except FileNotFoundError:
    print(f"ERROR: One or both files were not found.")
    print(f"Checked paths: {model_path} and {transformer_path}")
    print("Please verify the 'save_model' folder and its contents exist.")

# Create a Flask app instance
app = Flask(__name__)

@app.route("/")
def home():
    name = 'user name is : --------------------'
    return render_template("index.html")

def get_data():
    return {
        'from': request.form.get('from') or request.args.get('from'),
        'to': request.form.get('to') or request.args.get('to'),
        'flightType': request.form.get('flightType') or request.args.get('flightType'),
        'agency': request.form.get('agency') or request.args.get('agency'),
        'distance': request.form.get('distance') or request.args.get('distance')
    }

@app.route('/predict', methods=['GET', 'POST'])
def price_prediction():
    if request.method == 'POST':
        data = get_data()
        pre_data = preprocessor.transform(pd.DataFrame([data]))
        output = model.predict(pre_data)[0]
        return render_template('index.html', result=output, data=data, data2=pre_data)



    return render_template('index.html', result=output, data=data, data2=data)




if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)