from flask import Flask, render_template, request
import pandas as pd
import joblib

# Load trained model
model = joblib.load("rfc_model.pkl")

# Load dataset to get feature names
data = pd.read_csv("creditcard.csv")
columns = list(data.drop(['Class'], axis=1).columns)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html', columns=columns)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = []
    for col in columns:
        value = float(request.form[col])
        input_data.append(value)

    df = pd.DataFrame([input_data], columns=columns)
    prediction = model.predict(df.values)[0]
    
    result = "Fraudulent Transaction" if prediction == 1 else "Valid Transaction"
    
    return render_template('index.html', columns=columns, prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
