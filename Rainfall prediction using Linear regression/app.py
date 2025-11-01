from flask import Flask, request, render_template
import joblib
import numpy as np
app = Flask(__name__)
model=joblib.load("rainfall_model.pkl")

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
  tempmax = float(request.form['tempmax'])
  tempmin = float(request.form['tempmin'])
  humidity = float(request.form['humidity'])
  dew = float(request.form['dew'])

  prediction = model.predict([[tempmax,tempmin,humidity,dew]])[0]
  return render_template('index.html', prediction_text=f"Prediccted Rainfall: {prediction: 2f} mm")

if __name__ == "__main__":
  app.run(debug=True)