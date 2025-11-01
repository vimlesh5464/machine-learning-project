
import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

## import redge regressor and stander scalor pickle
ridge_model=pickle.load(open('models/ridge.pkl', 'rb'))
stander_scaler=pickle.load(open('models/scaler.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=="POST":
        # Collecting data from form
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        # # Convert into numpy array
        data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])

        # # Scale the data
        scaled_data = stander_scaler.transform(data)

        # # Predict
        prediction = ridge_model.predict(scaled_data)[0]

        return render_template("home.html", result=prediction)
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)
