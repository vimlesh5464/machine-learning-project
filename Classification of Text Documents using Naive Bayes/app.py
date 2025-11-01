from flask import Flask, request, render_template
import joblib

app=Flask(__name__)

#model load
model = joblib.load("model/nb_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

@app.route("/")
def Home():
  return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict_ro():
   text = request.form["text"]
   vect_text = vectorizer.transform([text])
   prediction = model.predict(vect_text)
   return render_template("index.html",prediction_text=f'Predicted class: {prediction[0]}')

if __name__ == "__main__":
    app.run(debug=True)