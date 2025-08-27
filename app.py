from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Custom transformer class (needed if pipeline uses it)
class OutlierHandler:
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X

# Load the saved model
with open("diabetes_prediction_pipeline.pkl", "rb") as f:
    data = pickle.load(f)

# If pickle contains a tuple
if isinstance(data, tuple):
    model = data[0]
else:
    model = data

# Mapping for predictions
class_labels = {
    0: "No Diabetes",
    1: "Pre-Diabetes",
    2: "Diabetes"
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from form
        age = float(request.form["age"])
        bmi = float(request.form["bmi"])
        hba1c = float(request.form["hba1c"])
        tg = float(request.form["tg"])

        # Validation
        if age <= 0 or bmi <= 0 or hba1c <= 0 or tg <= 0:
            return render_template("index.html", prediction="Invalid input values!")

    except ValueError:
        return render_template("index.html", prediction="Please enter only numeric values!")

    # Match features with training order
    features = [[hba1c, bmi, age, tg]]
    prediction = model.predict(features)[0]

    # Convert numeric prediction to label
    result = class_labels.get(prediction, "Unknown")

    # Send inputs + result to HTML
    return render_template(
        "index.html",
        prediction=result,
        age=age,
        bmi=bmi,
        hba1c=hba1c,
        tg=tg
    )



if __name__ == "__main__":
    app.run(debug=True)
