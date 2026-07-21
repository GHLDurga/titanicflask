from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

with open("model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    pclass = int(request.form["pclass"])
    sex = int(request.form["sex"])
    age = float(request.form["age"])
    sibsp = int(request.form["sibsp"])
    parch = int(request.form["parch"])
    fare = float(request.form["fare"])
    embarked = int(request.form["embarked"])

    features = [[
        pclass,
        sex,
        age,
        sibsp,
        parch,
        fare,
        embarked
    ]]

    prediction = model.predict(features)[0]

    probability = model.predict_proba(features)[0]

    confidence = round(max(probability) * 100, 2)

    if prediction == 1:
        result = "Passenger Survived 🚢"
        message = "Congratulations! The passenger is likely to survive."
    else:
        result = "Passenger Did Not Survive ❌"
        message = "Unfortunately, the passenger is unlikely to survive."

    return render_template(
        "result.html",
        result=result,
        message=message,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)