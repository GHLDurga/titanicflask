from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model, scaler and feature columns
model = joblib.load("telecom_model.joblib")
scaler = joblib.load("scaler.joblib")
feature_columns = joblib.load("feature_columns.joblib")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Create a dictionary with all features initialized to 0
    data = {col: 0 for col in feature_columns}

    # Numerical and binary features

    data["Gender"] = int(request.form["Gender"])
    data["Age"] = int(request.form["Age"])
    data["Married"] = int(request.form["Married"])
    data["Number of Dependents"] = int(
        request.form["Number of Dependents"]
    )

    data["Latitude"] = float(request.form["Latitude"])
    data["Longitude"] = float(request.form["Longitude"])

    data["Number of Referrals"] = int(
        request.form["Number of Referrals"]
    )

    data["Tenure in Months"] = int(
        request.form["Tenure in Months"]
    )

    data["Phone Service"] = int(
        request.form["Phone Service"]
    )

    data["Avg Monthly Long Distance Charges"] = float(
        request.form["Avg Monthly Long Distance Charges"]
    )

    data["Multiple Lines"] = int(
        request.form["Multiple Lines"]
    )

    data["Internet Service"] = int(
        request.form["Internet Service"]
    )

    data["Avg Monthly GB Download"] = float(
        request.form["Avg Monthly GB Download"]
    )

    data["Online Security"] = int(
        request.form["Online Security"]
    )

    data["Online Backup"] = int(
        request.form["Online Backup"]
    )

    data["Device Protection Plan"] = int(
        request.form["Device Protection Plan"]
    )

    data["Premium Tech Support"] = int(
        request.form["Premium Tech Support"]
    )

    data["Streaming TV"] = int(
        request.form["Streaming TV"]
    )

    data["Streaming Movies"] = int(
        request.form["Streaming Movies"]
    )

    data["Streaming Music"] = int(
        request.form["Streaming Music"]
    )

    data["Unlimited Data"] = int(
        request.form["Unlimited Data"]
    )

    data["Paperless Billing"] = int(
        request.form["Paperless Billing"]
    )

    monthly_charge = float(
        request.form["Monthly Charge"]
    )

    total_charges = float(
        request.form["Total Charges"]
    )

    data["Monthly Charge"] = monthly_charge
    data["Total Charges"] = total_charges

    data["Total Refunds"] = float(
        request.form["Total Refunds"]
    )

    data["Total Extra Data Charges"] = float(
        request.form["Total Extra Data Charges"]
    )

    data["Total Long Distance Charges"] = float(
        request.form["Total Long Distance Charges"]
    )

    data["Total Revenue"] = float(
        request.form["Total Revenue"]
    )

    data["avg_monthly_spend"] = float(
        request.form["avg_monthly_spend"]
    )

    data["spend_change"] = float(
        request.form["spend_change"]
    )

    # Derived features

    data["high_bill"] = int(
        request.form["high_bill"]
    )

    data["age_group_Senior"] = int(
        request.form["age_group_Senior"]
    )

    data["age_group_Young"] = int(
        request.form["age_group_Young"]
    )

    # One-hot encoding: Internet Type

    internet_type = request.form["internet_type"]

    if internet_type == "Cable":
        data["Internet Type_Cable"] = 1

    elif internet_type == "DSL":
        data["Internet Type_DSL"] = 1

    elif internet_type == "Fiber Optic":
        data["Internet Type_Fiber Optic"] = 1

    # One-hot encoding: Contract

    contract = request.form["contract_type"]

    if contract == "Month-to-Month":
        data["Contract_Month-to-Month"] = 1

    elif contract == "One Year":
        data["Contract_One Year"] = 1

    elif contract == "Two Year":
        data["Contract_Two Year"] = 1

    # One-hot encoding: Payment Method

    payment_method = request.form["payment_method"]

    if payment_method == "Bank Withdrawal":
        data["Payment Method_Bank Withdrawal"] = 1

    elif payment_method == "Credit Card":
        data["Payment Method_Credit Card"] = 1

    elif payment_method == "Mailed Check":
        data["Payment Method_Mailed Check"] = 1

    # Convert dictionary to DataFrame

    input_df = pd.DataFrame([data])

    # Rearrange columns

    input_df = input_df[feature_columns]

    # Scale data

    input_scaled = scaler.transform(input_df)

    # Predict

    prediction = model.predict(input_scaled)

    predicted_class = prediction[0]

    if predicted_class == 0:
        prediction_text = "Customer is likely to churn."

    elif predicted_class == 1:
        prediction_text = "Customer has recently joined."

    elif predicted_class == 2:
          prediction_text = "Customer is likely to stay."

    else:
          prediction_text = "Unknown prediction."
    return render_template(
        "index.html",
        prediction_text=prediction_text
    )


if __name__ == "__main__":
    app.run(debug=True)