from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('static/best_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        #  Get form inputs
        Age = int(request.form.get('Age'))
        Sex = request.form.get('Sex')
        ChestPainType = request.form.get('ChestPainType')
        RestingBP = int(request.form.get('RestingBP'))
        Cholesterol = int(request.form.get('Cholesterol'))
        FastingBS = int(request.form.get('FastingBS'))
        RestingECG = request.form.get('RestingECG')
        MaxHR = int(request.form.get('MaxHR'))
        ExerciseAngina = request.form.get('ExerciseAngina')
        Oldpeak = float(request.form.get('Oldpeak'))
        ST_Slope = request.form.get('ST_Slope')

        #  Create DataFrame with SAME columns as in training
        input_df = pd.DataFrame([{
            "Age": Age,
            "Sex": Sex,
            "ChestPainType": ChestPainType,
            "RestingBP": RestingBP,
            "Cholesterol": Cholesterol,
            "FastingBS": FastingBS,
            "RestingECG": RestingECG,
            "MaxHR": MaxHR,
            "ExerciseAngina": ExerciseAngina,
            "Oldpeak": Oldpeak,
            "ST_Slope": ST_Slope
        }])

        #  Predict and get probability
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] * 100  # heart disease chance

        # Categorize risk
        if probability < 40:
            risk_label = "Low Risk ✅"
            color = "green"
            msg = "You seem healthy! Maintain a balanced diet and exercise regularly."
        elif 40 <= probability < 70:
            risk_label = "Medium Risk 🟠"
            color = "orange"
            msg = "You have a moderate chance of heart issues. Please consult a doctor for a check-up."
        else:
            risk_label = "High Risk ⚠️"
            color = "red"
            msg = "You have a high chance of heart disease. Seek medical attention immediately."

        result = f"{risk_label} ({probability:.2f}% chance)"

        
        return render_template(
            'result.html',
            prediction=result,
            probability=probability,
            color=color,
            message=msg
        )

    except Exception as e:
        return f"⚠️ Error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
