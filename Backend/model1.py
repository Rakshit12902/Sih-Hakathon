import pickle
import pandas as pd

# Load model
model = pickle.load(open("best_model.pkl", "rb"))

# Example new patient (use same column names as training features)
new_patient = {
    "Age": 55,
    "Sex": "M",               # depends on how 'Sex' values are in your CSV (check exact strings)
    "ChestPainType": "ATA",   # use the same categories as present in the dataset
    "RestingBP": 140,
    "Cholesterol": 220,
    "FastingBS": 0,
    "RestingECG": "Normal",
    "MaxHR": 150,
    "ExerciseAngina": "N",
    "Oldpeak": 1.9,
    "ST_Slope": "Flat"
}

df_new = pd.DataFrame([new_patient])
pred = model.predict(df_new)            # 0 or 1
proba = model.predict_proba((df_new))[:,1] # probability of class 1 (disease)
result = float(proba[0] * 100)
rounded_result = round(result, 2)

print("Predicted class:", int(pred[0]))
print("Probability of heart disease:")
print(rounded_result) 
