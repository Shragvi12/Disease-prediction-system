import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
from fuzzywuzzy import process

# Load model and required files
with open('backend/model/disease_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('backend/model/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('backend/model/symptoms_list.pkl', 'rb') as f:
    symptoms_list = pickle.load(f)

# Load description and precaution data
desc_df = pd.read_csv('backend/database/disease_description.csv')
prec_df = pd.read_csv('backend/database/disease_precautions.csv')
sev_df = pd.read_csv('backend/database/symptom_severity.csv')

# Function to map user input to valid symptoms
def map_input_to_symptoms(user_input):
    words = user_input.lower().split()
    matched = set()

    for i in range(len(words)):
        for j in range(i + 1, len(words) + 1):
            phrase = "_".join(words[i:j])
            match, score = process.extractOne(phrase, symptoms_list)
            if score >= 85:
                matched.add(match)
    return list(matched)

# Function to calculate severity score
def calculate_severity(symptom_list):
    severity_score = 0
    for symptom in symptom_list:
        row = sev_df[sev_df['Symptom'] == symptom.replace('_', ' ')]
        if not row.empty:
            severity_score += int(row['weight'].values[0])
    return severity_score

# Predict disease from user input
def predict_disease(user_input):
    input_symptoms = map_input_to_symptoms(user_input)

    if not input_symptoms:
        print("No matching symptoms found.")
        return

    print(f"\nMatched Symptoms: {input_symptoms}\n")

    input_data = [1 if symptom in input_symptoms else 0 for symptom in symptoms_list]
    # input_data = label_encoder.transform(symptoms_list)
    # print(input_data)

    # ✅ Use DataFrame to avoid sklearn warning
    input_df = pd.DataFrame([input_data], columns=symptoms_list)
    probs = model.predict_proba(input_df)[0]
    top_indices = np.argsort(probs)[::-1][:3]  # Top 3 predictions

    results = []
    for i in top_indices:
        disease = label_encoder.inverse_transform([i])[0]
        probability = probs[i]
        description = desc_df[desc_df['Disease'] == disease]['Description'].values[0]
        precautions = prec_df[prec_df['Disease'] == disease].values[0][1:]
        severity = calculate_severity(input_symptoms)

        results.append({
            'disease': disease,
            'probability': probability,
            'description': description,
            'precautions': precautions,
            'severity': severity
        })

    most_severe = max(results, key=lambda x: x['severity'])

    # Output
    print("Disease Prediction Results:\n")
    print(f"Most Severe Disease: {most_severe['disease']}")
    print(f"Severity: {most_severe['severity']}")
    print(f"Description: {most_severe['description']}")
    print("Precautions:")
    for p in most_severe['precautions']:
        print(f"- {p}")

    print("\nOther Possible Diseases:")
    for res in results:
        if res['disease'] != most_severe['disease']:
            print(f"- {res['disease']} - Severity: {res['severity']}")

# Run prediction
if __name__ == "__main__":
    user_input = input("Enter your symptoms or description: ")
    predict_disease(user_input)








