from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
from fuzzywuzzy import process

app = Flask(__name__)

# Load model and resources
with open('backend/model/disease_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('backend/model/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('backend/model/symptoms_list.pkl', 'rb') as f:
    symptoms_list = pickle.load(f)

desc_df = pd.read_csv('backend/database/disease_description.csv')
prec_df = pd.read_csv('backend/database/disease_precautions.csv')
sev_df = pd.read_csv('backend/database/symptom_severity.csv')

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

def calculate_severity(symptom_list):
    severity_score = 0
    for symptom in symptom_list:
        row = sev_df[sev_df['Symptom'] == symptom.replace('_', ' ')]
        if not row.empty:
            severity_score += int(row['weight'].values[0])
    return severity_score

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['text']
    input_symptoms = map_input_to_symptoms(user_input)

    if not input_symptoms:
        return render_template('result.html', error="No matching symptoms found. Try again with different wording.")

    input_data = [1 if symptom in input_symptoms else 0 for symptom in symptoms_list]
    input_df = pd.DataFrame([input_data], columns=symptoms_list)

    probs = model.predict_proba(input_df)[0]
    top_indices = np.argsort(probs)[::-1][:3]

    results = []
    for i in top_indices:
        disease = label_encoder.inverse_transform([i])[0]
        probability = probs[i]
        description = desc_df[desc_df['Disease'] == disease]['Description'].values[0]
        precautions = list(prec_df[prec_df['Disease'] == disease].values[0][1:])
        severity = calculate_severity(input_symptoms)

        results.append({
            'disease': disease,
            'probability': round(float(probability), 2),
            'description': description,
            'precautions': precautions,
            'severity': severity
        })

    most_severe = max(results, key=lambda x: x['severity'])
    return render_template('result.html', most_severe=most_severe, all_diseases=results)

if __name__ == '__main__':
    app.run(debug=True)