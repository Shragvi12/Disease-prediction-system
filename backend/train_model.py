import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle


print("Loading dataset...")


df = pd.read_csv('backend/database/Training.csv')



# Rename target column
df.rename(columns={'Disease': 'prognosis'}, inplace=True)

# Get list of all symptoms
all_symptoms = []
for col in df.columns:
    if col.startswith('Symptom_'):
        all_symptoms.extend(df[col].dropna().unique())

# Remove 'disnan' or 'nan'
all_symptoms = [s.strip() for s in set(all_symptoms) if isinstance(s, str)]

# Create binary symptom matrix
def encode_symptoms(row):
    symptoms_present = set(str(row[col]).strip() for col in df.columns if col.startswith('Symptom_') and pd.notna(row[col]))
    return [1 if symptom in symptoms_present else 0 for symptom in all_symptoms]

X = df.apply(encode_symptoms, axis=1, result_type='expand')
X.columns = all_symptoms

# Encode prognosis
le = LabelEncoder()
y = le.fit_transform(df['prognosis'])

print("Training model...")
model = RandomForestClassifier()
model.fit(X, y)

# Save everything
os.makedirs('backend/model', exist_ok=True)
with open('backend/model/disease_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('backend/model/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

with open('backend/model/symptoms_list.pkl', 'wb') as f:
    pickle.dump(all_symptoms, f)

print("Model, label encoder, and symptoms list saved!")




