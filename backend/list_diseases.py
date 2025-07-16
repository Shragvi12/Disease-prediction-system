import pickle

# Load label encoder
with open('backend/model/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Print all disease classes
diseases = sorted(label_encoder.classes_)
print("All Predictable Diseases:\n")
for disease in diseases:
    print("-", disease)

print(f"\nTotal diseases: {len(diseases)}")
