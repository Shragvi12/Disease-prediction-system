import pickle

with open('backend/model/symptoms_list.pkl', 'rb') as f:
    symptoms = pickle.load(f)

print("Symptoms your model recognizes:\n")
print("\n".join(symptoms))
