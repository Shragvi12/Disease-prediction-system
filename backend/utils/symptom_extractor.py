import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import pandas as pd

def extract_symptoms(user_input):
    df = pd.read_csv("database/Training.csv")
    symptoms_list = df.columns[:-1].str.replace('_', ' ').str.lower().tolist()
    tokens = word_tokenize(user_input.lower())
    matched = [s.replace(' ', '_') for s in symptoms_list if all(word in tokens for word in s.split())]
    return matched
