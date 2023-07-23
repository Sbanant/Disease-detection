import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Step 1: Load your dataset
your_dataset_file = "C:/Users/anant/Downloads/archive/Symptom2Disease.csv" # Replace 'path_to_your_dataset.csv' with the actual file path
dataset = pd.read_csv(your_dataset_file)
# Step 2: Separate symptoms and diseases
symptoms = dataset['text']
diseases = dataset['label']

# Step 3: Vectorize symptoms using bag-of-words representation
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(symptoms)

# Step 4: Train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X, diseases)

def predict_disease(input_symptoms):
    # Vectorize the input symptoms
    input_vector = vectorizer.transform([input_symptoms])

    # Make predictions
    predicted_disease = classifier.predict(input_vector)

    return predicted_disease[0]

# Example usage
while True:
    user_input = input("Enter your symptoms (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    predicted_disease = predict_disease(user_input)
    print("Predicted Disease:", predicted_disease)
