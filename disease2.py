import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#Load your dataset
your_dataset_file = "path_to_your_dataset.csv" # Replace 'path_to_your_dataset.csv' with the actual file path
dataset = pd.read_csv(your_dataset_file)
# Step 2: Separate symptoms and diseases
symptoms = dataset['symptoms']#enter your column heading for symptoms
diseases = dataset['diseases']#enter your column heading for diseases

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(symptoms)


classifier = MultinomialNB()
classifier.fit(X, diseases)

def predict_disease(input_symptoms):
    # Vectorize the input symptoms
    input_vector = vectorizer.transform([input_symptoms])

    # Make predictions
    predicted_disease = classifier.predict(input_vector)

    return predicted_disease[0]


while True:
    user_input = input("Enter your symptoms (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    predicted_disease = predict_disease(user_input)
    print("Predicted Disease:", predicted_disease)
