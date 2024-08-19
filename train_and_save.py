from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import pickle
import pandas as pd

# Load your dataset
file_path = r'D:/Research/dataset/sample/sample.csv'
data = pd.read_csv(file_path, encoding='latin-1')

# Assuming 'Question' column contains the text and 'Category' column contains the labels
questions = data['Question'].values.astype('U')
categories = data['Category']

# Create and train the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=15000)
X_train_tfidf = tfidf_vectorizer.fit_transform(questions)

# Train the model
model = SVC()
model.fit(X_train_tfidf, categories)

# Save the model
with open('model/best_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the TF-IDF vectorizer
with open('model/tfidf_vectorizer_trained.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)

print("Model and TF-IDF vectorizer saved successfully.")
